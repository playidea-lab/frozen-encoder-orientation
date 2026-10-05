"""DINOv3 마지막 N개 블록을 풀고 rotfix 구성을 학습한다 (u-dinovq-rotfix 의 자식).

무엇을 가르는가: PA 49~50 천장이 (a) 동결된 특징의 정보 한계인가, (b) 토큰 분류 목적함수가
정보를 못 꺼내는 것인가. 자매 실험 u-dinovq-geoloss 가 (b)를 시험한다.

두 가지 손질이 필요하다.
  1. 모델이 인코더를 `with torch.no_grad():` 로 감싼다. 파라미터의 requires_grad 만 켜서는
     기울기가 안 흐른다. 인스턴스의 get_intermediate_layers 를 감싸 torch.enable_grad() 를
     씌운다 — 클래스가 아니라 인스턴스에만 붙이므로 state_dict 키는 그대로다.
  2. 트레이너가 만든 optimizer 는 전 파라미터에 같은 lr 을 준다. 사전학습 백본에 1e-4 는
     너무 크므로 파라미터 그룹을 나눠 재구성하고 lr_scheduler 도 새 optimizer 에 다시 건다.

CQ_PARAMS: seed, lr_step, epochs, unfreeze_blocks, backbone_lr_mult, batch, workers, head_axis, rot_from, init_from, resume
(unfreeze_blocks=0 이면 인코더 완전 동결 — 논문의 'both, frozen' 행과 시드 반복이 이 경로다.)
"""

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

# 데이터로더가 병목이다(GPU 0%). 워커마다 OpenCV가 스레드를 띄워 경합하므로 끈다.
cv2.setNumThreads(0)

# 경로는 환경변수로 받는다 (README 의 Environment variables 참조)
REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
VQVAE_SRC = Path(os.environ.get("MESH_VQVAE_DIR", "checkpoints/mesh_vqvae_54")).resolve()
VQVAE_DEFAULT = "checkpoint/MESH_VQVAE/mesh_vqvae_54"
REGRESSOR_OVERRIDES = ("head_axis", "rot_from")
LR_STEP = 100
PREVIEW = 6
# rebuild_optimizer 가 main 의 p 를 못 보므로 모듈 수준에서 한 번 읽는다
_PARAMS = json.loads(os.environ.get("CQ_PARAMS") or "{}")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and not dst.exists():
        dst.unlink()
    if not dst.exists():
        dst.symlink_to(src)


def prepare() -> None:
    if not (REPO / WEIGHTS).exists():
        cand = Path(os.environ.get("DINOV3_WEIGHTS", WEIGHTS)).resolve()
        if cand.exists():
            link(cand, REPO / WEIGHTS)
    link(VQVAE_SRC, REPO / "checkpoint" / "MESH_VQVAE" / "mesh_vqvae_54")


def build_train_list(name: str) -> Path:
    """프레임이 실제로 있는 데이터셋만 남긴다. MixedDataset 은 split("\\n") 이라 끝 줄바꿈 금지."""
    avail, missing = [], []
    for ln in (REPO / "configs" / name).read_text().split():
        (avail if (REPO / ln).exists() else missing).append(ln)
    out = REPO / "configs" / "active_train.txt"
    out.write_text("\n".join(avail))
    print(f"학습 데이터셋: {avail}", flush=True)
    if missing:
        print(f"제외(프레임 없음): {missing}", flush=True)
    return out


def find_blocks(dino) -> torch.nn.ModuleList:
    """DINOv3 의 트랜스포머 블록 목록을 찾는다. 이름이 다르면 크게 실패한다."""
    for attr in ("blocks", "layers", "encoder_layers"):
        mod = getattr(dino, attr, None)
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 1:
            print(f"블록 목록: dinov3.{attr} ({len(mod)}개)", flush=True)
            return mod
    raise RuntimeError(f"블록 목록을 못 찾았다. 후보 속성: {[n for n, _ in dino.named_children()]}")


def unfreeze(model, n_blocks: int) -> list:
    """마지막 n_blocks 개와 최종 norm 을 학습 대상으로 바꾸고, no_grad 봉인을 푼다."""
    dino = model.dinov3
    for p in dino.parameters():
        p.requires_grad_(False)
    if n_blocks <= 0:
        print("unfreeze_blocks=0 — 인코더 완전 동결(대조군과 동일)", flush=True)
        return []

    blocks = find_blocks(dino)
    opened = list(blocks[-n_blocks:])
    for m in opened:
        for p in m.parameters():
            p.requires_grad_(True)
    for name in ("norm", "ln_post", "final_norm"):
        mod = getattr(dino, name, None)
        if isinstance(mod, torch.nn.Module):
            for p in mod.parameters():
                p.requires_grad_(True)
            print(f"최종 정규화도 해제: dinov3.{name}", flush=True)
            break

    # 인스턴스 단위로만 감싼다 — state_dict 키는 건드리지 않는다
    cls_fn = type(dino).get_intermediate_layers

    def with_grad(self_, *a, **k):
        with torch.enable_grad():
            return cls_fn(self_, *a, **k)

    dino.get_intermediate_layers = types.MethodType(with_grad, dino)

    params = [p for p in dino.parameters() if p.requires_grad]
    print(f"인코더 해제: 마지막 {n_blocks}블록 / {len(blocks)}, "
          f"학습 파라미터 {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    return params


def rebuild_optimizer(trainer, backbone_params, lr: float, wd: float, mult: float) -> None:
    """백본에 낮은 lr 을 주는 두 그룹으로 optimizer 를 다시 만든다."""
    ids = {id(p) for p in backbone_params}
    head = [p for p in trainer.model.parameters() if p.requires_grad and id(p) not in ids]
    groups = [{"params": head, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * mult})
    trainer.optimizer = torch.optim.AdamW(groups, weight_decay=wd)
    # 부모 실행은 LR_STEP=100 이라 30에폭 동안 학습률이 한 번도 안 떨어졌다. 감쇠 없이
    # 멈춘 곡선으로는 '수렴했다'와 '정체했다'를 가릴 수 없어서 파라미터로 뺀다.
    step = int(_PARAMS.get("lr_step", LR_STEP))
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=step)
    print(f"LR StepLR step_size={step}", flush=True)
    print(f"optimizer 재구성: 헤드 {sum(p.numel() for p in head) / 1e6:.2f}M @ lr {lr}, "
          f"백본 {sum(p.numel() for p in backbone_params) / 1e6:.2f}M @ lr {lr * mult}", flush=True)


def init_from(trainer, spec: str | None) -> None:
    if not spec:
        print("warm-start 없음 — 무작위 초기화", flush=True)
        return
    ck = torch.load(spec, map_location="cpu")
    res = trainer.model.load_state_dict(ck["model"], strict=False)
    print(f"warm-start: {spec} (완료 epoch {ck.get('epoch')})\n"
          f"  missing {len(res.missing_keys)}: {res.missing_keys[:PREVIEW]}\n"
          f"  unexpected {len(res.unexpected_keys)}: {res.unexpected_keys[:PREVIEW]}", flush=True)


def resume_from(trainer, spec: str | None) -> None:
    if not spec:
        return
    path = Path(spec)
    if not path.exists():
        print(f"[!] 체크포인트 없음 ({spec})", flush=True)
        return
    trainer.load(str(path))
    done = trainer.load_epoch
    trainer.load_epoch = done + 1
    trainer.lr_scheduler.last_epoch = done
    print(f"재개: {path}  완료 epoch {done} -> {trainer.load_epoch} 부터", flush=True)


def install_metric_sink(start_epoch: int = 0) -> None:
    """트레이너의 loguru 로그를 CQ MetricWriter 형식으로 되쏜다."""
    import re

    from loguru import logger

    num = r"([0-9.eE+-]+)"
    tr = re.compile(rf"Training loss: {num} \|\| Accuracy: {num} \|\| PA-MPJPE: {num} \|\| V2V: {num}")
    va = re.compile(rf"Validation V2V: Accuracy: {num} \|\| PA-MPJPE: {num} \|\| V2V: {num}")
    state = {"epoch": start_epoch}

    def sink(message) -> None:
        text = str(message)
        if m := tr.search(text):
            loss, acc, pa, v2v = m.groups()
            print(f"@epoch={state['epoch']} @train_loss={loss} @train_acc={acc} "
                  f"@train_pampjpe={pa} @train_v2v={v2v}", flush=True)
        elif m := va.search(text):
            acc, pa, v2v = m.groups()
            print(f"@epoch={state['epoch']} @val_acc={acc} @val_pampjpe={pa} @val_v2v={v2v}", flush=True)
            state["epoch"] += 1

    logger.add(sink, level="INFO")


def main() -> int:
    prepare()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    from omegaconf import OmegaConf

    from mesh_vq_vae import FullyConvAE, MeshVQVAE
    from vq_hps import DatasetHMR, MeshRegressor, MeshRegressor_Train, MixedDataset, set_seed

    cfg = OmegaConf.load("configs/config.yaml")
    p = json.loads(os.environ.get("CQ_PARAMS") or "{}")
    if "epochs" in p:
        cfg.train.total_epoch = int(p["epochs"])
    if "batch" in p:
        cfg.train.batch = int(p["batch"])
        cfg.modelconv.batch = int(p["batch"])
    cfg.train.workers = int(p.get("workers", 18))
    for k in REGRESSOR_OVERRIDES:
        cfg.regressor[k] = p.get(k, {"head_axis": "token", "rot_from": "attn"}[k])
    n_blocks = int(p.get("unfreeze_blocks", 4))
    mult = float(p.get("backbone_lr_mult", 0.1))
    # 시드 반복(M3): 원본 실행은 set_seed() 기본값 0 이었다. 파라미터로 받되 기본은 그대로.
    seed = int(p.get("seed", 0))
    set_seed(seed)
    print(f"@seed={seed}", flush=True)
    print(f"epochs={cfg.train.total_epoch} batch={cfg.train.batch} lr={cfg.train.lr} "
          f"unfreeze_blocks={n_blocks} backbone_lr_mult={mult} "
          f"regressor={OmegaConf.to_container(cfg.regressor)}", flush=True)

    ref_bm = np.load("body_models/smplh/neutral/model.npz")
    training_data = MixedDataset(str(build_train_list(cfg.training_data.file)),
                                 augment=True, flip=True, proportion=1.0)
    validation_data = DatasetHMR(dataset_file=f"datasets/{cfg.validation_data.file}",
                                 augment=False, proportion=0.1)
    print(f"train={len(training_data)} val={len(validation_data)}", flush=True)

    conv = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(conv, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model=str(p.get("vqvae", VQVAE_DEFAULT)))
    conv.init_test_mode()

    model = MeshRegressor(**cfg.regressor)
    backbone_params = unfreeze(model, n_blocks)
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"@trainable_M={trainable / 1e6:.2f}", flush=True)

    J = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()
    J24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()
    trainer = MeshRegressor_Train(model, training_data, validation_data, cfg.train,
                                  faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
                                  vqvae_mesh=mesh_vqvae, joints_regressor=J, joints_regressor_smpl=J24)
    rebuild_optimizer(trainer, backbone_params, float(cfg.train.lr),
                      float(cfg.train.weight_decay), mult)
    init_from(trainer, p.get("init_from"))
    resume_from(trainer, p.get("resume"))

    if int(p.get("check_grad", 0)):
        return check_grad(trainer, mesh_vqvae, J, n_blocks)

    install_metric_sink(start_epoch=trainer.load_epoch)
    print(f"checkpoint dir: {trainer.follow.path}", flush=True)
    trainer.fit()
    return 0


def check_grad(trainer, vqvae, joints_reg, n_blocks: int) -> int:
    """한 배치만 흘려 인코더에 기울기가 실제로 닿는지 확인하고 끝낸다.

    torch.no_grad() 봉인을 enable_grad 로 풀었는지가 이 실험의 전제다. 틀리면 학습은
    조용히 성공하는데 인코더는 한 번도 안 바뀐다 — 하루를 버리기 전에 여기서 잡는다.
    """
    from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix

    model, dev = trainer.model, trainer.device
    data = next(iter(trainer.training_loader))
    pred_v = trainer.tpose.repeat(data["img"].shape[0], 1, 1)
    jr = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
    pose_init = torch.matmul(jr, pred_v).reshape(-1, 51)

    model.train()
    trainer.optimizer.zero_grad()
    out = model(data["img"].to(dev), pose_init=pose_init.to(dev),
                mesh_decoder=vqvae, joints_reg=joints_reg)
    gt_idx = vqvae.get_codebook_indices(data["local_mesh"].to(dev)).cpu()
    ce = torch.nn.CrossEntropyLoss()(torch.flatten(out["prob_indices"], 0, 1),
                                     torch.flatten(gt_idx).long())
    rot = torch.nn.MSELoss()(rotation_6d_to_matrix(out["pred_rot"]),
                             axis_angle_to_matrix(data["rotation"]))
    (ce + rot).backward()

    blocks = find_blocks(model.dinov3)
    opened, frozen = [], []
    for i, blk in enumerate(blocks):
        g = sum(q.grad.abs().sum().item() for q in blk.parameters() if q.grad is not None)
        (opened if i >= len(blocks) - n_blocks else frozen).append((i, g))
    head_g = sum(q.grad.abs().sum().item() for n, q in model.named_parameters()
                 if q.grad is not None and not n.startswith("dinov3."))
    print(f"\n===== 기울기 확인 =====", flush=True)
    print(f"token CE {ce.item():.3f}  rot MSE {rot.item():.4f}", flush=True)
    print(f"헤드 기울기 합 {head_g:.2f}", flush=True)
    print(f"해제된 블록 {[i for i, _ in opened]}: 기울기 합 "
          f"{[f'{g:.2f}' for _, g in opened]}", flush=True)
    print(f"동결 블록 중 기울기가 흐른 것: "
          f"{[i for i, g in frozen if g > 0] or '없음 (정상)'}", flush=True)
    ok = head_g > 0 and all(g > 0 for _, g in opened) and not any(g > 0 for _, g in frozen)
    print(f"@grad_check_ok={int(ok)} @head_grad={head_g:.3f} "
          f"@opened_grad={sum(g for _, g in opened):.3f}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
