"""회전 헤드·토큰 헤드를 고친 변형을 학습한다 (u-dinovq-retrain-repro 의 자식).

부모 train.py 와 같고 세 가지가 추가됐다.
  - CQ_PARAMS 의 head_axis / rot_from 을 cfg.regressor 에 얹는다 (기본값 = 원본 동작)
  - init_from: 다른 run 의 가중치로 warm-start (optimizer·epoch 은 새로, 새 모듈은 무작위 초기화)
  - epochs 로 짧은 1단계 실험을 돌릴 수 있다
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 데이터로더가 병목이다(GPU 0%, epoch 23.5분). 워커마다 OpenCV가 스레드를
# 띄워 서로 경합하므로 끈다 — 학습 결과에는 영향이 없다.
cv2.setNumThreads(0)

# 경로는 환경변수로 받는다 (README 의 Environment variables 참조)
REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
VQVAE_SRC = Path(os.environ.get("MESH_VQVAE_DIR", "checkpoints/mesh_vqvae_54")).resolve()
REGRESSOR_OVERRIDES = ("head_axis", "rot_from")
PREVIEW_KEYS = 6


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
    """프레임이 실제로 있는 데이터셋만 남긴다.

    MixedDataset은 split("\\n")을 쓰므로 끝 줄바꿈은 빈 항목이 된다.
    """
    avail, missing = [], []
    for ln in (REPO / "configs" / name).read_text().split():
        (avail if (REPO / ln).exists() else missing).append(ln)
    out = REPO / "configs" / "active_train.txt"
    out.write_text("\n".join(avail))
    print(f"학습 데이터셋: {avail}", flush=True)
    if missing:
        print(f"제외(프레임 없음): {missing}", flush=True)
    return out


def init_from(trainer, spec: str | None) -> None:
    """다른 run 의 가중치로 warm-start 한다.

    토큰 경로(트랜스포머·임베딩·헤드)는 그대로 받고, 새 모듈(RotCamHead)은 체크포인트에
    없으니 strict=False 로 싣는다. 무엇이 새로 들어갔는지는 반드시 남긴다.
    """
    if not spec:
        print("warm-start 없음 — 무작위 초기화", flush=True)
        return
    ck = torch.load(spec, map_location="cpu")
    res = trainer.model.load_state_dict(ck["model"], strict=False)
    print(f"warm-start: {spec} (완료 epoch {ck.get('epoch')})\n"
          f"  missing {len(res.missing_keys)}: {res.missing_keys[:PREVIEW_KEYS]}\n"
          f"  unexpected {len(res.unexpected_keys)}: {res.unexpected_keys[:PREVIEW_KEYS]}",
          flush=True)


def resume_from(trainer, spec: str | None) -> None:
    """중단된 학습을 이어붙인다 (같은 구성의 체크포인트만)."""
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
    print(f"재개: {path}\n  완료 epoch {done} -> epoch {trainer.load_epoch} 부터, "
          f"lr={trainer.optimizer.param_groups[0]['lr']}", flush=True)


def install_metric_sink(start_epoch: int = 0) -> None:
    """트레이너의 loguru 로그를 CQ MetricWriter가 읽는 형식으로 되쏜다."""
    import re

    from loguru import logger

    num = r"([0-9.eE+-]+)"
    train_re = re.compile(
        rf"Training loss: {num} \|\| Accuracy: {num} \|\| PA-MPJPE: {num} \|\| V2V: {num}")
    val_re = re.compile(
        rf"Validation V2V: Accuracy: {num} \|\| PA-MPJPE: {num} \|\| V2V: {num}")
    state = {"epoch": start_epoch}

    def sink(message) -> None:
        text = str(message)
        if m := train_re.search(text):
            loss, acc, pa, v2v = m.groups()
            print(f"@epoch={state['epoch']} @train_loss={loss} @train_acc={acc} "
                  f"@train_pampjpe={pa} @train_v2v={v2v}", flush=True)
        elif m := val_re.search(text):
            acc, pa, v2v = m.groups()
            print(f"@epoch={state['epoch']} @val_acc={acc} @val_pampjpe={pa} "
                  f"@val_v2v={v2v}", flush=True)
            state["epoch"] += 1

    logger.add(sink, level="INFO")


def main() -> int:
    prepare()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    from omegaconf import OmegaConf

    from mesh_vq_vae import FullyConvAE, MeshVQVAE
    from vq_hps import (
        DatasetHMR,
        MeshRegressor,
        MeshRegressor_Train,
        MixedDataset,
        set_seed,
    )

    cfg = OmegaConf.load("configs/config.yaml")
    params = json.loads(os.environ.get("CQ_PARAMS") or "{}")
    if "epochs" in params:
        cfg.train.total_epoch = int(params["epochs"])
    cfg.train.workers = int(params.get("workers", 18))
    for key in REGRESSOR_OVERRIDES:
        if key in params:
            cfg.regressor[key] = params[key]
    set_seed()
    print(f"epochs={cfg.train.total_epoch} batch={cfg.train.batch} "
          f"lr={cfg.train.lr} wd={cfg.train.weight_decay} workers={cfg.train.workers} "
          f"regressor={OmegaConf.to_container(cfg.regressor)}", flush=True)

    ref_bm = np.load("body_models/smplh/neutral/model.npz")

    training_data = MixedDataset(
        str(build_train_list(cfg.training_data.file)),
        augment=True, flip=True, proportion=1.0,
    )
    validation_data = DatasetHMR(
        dataset_file=f"datasets/{cfg.validation_data.file}",
        augment=False, proportion=0.1,
    )
    print(f"train={len(training_data)} val={len(validation_data)}", flush=True)

    convmesh_model = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    convmesh_model.init_test_mode()

    mesh_regressor = MeshRegressor(**cfg.regressor)
    trainable = sum(p.numel() for p in mesh_regressor.parameters()) - sum(
        p.numel() for p in mesh_regressor.dinov3.parameters())
    print(f"@trainable_M={trainable / 1e6:.2f}", flush=True)

    J_regressor = torch.from_numpy(
        np.load("body_models/J_regressor_h36m.npy")).float()
    J_regressor_24 = torch.from_numpy(
        np.load("body_models/J_regressor_24.npy")).float()

    trainer = MeshRegressor_Train(
        mesh_regressor,
        training_data,
        validation_data,
        cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
        vqvae_mesh=mesh_vqvae,
        joints_regressor=J_regressor,
        joints_regressor_smpl=J_regressor_24,
    )
    init_from(trainer, params.get("init_from"))
    resume_from(trainer, params.get("resume"))
    install_metric_sink(start_epoch=trainer.load_epoch)
    print(f"checkpoint dir: {trainer.follow.path}", flush=True)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
