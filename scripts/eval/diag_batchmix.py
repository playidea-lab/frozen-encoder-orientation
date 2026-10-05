"""토큰 헤드가 배치 안 다른 샘플을 보는지 실측한다.

같은 이미지 x0 를 (a) 배치 0번 자리, (b) 배치 31번 자리(앞 31개는 다른 이미지)에 넣고
x0 의 토큰 로짓·예측 인덱스·회전을 비교한다. 순전파가 샘플별로 독립이면 완전히 같아야 한다.

원 실행은 같은 폴더의 diagnose.py 에서 prepare/build_vqvae/build_regressor 를 가져왔다.
공개판은 그 세 함수를 여기에 그대로 옮겼다. 원본 구성(head_axis=batch, rot_from=pooled)
체크포인트 경로는 CKPT_BATCHMIX 환경변수로 준다.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)

REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
VQVAE_SRC = Path(os.environ.get("MESH_VQVAE_DIR", "checkpoints/mesh_vqvae_54")).resolve()
CKPT = os.environ.get("CKPT_BATCHMIX", "")
N_PROBE = 8


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


def build_vqvae(cfg, dev):
    from mesh_vq_vae import FullyConvAE, MeshVQVAE

    conv = FullyConvAE(cfg.modelconv, test_mode=True)
    vq = MeshVQVAE(conv, **cfg.vqvaemesh)
    vq.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    conv.init_test_mode()
    return vq.to(dev).eval()


def build_regressor(cfg, dev):
    from vq_hps import MeshRegressor

    m = MeshRegressor(**cfg.regressor).to(dev)
    joints_reg = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()
    tpose = torch.from_numpy(np.load("body_models/tpose.npy")).float()
    return m, joints_reg, tpose


def forward_logits(model, img, vq, joints_reg, tpose, dev):
    pred_v = tpose.repeat(img.shape[0], 1, 1)
    jr = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
    pose_init = torch.matmul(jr, pred_v).reshape(-1, 51)
    out = model(img.to(dev), pose_init=pose_init.to(dev), mesh_decoder=vq, joints_reg=joints_reg)
    return out["prob_indices"], out["pred_indices"], out["pred_rot"]


def main() -> int:
    if not CKPT:
        raise SystemExit("CKPT_BATCHMIX 환경변수에 원본 구성 체크포인트 경로를 지정하라")
    prepare()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf

    from vq_hps import DatasetHMR, set_seed

    cfg = OmegaConf.load("configs/config.yaml")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()
    ds = DatasetHMR(dataset_file=f"datasets/{cfg.validation_data.file}", augment=False, proportion=0.1)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch, shuffle=False, num_workers=4,
                                         drop_last=True)
    it = iter(loader)
    batch_a, batch_b = next(it), next(it)
    vq = build_vqvae(cfg, dev)
    model, joints_reg, tpose = build_regressor(cfg, dev)
    ck = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ck["model"]); del ck
    model.eval()

    img_a, img_b = batch_a["img"], batch_b["img"]
    with torch.no_grad():
        la, ia, ra = forward_logits(model, img_a, vq, joints_reg, tpose, dev)
        # 같은 이미지를 두 번 넣어 결정성부터 확인
        la2, _, _ = forward_logits(model, img_a, vq, joints_reg, tpose, dev)
        print(f"결정성: 같은 배치 두 번 → 로짓 max|Δ| = {(la - la2).abs().max():.2e}", flush=True)
        for k in range(N_PROBE):
            # x_k 를 batch_b 의 마지막 자리(31)에 심는다: 앞 31개는 전부 다른 이미지
            mixed = img_b.clone()
            mixed[-1] = img_a[k]
            lm, im, rm = forward_logits(model, mixed, vq, joints_reg, tpose, dev)
            dl = (la[k] - lm[-1]).abs().max().item()
            tok_diff = (ia[k] != im[-1]).sum().item()
            drot = (ra[k] - rm[-1]).abs().max().item()
            print(f"x{k}: 자리 0 vs 자리 31 → 로짓 max|Δ|={dl:.3e}  달라진 토큰 {tok_diff}/54  "
                  f"회전 max|Δ|={drot:.2e}", flush=True)
        # 반대로: x_k 를 자리 0 에 두고 뒤 31개만 바꾼다 (causal 이면 자리 0 은 영향 없어야)
        mixed = img_b.clone()
        mixed[0] = img_a[0]
        lm, im, _ = forward_logits(model, mixed, vq, joints_reg, tpose, dev)
        print(f"x0 자리 0 고정, 뒤 31개 교체 → 로짓 max|Δ|={(la[0] - lm[0]).abs().max():.3e}  "
              f"달라진 토큰 {(ia[0] != im[0]).sum().item()}/54", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
