"""남아 있는 표 3 체크포인트의 전역 방향 오차를 직접 잰다 (평가만, 학습 없음).

논문 A는 PVE 격차를 직교 합성 모델로 역산해 '약 19°'라고 썼다. 모델은 6D 회전을 직접
예측하므로 GT 회전과의 측지 각도를 그대로 잴 수 있다. 여기서는 그것을 잰다.

평가 구성 (체크포인트 경로는 CKPT_BOTH, CKPT_UNFROZEN 환경변수):
  both        = 두 결함 수정, 인코더 동결
  unfrozen    = 두 결함 수정, 4블록 해제
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)

# 경로는 환경변수로 받는다 (README 의 Environment variables 참조)
REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
CK_ROOT = REPO / "checkpoint" / "MESH_REGRESSOR"
VQVAE_DEFAULT = "checkpoint/MESH_VQVAE/mesh_vqvae_54"
CONFIGS = {
    "both": Path(os.environ.get("CKPT_BOTH", CK_ROOT / "both" / "model_checkpoint")),
    "unfrozen": Path(os.environ.get("CKPT_UNFROZEN", CK_ROOT / "unfrozen" / "model_checkpoint")),
}
SPLITS = {
    "val": ("datasets/3DPW/3DPW_validation.npz", 0.1),   # 학습 중 검증과 같은 부분집합
    "test": ("datasets/3DPW/3DPW_test.npz", 1.0),
}
BATCH = 32
MM = 1000.0


def geodesic_deg(r_pred: torch.Tensor, r_gt: torch.Tensor) -> torch.Tensor:
    """두 회전행렬 사이 측지 각도(도)."""
    rel = torch.matmul(r_pred.transpose(1, 2), r_gt)
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def evaluate(trainer, loader, joints_reg, dev) -> dict:
    from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix

    from vq_hps.utils.eval import mpjpe, pa_mpjpe, v2v

    model = trainer.model
    model.eval()
    rot, mp, pa, vv = [], [], [], []
    with torch.no_grad():
        for data in loader:
            pred_v = trainer.tpose.repeat(data["img"].shape[0], 1, 1)
            jr = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
            pose_init = torch.matmul(jr, pred_v).reshape(-1, 51)
            out = model(data["img"].to(dev), pose_init=pose_init.to(dev),
                        mesh_decoder=trainer.vqvae_mesh, joints_reg=joints_reg)
            r_p = rotation_6d_to_matrix(out["pred_rot"]).float().cpu()
            r_g = axis_angle_to_matrix(data["rotation"].float())
            rot += geodesic_deg(r_p, r_g).tolist()
            gt, pr = data["mesh"], out["pred_mesh"].cpu()
            mp += (MM * mpjpe(gt, pr, joints_reg, reduction=False)).reshape(len(gt), -1).mean(-1).tolist()
            pa += (MM * pa_mpjpe(gt, pr, joints_reg, reduction=False)).reshape(len(gt), -1).mean(-1).tolist()
            vv += (MM * v2v(gt, pr, reduction=False)).reshape(len(gt), -1).mean(-1).tolist()
    rot = np.array(rot)
    return {"n": len(rot), "rot_mean_deg": float(rot.mean()), "rot_median_deg": float(np.median(rot)),
            "rot_p90_deg": float(np.percentile(rot, 90)),
            "mpjpe": float(np.mean(mp)), "pampjpe": float(np.mean(pa)), "v2v": float(np.mean(vv))}


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from mesh_vq_vae import FullyConvAE, MeshVQVAE
    from vq_hps import DatasetHMR, MeshRegressor, MeshRegressor_Train

    cfg = OmegaConf.load("configs/config.yaml")
    cfg.regressor["head_axis"] = "token"
    cfg.regressor["rot_from"] = "attn"
    ref_bm = np.load("body_models/smplh/neutral/model.npz")
    conv = FullyConvAE(cfg.modelconv, test_mode=True)
    vq = MeshVQVAE(conv, **cfg.vqvaemesh)
    vq.load(path_model=VQVAE_DEFAULT)
    conv.init_test_mode()
    J = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()
    J24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()

    datasets = {k: DatasetHMR(dataset_file=f, augment=False, proportion=p) for k, (f, p) in SPLITS.items()}
    results = {}
    for name, ck in CONFIGS.items():
        model = MeshRegressor(**cfg.regressor)
        trainer = MeshRegressor_Train(model, datasets["val"], datasets["val"], cfg.train,
                                      faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
                                      vqvae_mesh=vq, joints_regressor=J, joints_regressor_smpl=J24)
        state = torch.load(str(ck), map_location="cpu")
        res = trainer.model.load_state_dict(state["model"], strict=True)
        print(f"[{name}] {ck}  epoch={state.get('epoch')}  load={res}", flush=True)
        dev = trainer.device
        for split, ds in datasets.items():
            # 메시 디코더가 배치 32 고정이라 마지막 조각은 버린다 (학습 중 검증도 drop_last)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=12, drop_last=True)
            r = evaluate(trainer, loader, J, dev)
            results[f"{name}/{split}"] = r
            print(f"[{name}/{split}] n={r['n']} rot mean {r['rot_mean_deg']:.2f}° "
                  f"median {r['rot_median_deg']:.2f}° p90 {r['rot_p90_deg']:.2f}°  "
                  f"MPJPE {r['mpjpe']:.2f}  PA {r['pampjpe']:.2f}  V2V {r['v2v']:.2f}", flush=True)
            tag = f"{name}_{split}"
            print(f"@{tag}_rot_mean_deg={r['rot_mean_deg']:.3f} @{tag}_rot_median_deg={r['rot_median_deg']:.3f} "
                  f"@{tag}_mpjpe={r['mpjpe']:.3f} @{tag}_pampjpe={r['pampjpe']:.3f} @{tag}_v2v={r['v2v']:.3f}",
                  flush=True)
        del trainer, model
        torch.cuda.empty_cache()
    print("RESULTS_JSON " + json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
