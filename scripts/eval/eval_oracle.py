"""예측 회전 대 GT 회전 대입 — 방향 판독이 PVE 에 얼마를 주는지 직접 잰다 (평가만).

논문 A 는 PVE 격차를 직교 합성 모델로 역산해 '약 19°' 라고 썼다(IVC 리뷰 M3).
모델은 캐노니컬 메시에 예측 회전을 곱해 내므로(pred_mesh = R_pred · canonical),
같은 캐노니컬에 GT 회전을 곱하면 '방향이 완벽했다면' 의 PVE 가 바로 나온다.
회전 오차는 상대회전 R_pred · R_gt^T 의 회전벡터를 카메라 축 성분으로 나눠 보고한다
(y = 영상 세로축 ≈ 방위각/yaw, x ≈ pitch, z = 광축 ≈ roll).

체크포인트 (공개 저장소에는 가중치가 없다 — train/*.py 로 학습한 뒤 환경변수
CKPT_<NAME 대문자> 로 경로를 지정한다. 예: CKPT_BOTH_S0=/path/to/model_checkpoint):
  asrecv_s1/s2  수정 없음 (batch, pooled), 30ep e29  — 시드 반복
  both_s0/s1/s2 두 결함 수정 (token, attn), 30ep e29
  unfrozen      두 결함 수정 + 4블록 해제, e59
  pooled_e103   원본 구조 장기 학습 val-best (논문의 29.7° 출처)
CQ_PARAMS: test_for (쉼표 구분, test split 까지 평가할 이름)
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
CK = REPO / "checkpoint" / "MESH_REGRESSOR"
VQVAE_DEFAULT = "checkpoint/MESH_VQVAE/mesh_vqvae_54"


def ckpt(name: str) -> Path:
    """체크포인트 경로: CKPT_<NAME> 환경변수, 없으면 CK/<name>/model_checkpoint."""
    return Path(os.environ.get(f"CKPT_{name.upper()}", CK / name / "model_checkpoint"))


CONFIGS = {
    "asrecv_s1": (ckpt("asrecv_s1"), "batch", "pooled"),
    "asrecv_s2": (ckpt("asrecv_s2"), "batch", "pooled"),
    "both_s0": (ckpt("both_s0"), "token", "attn"),
    "both_s1": (ckpt("both_s1"), "token", "attn"),
    "both_s2": (ckpt("both_s2"), "token", "attn"),
    "unfrozen": (ckpt("unfrozen"), "token", "attn"),
    "pooled_e103": (ckpt("pooled_e103"), "batch", "pooled"),
}
SPLITS = {"val": ("datasets/3DPW/3DPW_validation.npz", 0.1),
          "test": ("datasets/3DPW/3DPW_test.npz", 1.0)}
DEFAULTS = dict(test_for="asrecv_s1,both_s0,unfrozen,pooled_e103")
BATCH = 32
MM = 1000.0


def geodesic_deg(r_pred, r_gt):
    rel = torch.matmul(r_pred.transpose(1, 2), r_gt)
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def evaluate(trainer, loader, J, dev) -> dict:
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix

    from vq_hps.utils.eval import mpjpe, pa_mpjpe, v2v

    model = trainer.model
    model.eval()
    acc = {k: [] for k in ("rot", "ax_x", "ax_y", "ax_z", "mp", "pa", "vv", "mp_o", "pa_o", "vv_o")}

    def per(fn, gt, pr, *a):
        return (MM * fn(gt, pr, *a, reduction=False)).reshape(len(gt), -1).mean(-1).tolist()

    with torch.no_grad():
        for data in loader:
            n = data["img"].shape[0]
            pred_v = trainer.tpose.repeat(n, 1, 1)
            jr = J[None, :].expand(n, -1, -1).to(pred_v)
            pose_init = torch.matmul(jr, pred_v).reshape(-1, 51)
            out = model(data["img"].to(dev), pose_init=pose_init.to(dev),
                        mesh_decoder=trainer.vqvae_mesh, joints_reg=J)
            r_p = rotation_6d_to_matrix(out["pred_rot"]).float().cpu()
            r_g = axis_angle_to_matrix(data["rotation"].float())
            acc["rot"] += geodesic_deg(r_p, r_g).tolist()
            # 상대회전(카메라 좌표계)의 회전벡터 성분, 도 단위
            aa = torch.rad2deg(matrix_to_axis_angle(r_p @ r_g.transpose(1, 2))).abs()
            acc["ax_x"] += aa[:, 0].tolist()
            acc["ax_y"] += aa[:, 1].tolist()
            acc["ax_z"] += aa[:, 2].tolist()
            gt, pr = data["mesh"], out["pred_mesh"].float()
            can = out["pred_mesh_canonical"].float()
            orc = (r_g @ can.transpose(2, 1)).transpose(2, 1)          # GT 회전 대입
            acc["mp"] += per(mpjpe, gt, pr, J)
            acc["pa"] += per(pa_mpjpe, gt, pr, J)
            acc["vv"] += per(v2v, gt, pr)
            acc["mp_o"] += per(mpjpe, gt, orc, J)
            acc["pa_o"] += per(pa_mpjpe, gt, orc, J)
            acc["vv_o"] += per(v2v, gt, orc)
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    r["rot_median"] = float(np.median(acc["rot"]))
    r["n"] = len(acc["rot"])
    return r


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from mesh_vq_vae import FullyConvAE, MeshVQVAE
    from vq_hps import DatasetHMR, MeshRegressor, MeshRegressor_Train

    import tempfile

    from vq_hps.train.train_mesh_regressor import follow_up_regressor as FU

    def _scratch_dir(self):
        """평가만 하므로 체크포인트 폴더를 만들지 않는다 (같은 분에 두 번 만들면 충돌)."""
        d = Path(tempfile.mkdtemp(prefix="eval_follow_"))
        self.path = self.path_samples = self.path_samples_train = d

    FU.Follow.create_directory = _scratch_dir

    p = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    test_for = set(str(p["test_for"]).split(","))
    skip = set(filter(None, str(p.get("skip", "")).split(",")))
    print(f"params={p}", flush=True)
    cfg = OmegaConf.load("configs/config.yaml")
    ref_bm = np.load("body_models/smplh/neutral/model.npz")
    conv = FullyConvAE(cfg.modelconv, test_mode=True)
    vq = MeshVQVAE(conv, **cfg.vqvaemesh)
    vq.load(path_model=VQVAE_DEFAULT)
    conv.init_test_mode()
    J = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()
    J24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()
    datasets = {k: DatasetHMR(dataset_file=f, augment=False, proportion=pr) for k, (f, pr) in SPLITS.items()}

    results = {}
    for name, (ck, head_axis, rot_from) in CONFIGS.items():
        if name in skip:
            continue
        cfg.regressor["head_axis"] = head_axis
        cfg.regressor["rot_from"] = rot_from
        model = MeshRegressor(**cfg.regressor)
        trainer = MeshRegressor_Train(model, datasets["val"], datasets["val"], cfg.train,
                                      faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
                                      vqvae_mesh=vq, joints_regressor=J, joints_regressor_smpl=J24)
        state = torch.load(str(ck), map_location="cpu")
        res = trainer.model.load_state_dict(state["model"], strict=True)
        print(f"\n[{name}] {ck} epoch={state.get('epoch')} axis={head_axis} rot={rot_from} load={res}",
              flush=True)
        dev = trainer.device
        for split, ds in datasets.items():
            if split == "test" and name not in test_for:
                continue
            # 메시 디코더 배치 32 고정, 순서 고정(무수정 구성은 배치 구성에 의존하므로)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=12, drop_last=True)
            r = evaluate(trainer, loader, J, dev)
            results[f"{name}/{split}"] = r
            print(f"[{name}/{split}] n={r['n']} rot {r['rot']:.2f}° (med {r['rot_median']:.2f}) "
                  f"|x| {r['ax_x']:.2f} |y| {r['ax_y']:.2f} |z| {r['ax_z']:.2f}  "
                  f"pred: MPJPE {r['mp']:.2f} PA {r['pa']:.2f} PVE {r['vv']:.2f}  "
                  f"GT-rot: MPJPE {r['mp_o']:.2f} PA {r['pa_o']:.2f} PVE {r['vv_o']:.2f}", flush=True)
            t = f"{name}_{split}"
            print(f"@{t}_rot={r['rot']:.3f} @{t}_rot_med={r['rot_median']:.3f} @{t}_ax_x={r['ax_x']:.3f} "
                  f"@{t}_ax_y={r['ax_y']:.3f} @{t}_ax_z={r['ax_z']:.3f} @{t}_pve={r['vv']:.3f} "
                  f"@{t}_pve_gtrot={r['vv_o']:.3f} @{t}_mpjpe={r['mp']:.3f} @{t}_mpjpe_gtrot={r['mp_o']:.3f} "
                  f"@{t}_pa={r['pa']:.3f} @{t}_pa_gtrot={r['pa_o']:.3f}", flush=True)
        del trainer, model, state
        torch.cuda.empty_cache()
    print("RESULTS_JSON " + json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
