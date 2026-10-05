"""정점 기반 회전 전달함수의 환산 계수 r̄_vertices / r̄_joints 를 잰다.

**배경**: 회전 주입 실험(rotation_blindness.py)은 14관절 MPJPE 로 전달함수를 냈는데, A 재구성의
절제표에서 회전 수정이 움직이는 것은 **PVE** 다. 단위가 달라 직접 역산할 수 없었다.

**원리**: 점집합을 제 무게중심 둘레로 축 a 만큼 θ 회전시키면 각 점의 이동거리는
  ||R x - x|| = 2 sin(θ/2) * r_perp
이므로, 평균 이동거리는 **평균 수직 반경 r̄ 에만 비례**한다. 관절 곡선을 정점 곡선으로 옮기는
계수는 r̄_v / r̄_j 하나다.

그리고 오차 합성은 선형이 아니라 직교다 — 실측으로 확인했다:
  MPJPE(θ) ≈ sqrt(MPJPE(0)^2 + d(θ)^2),  d(θ)=2 sin(θ/2) r̄_j
θ<=10° 에서 0.4%, 90° 에서 3.6% 오차. 이 모델로 PVE 격차를 각도로 역산한다.

GT SMPL 파라미터(body_pose, betas, gender)는 TokenHMR 평가용 3dpw_test.npz 에 있고,
smplx 로 메시를 만든다. 경로는 환경변수 TOKENHMR_3DPW_NPZ, PRED_DIR, SMPL_DIR 로 받는다.
"""

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ANN = os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/datasets/3dpw_test.npz")
PKL = str(Path(os.environ.get("PRED_DIR", "data/predictions")) / "4dhumans_3dpw_cuda.pkl")
SMPL_DIR = os.environ.get("SMPL_DIR", "data/body_models")  # smplx 는 <dir>/smplh/SMPLH_NEUTRAL.npz 를 찾는다
N_SAMPLE = 400
N_AXES = 24

# A 절제표: 회전 수정이 PVE 를 이만큼 움직인다 (두 수준에서)
PVE_BROKEN, PVE_FIXED = 152.1, 94.6
PVE_BROKEN2, PVE_FIXED2 = 149.6, 92.5


def mean_perp_radius(P: np.ndarray, axes: np.ndarray) -> float:
    C = P - P.mean(1, keepdims=True)
    out = []
    for a in axes:
        a = a / np.linalg.norm(a)
        par = (C @ a)[..., None] * a
        out.append(np.linalg.norm(C - par, axis=-1).mean())
    return float(np.mean(out))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(N_AXES, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    # --- 14관절 ---
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    idx = np.linspace(0, len(data) - 1, N_SAMPLE).astype(int)
    J = np.array([data[i]["gt_joints_3d"].reshape(14, 3) for i in idx])
    r_j = mean_perp_radius(J, axes)
    print(f"14관절   평균 수직 반경 {r_j*1000:7.1f} mm  (n={len(J)})", flush=True)

    # --- 6890 정점 ---
    import smplx
    ann = np.load(ANN, allow_pickle=True)
    pose = np.asarray(ann["body_pose"])[idx]
    betas = np.asarray(ann["betas"])[idx]
    print(f"body_pose {pose.shape}  betas {betas.shape}", flush=True)

    model = smplx.create(SMPL_DIR, model_type="smplh", gender="neutral",
                         use_pca=False, batch_size=len(idx), ext="npz")
    with torch.no_grad():
        out = model(betas=torch.tensor(betas[:, :10], dtype=torch.float32),
                    body_pose=torch.tensor(pose[:, 3:66], dtype=torch.float32),
                    global_orient=torch.zeros(len(idx), 3))
    V = out.vertices.numpy()
    r_v = mean_perp_radius(V, axes)
    print(f"{V.shape[1]} 정점  평균 수직 반경 {r_v*1000:7.1f} mm", flush=True)

    ratio = r_v / r_j
    print(f"\n환산 계수 r̄_v / r̄_j = {ratio:.4f}", flush=True)
    print(f"@r_joints_mm={r_j*1000:.2f} @r_vertices_mm={r_v*1000:.2f} @ratio={ratio:.4f}",
          flush=True)

    # --- 역산: 절제표의 PVE 격차는 몇 도인가 ---
    print("\n=== PVE 격차의 각도 환산 (직교 합성 모델) ===", flush=True)
    for lo, hi, label in ((PVE_FIXED, PVE_BROKEN, "토큰 파손 상태"),
                          (PVE_FIXED2, PVE_BROKEN2, "토큰 수정 상태")):
        d2 = hi ** 2 - lo ** 2
        if d2 <= 0:
            continue
        d = math.sqrt(d2) / 1000.0                       # m
        s = d / (2 * r_v)
        deg = 2 * math.degrees(math.asin(min(1.0, s)))
        print(f"  {label}: {hi:.1f} -> {lo:.1f} mm  "
              f"직교 성분 {d*1000:.1f} mm  =>  약 {deg:.1f}°", flush=True)
        print(f"  @deg_{label.split()[0]}={deg:.2f}", flush=True)

    print(f"\n({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
