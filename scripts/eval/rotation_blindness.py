"""전역 회전 오차를 통제 주입해 PA-MPJPE 와 MPJPE 의 반응을 잰다.

**왜**: A 재구성의 논지는 "Procrustes 정렬이 전역 회전 실패를 가린다"이다. 지금 증거는
절제표 한 줄뿐이다 — 회전 헤드를 복원하면 MPJPE 가 152.1→94.6(−57.5mm)인데 PA 는
53.01→53.13(+0.12mm)이다. 일화를 법칙으로 바꾸려면 전달함수가 필요하다.

**설계**: 예측 관절을 제 무게중심 둘레로 각도 θ 만큼 돌린다. 평행이동과 스케일은 건드리지
않으므로 순수 회전만 주입된다. PA 는 회전을 최적화로 제거하므로 **정확히 불변이어야 한다** —
이건 가정이 아니라 확인할 사실이다. MPJPE 는 단조 증가해야 한다.

**축 선택**: 축이 자세와 정렬되면 효과가 달라지므로 (a) 무작위 축 다수 평균과 (b) 수직축
(Y) 두 가지를 낸다. 실제 회전 헤드 실패는 대체로 수직축 주변이라 둘 다 필요하다.

**역산**: 전달함수가 있으면 "받은 코드의 MPJPE 146mm"가 몇 도에 해당하는지 읽을 수 있다.

입력: 3DPW test 에 대한 공개 모델(TokenHMR, 4D-Humans/HMR 2.0, SPIN)의 14관절 예측 pkl.
공개 저장소에는 넣지 않는다 — README 의 'Third-party predictions' 절차로 만든 뒤
PRED_DIR 환경변수로 폴더를 지정한다.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

N_JOINTS, N_COORD, MM = 14, 3, 1000.0
PRED_DIR = Path(os.environ.get("PRED_DIR", "data/predictions"))
BACKBONES = {
    "TokenHMR": PRED_DIR / "tokenhmr_3dpw_cuda.pkl",
    "4DHumans": PRED_DIR / "4dhumans_3dpw_cuda.pkl",
    "SPIN": PRED_DIR / "spin_3dpw_14j.pkl",
}
ANGLES_DEG = [0, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
N_AXES = 16                     # 무작위 축 표본 수
OUT = Path(os.environ.get("OUT_DIR", "results/rot_blind"))


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """표준 Procrustes(최소제곱 스케일 포함), 벡터화. 논문 전체와 동일 규약."""
    P = pred - pred.mean(1, keepdims=True)
    G = gt - gt.mean(1, keepdims=True)
    sp = np.sqrt((P ** 2).sum((1, 2)))[:, None, None]
    sg = np.sqrt((G ** 2).sum((1, 2)))[:, None, None]
    Pn, Gn = P / (sp + 1e-8), G / (sg + 1e-8)
    H = np.einsum("nji,njk->nik", Pn, Gn)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik",
                                        Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("nij,njk,nkl->nil", Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))
    scale = S[:, 0] + S[:, 1] + d * S[:, 2]
    aligned = sg * scale[:, None, None] * np.einsum("nij,nkj->nki", R, Pn) \
        + gt.mean(1, keepdims=True)
    return np.sqrt(((aligned - gt) ** 2).sum(-1)).mean(-1) * MM


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """정렬 없는 관절 평균 거리. exp700 규약과 동일."""
    return np.sqrt(((pred - gt) ** 2).sum(-1)).mean(-1) * MM


def rot_matrix(axis: np.ndarray, deg: float) -> np.ndarray:
    """로드리게스 공식. axis 는 단위벡터."""
    a = axis / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rotate_about_centroid(pred: np.ndarray, R: np.ndarray) -> np.ndarray:
    """제 무게중심 둘레로 회전 — 평행이동·스케일은 불변."""
    c = pred.mean(1, keepdims=True)
    return np.einsum("ij,nkj->nki", R, pred - c) + c


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    pred = np.array([s["pred_joints_3d_eval"].reshape(N_JOINTS, N_COORD) for s in d])
    gt = np.array([s["gt_joints_3d"].reshape(N_JOINTS, N_COORD) for s in d])
    return pred, gt


def main() -> int:
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(N_AXES, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    y_axis = np.array([0.0, 1.0, 0.0])

    out, t0 = {}, time.time()
    for bb, path in BACKBONES.items():
        pred, gt = load(path)
        pa0, mj0 = pa_mpjpe(pred, gt).mean(), mpjpe(pred, gt).mean()
        print(f"\n===== {bb}  n={len(pred)}  θ=0 에서 PA {pa0:.4f}  MPJPE {mj0:.4f}",
              flush=True)
        print(f"{'θ(deg)':>7} | {'PA(무작위축)':>13} {'ΔPA':>9} | "
              f"{'MPJPE(무작위축)':>15} {'ΔMPJPE':>9} | {'MPJPE(Y축)':>11}", flush=True)

        rows = []
        for deg in ANGLES_DEG:
            pas, mjs = [], []
            for ax in axes:
                R = rot_matrix(ax, deg)
                pr = rotate_about_centroid(pred, R)
                pas.append(pa_mpjpe(pr, gt).mean())
                mjs.append(mpjpe(pr, gt).mean())
            pa_r, mj_r = float(np.mean(pas)), float(np.mean(mjs))
            pr_y = rotate_about_centroid(pred, rot_matrix(y_axis, deg))
            mj_y = float(mpjpe(pr_y, gt).mean())
            pa_y = float(pa_mpjpe(pr_y, gt).mean())
            rows.append({"deg": deg, "pa_rand": pa_r, "mpjpe_rand": mj_r,
                         "pa_y": pa_y, "mpjpe_y": mj_y,
                         "pa_sd_over_axes": float(np.std(pas))})
            print(f"{deg:7d} | {pa_r:13.4f} {pa_r-pa0:+9.4f} | "
                  f"{mj_r:15.4f} {mj_r-mj0:+9.4f} | {mj_y:11.4f}", flush=True)

        drift = max(abs(r["pa_rand"] - pa0) for r in rows)
        drift_y = max(abs(r["pa_y"] - pa0) for r in rows)
        # 작은 각도 구간의 기울기 — 실제 회전 오차는 보통 여기 있다
        small = [r for r in rows if 0 < r["deg"] <= 20]
        slope = float(np.mean([(r["mpjpe_rand"] - mj0) / r["deg"] for r in small]))
        out[bb] = {"pa0": float(pa0), "mpjpe0": float(mj0), "rows": rows,
                   "pa_drift_max_mm": float(drift), "pa_drift_max_y_mm": float(drift_y),
                   "mpjpe_mm_per_deg_small": slope}
        print(f"  PA 최대 이탈 {drift:.5f} mm (무작위축) / {drift_y:.5f} mm (Y축)", flush=True)
        print(f"  작은각 기울기 {slope:.3f} mm/deg", flush=True)
        print(f"  @{bb}_pa_drift_max={drift:.5f} @{bb}_mpjpe_per_deg={slope:.4f}", flush=True)

    # 받은 코드의 MPJPE 146mm 는 몇 도에 해당하는가 — 4DHumans 전달함수로 역산
    ref = out["4DHumans"]
    target = 146.0 - 94.6                       # 회전 헤드 복원으로 사라진 몫
    deg_eq = target / ref["mpjpe_mm_per_deg_small"]
    print(f"\n역산: 회전 헤드 복원이 되돌린 {target:.1f} mm 는 "
          f"약 {deg_eq:.1f}° 의 전역 회전 오차에 해당한다 "
          f"(4DHumans 전달함수 {ref['mpjpe_mm_per_deg_small']:.3f} mm/deg 기준)", flush=True)
    print(f"@deg_equiv_146={deg_eq:.2f}", flush=True)
    out["_inversion"] = {"mpjpe_gap_mm": target, "deg_equivalent": float(deg_eq)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n저장: {OUT/'results.json'}  ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
