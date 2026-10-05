"""A(IVC) graphical abstract: 동결 인코더 → pooled/spatial 판독 → 방향 오차 → 지표가 보는 것.

수치 출처: probe job 2075940b·8b249e24 (3시드, 학습셋 오차), 절제 표 시드 0 (job d67fcc0d/b8908f4e).
IVC 요구: 531×1328 px 이상(세로×가로). 2.5:1 비율로 2656×1062 px 로 저장한다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent  # 그림은 이 폴더에 저장한다
INK, MUTED, POOL, SPAT, ACC = "#222222", "#666666", "#9aa3ad", "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})

fig = plt.figure(figsize=(13.28, 5.31))
# 패널 위치: [left, bottom, width, height]
ax0 = fig.add_axes([0.01, 0.06, 0.36, 0.80])
ax1 = fig.add_axes([0.45, 0.20, 0.22, 0.62])
ax2 = fig.add_axes([0.76, 0.20, 0.22, 0.62])


def box(ax, x, y, w, h, text, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=13)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=color, lw=1.6))


# (1) 구조 도식
ax0.set_xlim(-0.04, 1.04); ax0.set_ylim(0, 1); ax0.axis("off")
box(ax0, 0.00, 0.40, 0.26, 0.22, "frozen ViT\npatch grid", "#44505c")
box(ax0, 0.40, 0.66, 0.34, 0.18, "global average\npool → MLP", POOL, INK)
box(ax0, 0.40, 0.18, 0.34, 0.18, "keep spatial\nlayout → head", SPAT)
arrow(ax0, (0.26, 0.56), (0.40, 0.74))
arrow(ax0, (0.26, 0.46), (0.40, 0.27))
box(ax0, 0.82, 0.66, 0.18, 0.18, "rotation", POOL, INK)
box(ax0, 0.82, 0.18, 0.18, 0.18, "rotation", SPAT)
arrow(ax0, (0.74, 0.75), (0.82, 0.75))
arrow(ax0, (0.74, 0.27), (0.82, 0.27))
ax0.text(0.0, 0.97, "Same frozen features, two readouts", fontsize=15, color=INK,
         fontweight="bold", va="top")

# (2) 세 인코더의 학습셋 오차: 평균 pooling 대 격자 펼침 (3시드, job 2075940b·8b249e24)
enc = ["DINOv3", "DINOv2", "CLIP"]
pool_tr = [24.85, 30.49, 43.61]
grid_tr = [4.01, 2.29, 2.56]
xs = range(3)
w = 0.36
ax1.bar([x - w / 2 for x in xs], pool_tr, width=w, color=POOL, label="global average pool")
ax1.bar([x + w / 2 for x in xs], grid_tr, width=w, color=SPAT, label="spatial grid kept")
for x, v in zip(xs, pool_tr):
    ax1.text(x - w / 2, v + 1.0, f"{v:.0f}°", ha="center", color=INK, fontsize=12)
for x, v in zip(xs, grid_tr):
    ax1.text(x + w / 2, v + 1.0, f"{v:.0f}°", ha="center", color=INK, fontsize=12)
ax1.set_xticks(list(xs)); ax1.set_xticklabels(enc, fontsize=12)
ax1.set_ylim(0, 52); ax1.set_ylabel("orientation error on\ntraining images (deg)")
ax1.legend(loc="upper left", frameon=False, fontsize=10)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
ax1.set_title("Averaging loses orientation\nin three frozen encoders", loc="left", fontsize=15,
              color=INK, fontweight="bold")

# (3) 지표가 보는 것: 방향 수정의 효과 (부호 있는 값)
ax2.barh([1, 0], [0.12, -57.5], color=[POOL, ACC], height=0.55)
ax2.axvline(0, color=MUTED, lw=0.8)
ax2.set_yticks([1, 0]); ax2.set_yticklabels(["PA-MPJPE", "PVE"])
ax2.text(-1.5, 1, "+0.12 mm", va="center", ha="right", color=INK, fontsize=13)
ax2.text(-56, 0, "−57.5 mm", va="center", ha="left", color="white", fontsize=13)
ax2.set_xlim(-62, 6); ax2.set_xlabel("change from fixing the readout (mm)")
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.set_title("Procrustes alignment\nhides the fix", loc="left", fontsize=15,
              color=INK, fontweight="bold")

fig.savefig(OUT / "graphical_abstract.png", dpi=200, facecolor="white")
fig.savefig(OUT / "graphical_abstract.pdf", facecolor="white")
print("saved")
