"""A(IVC) Fig. 2 대체: 시스템 도식 — 두 방향 판독기(as received / fixed)를 함께 그린다.

원본 그림(fig3.png)은 풀링 MLP 방향 헤드를 '방법'으로 그려 결함을 방법처럼 보이게 했다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent  # 그림은 이 폴더에 저장한다
INK, MUTED, FROZEN, TRAIN, POOL, SPAT = "#222222", "#666666", "#dfe3e8", "#ffffff", "#9aa3ad", "#0072B2"
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(10.4, 4.3))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.3); ax.axis("off")


def box(x, y, w, h, text, fc, ec=INK, tc=INK, ls="-", lw=0.9, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)


def arrow(a, b, color=MUTED, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color,
                                 lw=1.0, linestyle=ls))


# 주 경로 (위쪽 줄)
box(0.05, 2.55, 0.9, 0.8, "image\n224×224", TRAIN)
box(1.25, 2.4, 1.35, 1.1, "DINOv3 ViT-L\n(frozen)\npatch grid\n14×14×1024", FROZEN)
box(2.95, 2.55, 1.35, 0.8, "domain\ntransformer\n(1 enc + 1 dec)", TRAIN)
box(4.65, 2.55, 1.35, 0.8, "token head\n54 × 512 logits\n→ arg max", TRAIN)
box(6.35, 2.4, 1.4, 1.1, "Mesh VQ-VAE\ncodebook +\ndecoder (frozen)", FROZEN)
box(8.1, 2.55, 1.0, 0.8, "canonical\nmesh", TRAIN)
box(9.35, 2.55, 1.0, 0.8, "R · mesh\n(camera\nframe)", TRAIN)
for a, b in [((0.95, 2.95), (1.25, 2.95)), ((2.6, 2.95), (2.95, 2.95)),
             ((4.3, 2.95), (4.65, 2.95)), ((6.0, 2.95), (6.35, 2.95)),
             ((7.75, 2.95), (8.1, 2.95)), ((9.1, 2.95), (9.35, 2.95))]:
    arrow(a, b)

# 방향 판독기 두 갈래 (아래쪽)
box(2.95, 1.25, 2.6, 0.75, "as received: global average pool\n→ MLP → 6D rotation", "#f1f3f5",
    ec=POOL, tc=INK, ls="--")
box(2.95, 0.2, 2.6, 0.75, "fixed: learned query cross-attends\nover the grid → 6D rotation", "#e6f0f8",
    ec=SPAT, tc=INK)
box(6.05, 0.2, 2.0, 0.75, "camera head (s, t$_x$, t$_y$)\n2D reprojection loss only", TRAIN, fs=7.5)
arrow((1.925, 2.4), (2.95, 1.62), color=POOL, ls="--")
arrow((1.925, 2.4), (2.95, 0.57), color=SPAT)
# 회전 → R·mesh
ax.plot([5.55, 9.85], [1.62, 1.62], color=POOL, lw=1.0, ls="--")
ax.plot([5.55, 5.8, 5.8, 9.85], [0.57, 0.57, 1.45, 1.45], color=SPAT, lw=1.0)
ax.plot([9.85, 9.85], [1.45, 1.62], color=MUTED, lw=1.0)
arrow((9.85, 1.62), (9.85, 2.55), color=MUTED)
ax.text(7.6, 1.68, "global orientation R", color=MUTED, fontsize=7.5, ha="center", va="bottom")
ax.text(0.05, 4.05, "grey fill = frozen; white = trained (26.9 M parameters)",
        color=MUTED, fontsize=7.5, va="center")
fig.savefig(OUT / "fig_system.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_system.png", dpi=160, bbox_inches="tight")
print("saved")
