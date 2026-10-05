"""Fig. factorial(2×2 절제)와 Fig. rotation-blind(회전 주입)를 그린다.

수치는 results/ 의 표와 같은 값이다(회전 주입: rotation_blindness.py, 절제: 학습 run 의 e25–29 평균, seed 0).
출력 폴더는 FIG_DIR 환경변수(기본 ./figures/out).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(os.environ.get("FIG_DIR", "figures/out"))
OUT.mkdir(parents=True, exist_ok=True)

C = {"TokenHMR": "#0072B2", "HMR 2.0": "#D55E00", "SPIN": "#009E73"}
MK = {"TokenHMR": "o", "HMR 2.0": "s", "SPIN": "^"}
LS = {"TokenHMR": "-", "HMR 2.0": "--", "SPIN": ":"}
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 7.5,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 회전 주입: 무작위축 16개 평균 MPJPE(mm), 수직축 회전 MPJPE(mm)
TH = [0, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
RAND = {
    "TokenHMR": [73.4260, 73.6841, 74.5682, 80.6157, 98.9581, 122.9812, 149.7019,
                 206.4459, 293.0689, 377.0048, 527.1059, 643.1371, 741.4710],
    "HMR 2.0": [81.2683, 81.5527, 82.4048, 88.0379, 105.0199, 127.5711, 153.0245,
                207.8224, 292.3613, 374.6755, 522.2531, 636.4642, 733.1954],
    "SPIN": [92.8702, 93.1441, 93.7942, 97.8592, 110.4840, 128.2086, 149.0473,
             195.5419, 269.3040, 342.1378, 473.7296, 576.0183, 662.9310],
}
VERT = {
    "TokenHMR": [73.4260, 73.4044, 73.5618, 75.0338, 80.3194, 88.3880, 98.5000,
                 122.6894, 163.5323, 205.2178, 282.0537, 342.5089, 394.8497],
    "HMR 2.0": [81.2683, 81.0577, 81.0156, 81.8773, 86.3306, 93.8823, 103.7235,
                127.8688, 169.3013, 211.9365, 291.2128, 354.2269, 410.1409],
    "SPIN": [92.8702, 92.8233, 92.8946, 93.8138, 97.5866, 103.7885, 111.9526,
             132.4055, 168.5274, 206.5971, 278.4820, 336.0589, 386.8834],
}


def fig_rotation_blind():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for bb, ys in RAND.items():
        a.plot(TH, [y - ys[0] for y in ys], LS[bb], marker=MK[bb], ms=3.5, lw=1.6,
               color=C[bb], label=bb)
    a.plot(TH, [0] * len(TH), "-", color=INK, lw=2.2, label="PA-MPJPE, all three")
    a.set_xlabel(r"injected global rotation $\theta$ (deg)")
    a.set_ylabel(r"$\Delta$ error vs. $\theta{=}0$ (mm)")
    a.set_title("(a) Random axis: PA-MPJPE does not move", loc="left")
    a.set_xticks([0, 30, 60, 90, 120, 180])
    style(a)
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.33))
    for bb in RAND:
        dr = [y - RAND[bb][0] for y in RAND[bb]]
        dv = [y - VERT[bb][0] for y in VERT[bb]]
        pts = [(t, v / r) for t, v, r in zip(TH, dv, dr) if r > 1 and 5 <= t <= 90]
        b.plot([p[0] for p in pts], [p[1] for p in pts], LS[bb], marker=MK[bb], ms=3.5,
               lw=1.6, color=C[bb], label=bb)
    b.axhline(1.0, color=MUTED, lw=0.8)
    b.text(88, 1.0, "random-axis cost", color=MUTED, fontsize=7, ha="right", va="bottom")
    b.set_ylim(0, 1.05)
    b.set_xlabel(r"injected rotation $\theta$ (deg)")
    b.set_ylabel("vertical-axis / random-axis\n" + r"$\Delta$MPJPE")
    b.set_title("(b) Azimuth is the least visible axis", loc="left")
    style(b)
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rotation_blind.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_factorial():
    cfg = ["as received", "+ token fix", "+ orientation\nfix", "+ both", "+ both,\nunfrozen"]
    pa = [53.01, 51.99, 53.13, 51.89, 45.56]
    pve = [152.1, 149.6, 94.6, 92.5, 79.9]
    fill = ["#9aa3ad", "#9aa3ad", "#0072B2", "#0072B2", "#D55E00"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, vals, name, lim in ((a, pa, "PA-MPJPE (mm)", (40, 56)),
                                (b, pve, "PVE (mm)", (0, 165))):
        ys = list(range(len(cfg)))[::-1]
        ax.barh(ys, vals, color=fill, height=0.62, edgecolor="white", lw=2)
        for y, v in zip(ys, vals):
            ax.text(v + (lim[1] - lim[0]) * 0.012, y, f"{v:.2f}" if v < 60 else f"{v:.1f}",
                    va="center", fontsize=7.5, color=INK)
        ax.set_xlim(*lim)
        ax.set_xlabel(name)
        ax.set_yticks(ys)
        ax.set_yticklabels(cfg)
        style(ax)
        ax.grid(False, axis="y")
    a.set_title("(a) aligned metric", loc="left")
    b.set_title("(b) unaligned metric", loc="left")
    a.text(40.4, 2, "orientation fix: +0.12 mm", va="center", fontsize=7, color="white")
    b.text(4, 2, "orientation fix: $-$57.5 mm", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(OUT / "fig_factorial.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_rotation_blind()
    fig_factorial()
