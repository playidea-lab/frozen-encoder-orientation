"""동결 DINOv3 특징에서 전역 회전을 읽는 판독기 대조 — '풀링 자체'인가 '평균 풀링'인가.

IVC 리뷰 M1: 기존 spatial 헤드(학습 쿼리 하나의 cross-attention)도 결국 격자를 한 벡터로
줄이는 '학습된 풀링'이다. 그래서 '평균 풀링 + MLP < attention 풀링'만 보였을 뿐, 풀링 자체가
방향을 지운다는 근거는 아니다. 판독기를 늘려 변수를 가른다.

  pooled_w1024   평균 풀링 → MLP(1024)                 기존
  pooled_w2048   평균 풀링 → MLP(2048)                 기존, 용량 대조
  cls_w1024      CLS 토큰 → MLP(1024)                  DINOv3 의 표준 전역 특징
  grid_flat      2×2 평균(7×7) → 1×1 투영 96 → flatten → MLP   위치 보존, attention 없음
  conv_pool      3×3 conv 두 층 → 평균 풀링 → MLP       국소 처리 후 풀링
  attn_1q        학습 쿼리 1개 cross-attn (위치 임베딩 없음)  기존 spatial
  attn_1q_pos    위 + 학습 2D 위치 임베딩을 key 에 더함
  attn_4q        학습 쿼리 4개 → 이어붙여 MLP

특징·라벨·분할·손실·epoch 은 probe_rotation.py 와 같다. 특징은 한 번만 뽑고 시드마다 헤드를
새로 초기화한다. 보고: COCO 홀드아웃·3DPW val 측지 오차(평균·중앙값), 학습셋 오차, 파라미터 수.
CQ_PARAMS: n_train, n_val3dpw, epochs, batch, lr, seeds
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

cv2.setNumThreads(0)

# 경로는 환경변수로 받는다 (README 의 Environment variables 참조)
REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DEFAULTS = dict(n_train=9000, n_val3dpw=1000, epochs=60, batch=256, lr=3e-4,
                workers=12, extract_batch=64, holdout=0.12, seeds="0,1,2")
DIM = 1024
GRID = 14
N_INTERMEDIATE = 23          # 모델과 같은 호출: 마지막 23블록 중 [-1] = 최종(24번째) 블록, norm 적용
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def rot_out(width: int) -> nn.Linear:
    """6D 회전 출력층 — 항등 회전으로 시작한다."""
    lin = nn.Linear(width, 6)
    nn.init.zeros_(lin.weight)
    with torch.no_grad():
        lin.bias.copy_(torch.tensor(IDENTITY_6D))
    return lin


class PooledHead(nn.Module):
    """평균 풀링(또는 CLS) → 2층 MLP → 6D. 원본 구조."""

    def __init__(self, width=1024, dropout=0.5, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.mlp = nn.Sequential(nn.Linear(DIM, width), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):                  # spatial (B, dim, hw), cls (B, dim)
        x = cls if self.use_cls else spatial.mean(-1)
        return self.rot(self.mlp(x))


class GridFlatHead(nn.Module):
    """위치를 보존하되 attention 은 쓰지 않는다: 7×7 로 줄여 채널 투영 후 펼친다."""

    def __init__(self, proj=96, width=1024, dropout=0.5):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.proj = nn.Conv2d(DIM, proj, 1)
        self.mlp = nn.Sequential(nn.Linear(proj * 49, width), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):
        b = spatial.shape[0]
        x = self.proj(self.down(spatial.view(b, DIM, GRID, GRID)))
        return self.rot(self.mlp(x.flatten(1)))


class ConvPoolHead(nn.Module):
    """국소 처리 뒤 평균 풀링: 풀링 전에 계산을 주면 충분한가."""

    def __init__(self, ch=256, width=1024, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(DIM, ch, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(ch, width), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):
        b = spatial.shape[0]
        x = self.conv(spatial.view(b, DIM, GRID, GRID)).mean((2, 3))
        return self.rot(self.mlp(x))


class AttnHead(nn.Module):
    """학습 쿼리 n_q 개가 공간 특징에 cross-attention. pos=True 면 학습 2D 위치 임베딩을 key 에 더한다."""

    def __init__(self, n_q=1, pos=False, nhead=8, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_q, 1, DIM))
        nn.init.normal_(self.query, std=0.02)
        self.pos = nn.Parameter(torch.zeros(GRID * GRID, 1, DIM)) if pos else None
        if pos:
            nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(DIM, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM * n_q, DIM), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(DIM)

    def forward(self, spatial, cls):
        mem = spatial.permute(2, 0, 1)                # (hw, B, dim)
        key = mem + self.pos if self.pos is not None else mem
        q = self.query.expand(-1, mem.shape[1], -1)
        x, _ = self.attn(q, key, mem)                 # (n_q, B, dim)
        x = self.norm(x).permute(1, 0, 2).flatten(1)
        return self.rot(self.mlp(x))


def make_heads(dev, has_cls: bool) -> dict:
    heads = {"pooled_w1024": PooledHead(1024), "pooled_w2048": PooledHead(2048)}
    if has_cls:
        heads["cls_w1024"] = PooledHead(1024, use_cls=True)
    heads.update({"grid_flat": GridFlatHead(), "conv_pool": ConvPoolHead(),
                  "attn_1q": AttnHead(1), "attn_1q_pos": AttnHead(1, pos=True),
                  "attn_4q": AttnHead(4)})
    return {k: h.to(dev) for k, h in heads.items()}


def geodesic_deg(a, b):
    tr = (a.transpose(1, 2) @ b).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


@torch.no_grad()
def extract(dino, ds, n, batch, workers, dev, tag):
    """이미지 → 동결 DINOv3 공간 특징·CLS (fp16, CPU) + 회전행렬. 표본은 probe_rotation.py 와 같은 시드 0 순열."""
    from pytorch3d.transforms import axis_angle_to_matrix

    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                                         batch_size=batch, shuffle=False,
                                         num_workers=workers, drop_last=False)
    feats, clss, rots = [], [], []
    has_cls = True
    t0 = time.time()
    for b, data in enumerate(loader):
        img = data["img"].to(dev)
        try:
            out = dino.get_intermediate_layers(img, n=N_INTERMEDIATE, reshape=True,
                                               return_class_token=True, norm=True)[-1]
            f, c = out[0], out[1]
        except TypeError:
            has_cls = False
            f = dino.get_intermediate_layers(img, n=N_INTERMEDIATE, reshape=True, norm=True)[-1]
            c = torch.zeros(f.shape[0], DIM, device=dev)
        feats.append(f.flatten(2).half().cpu())
        clss.append(c.half().cpu())
        rots.append(axis_angle_to_matrix(data["rotation"]).cpu())
        if b % 40 == 0:
            print(f"  [{tag}] {b * batch}/{n}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    F, C, R = torch.cat(feats), torch.cat(clss), torch.cat(rots)
    print(f"  [{tag}] 특징 {tuple(F.shape)} CLS {tuple(C.shape)} has_cls={has_cls}", flush=True)
    return F, C, R, has_cls


@torch.no_grad()
def eval_heads(heads, F, C, R, dev, batch=256):
    from pytorch3d.transforms import rotation_6d_to_matrix

    for h in heads.values():
        h.eval()
    out = {k: [] for k in heads}
    for i in range(0, len(F), batch):
        x, c = F[i:i + batch].to(dev).float(), C[i:i + batch].to(dev).float()
        gt = R[i:i + batch].to(dev)
        for k, h in heads.items():
            out[k].append(geodesic_deg(rotation_6d_to_matrix(h(x, c)), gt).cpu())
    for h in heads.values():
        h.train()
    return {k: torch.cat(v) for k, v in out.items()}


def describe_labels() -> None:
    """COCO 회전 라벨의 출처를 기록한다 (npz 키 목록)."""
    for rel in ("datasets/coco/coco.npz", "datasets/3DPW/3DPW_validation.npz"):
        with np.load(REPO / rel, allow_pickle=True) as d:
            keys = list(d.keys())
            print(f"[labels] {rel}: n={len(d[keys[0]])} keys={keys}", flush=True)


def train_seed(seed, F_tr, C_tr, R_tr, evalsets, p, dev, has_cls) -> dict:
    from pytorch3d.transforms import rotation_6d_to_matrix

    torch.manual_seed(seed)
    np.random.seed(seed)
    heads = make_heads(dev, has_cls)
    params = {k: sum(x.numel() for x in h.parameters()) for k, h in heads.items()}
    opts = {k: torch.optim.AdamW(h.parameters(), lr=float(p["lr"]), weight_decay=1e-3)
            for k, h in heads.items()}
    mse = nn.MSELoss()
    bs, t0 = int(p["batch"]), time.time()
    gen = torch.Generator().manual_seed(1000 + seed)
    print(f"\n@seed={seed}  params(M)=" + " ".join(f"{k}:{v / 1e6:.2f}" for k, v in params.items()),
          flush=True)
    for ep in range(int(p["epochs"])):
        perm = torch.randperm(len(F_tr), generator=gen)
        for i in range(0, len(F_tr) - bs + 1, bs):
            sl = perm[i:i + bs]
            x, c, gt = F_tr[sl].to(dev).float(), C_tr[sl].to(dev).float(), R_tr[sl].to(dev)
            for k, h in heads.items():
                opts[k].zero_grad()
                mse(rotation_6d_to_matrix(h(x, c)), gt).backward()
                opts[k].step()
        if ep % 20 == 0 or ep == int(p["epochs"]) - 1:
            print(f"  seed {seed} ep {ep:3d}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = {}
    for split, (Fs, Cs, Rs) in {"train": (F_tr, C_tr, R_tr), **evalsets}.items():
        e = eval_heads(heads, Fs, Cs, Rs, dev)
        for k in heads:
            res[f"{k}/{split}"] = (float(e[k].mean()), float(e[k].median()))
    for k in heads:
        tr, ho, dv = res[f"{k}/train"], res[f"{k}/coco_ho"], res[f"{k}/3dpw_val"]
        print(f"  {k:13s} train {tr[0]:5.1f}°  COCO-ho {ho[0]:5.1f}° (med {ho[1]:5.1f})  "
              f"3DPW {dv[0]:5.1f}° (med {dv[1]:5.1f})", flush=True)
        print(f"@seed{seed}_{k}_train={tr[0]:.2f} @seed{seed}_{k}_ho={ho[0]:.2f} "
              f"@seed{seed}_{k}_ho_med={ho[1]:.2f} @seed{seed}_{k}_3dpw={dv[0]:.2f} "
              f"@seed{seed}_{k}_3dpw_med={dv[1]:.2f}", flush=True)
    del heads, opts
    torch.cuda.empty_cache()
    return {"params": params, "res": res}


def main() -> int:
    if not (REPO / WEIGHTS).exists():
        raise SystemExit(f"DINOv3 가중치 없음: {REPO / WEIGHTS}")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf

    from vq_hps import DatasetHMR

    p = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    seeds = [int(s) for s in str(p["seeds"]).split(",")]
    OmegaConf.load("configs/config.yaml")
    dev = torch.device("cuda")
    print(f"params={p}", flush=True)
    describe_labels()

    torch.manual_seed(0)
    dino = torch.hub.load(repo_or_dir="", model="dinov3_vitl16", source="local",
                          pretrained=True, weights=WEIGHTS, check_hash=False)
    for q in dino.parameters():
        q.requires_grad_(False)
    dino = dino.to(dev).eval()
    print(f"[layer] get_intermediate_layers(n={N_INTERMEDIATE})[-1] of {len(dino.blocks)} blocks, norm=True",
          flush=True)

    print("\n[1] 특징 추출 (한 번만)", flush=True)
    coco = DatasetHMR(dataset_file="datasets/coco/coco.npz", augment=False, proportion=1.0)
    dpw = DatasetHMR(dataset_file="datasets/3DPW/3DPW_validation.npz", augment=False, proportion=1.0)
    Fc, Cc, Rc, has_cls = extract(dino, coco, int(p["n_train"]), int(p["extract_batch"]),
                                  int(p["workers"]), dev, "coco")
    Fd, Cd, Rd, _ = extract(dino, dpw, int(p["n_val3dpw"]), int(p["extract_batch"]),
                            int(p["workers"]), dev, "3dpw")
    del dino
    torch.cuda.empty_cache()

    n_ho = int(float(p["holdout"]) * len(Fc))
    evalsets = {"coco_ho": (Fc[:n_ho], Cc[:n_ho], Rc[:n_ho]), "3dpw_val": (Fd, Cd, Rd)}
    F_tr, C_tr, R_tr = Fc[n_ho:], Cc[n_ho:], Rc[n_ho:]
    print(f"학습 {len(F_tr)}  COCO 홀드아웃 {n_ho}  3DPW val {len(Fd)}", flush=True)

    print("\n[2] 헤드 학습 (시드별)", flush=True)
    allres = {s: train_seed(s, F_tr, C_tr, R_tr, evalsets, p, dev, has_cls) for s in seeds}

    print("\n===== 시드 요약 (평균 ± 표준편차) =====", flush=True)
    names = list(allres[seeds[0]]["params"])
    summary = {}
    for k in names:
        row = {"params_M": allres[seeds[0]]["params"][k] / 1e6}
        for split in ("train", "coco_ho", "3dpw_val"):
            for j, stat in enumerate(("mean", "median")):
                v = np.array([allres[s]["res"][f"{k}/{split}"][j] for s in seeds])
                row[f"{split}_{stat}"] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        summary[k] = row
        f = lambda key: f"{row[key][0]:5.2f}±{row[key][1]:4.2f}"  # noqa: E731
        print(f"{k:13s} {row['params_M']:5.2f}M  train {f('train_mean')}  "
              f"COCO-ho {f('coco_ho_mean')} (med {f('coco_ho_median')})  "
              f"3DPW {f('3dpw_val_mean')} (med {f('3dpw_val_median')})", flush=True)
    print("SUMMARY_JSON " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
