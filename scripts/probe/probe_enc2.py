"""동결 인코더 셋(DINOv3·DINOv2·CLIP)에서 전역 방향 판독기 대조 + 좌우반전 불변성 검사.

IVC 투고 전 보강:
  (1) 'global average pooling 이 방향을 지운다'가 DINOv3 하나의 성질인지, 인코더를 바꿔도 성립하는지.
      probe_rotation2.py 와 같은 헤드 7종·표본·라벨·손실·epoch 을 쓴다.
  (2) Discussion 가설: 평균 풀링 특징이 좌우반전에 거의 불변이라 방향(특히 방위각)이 지워진다.
      - 원본/반전 영상의 평균 풀링 특징 코사인 vs 공간 격자(반전 되돌림) 코사인 vs 서로 다른 영상 기준선
      - 학습된 헤드가 원본과 거울상(반전) 영상에 서로 다른 방향을 내는지(거울 구분 능력)

CQ_PARAMS: enc (dinov3|dinov2|clip), seeds, n_train, n_val3dpw, epochs, batch, lr
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
DINOV3_WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DEFAULTS = dict(enc="dinov2", n_train=9000, n_val3dpw=1000, epochs=60, batch=256, lr=3e-4,
                workers=12, extract_batch=64, holdout=0.12, seeds="0,1,2")
DIM = 1024
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
# 좌우반전의 회전 거울상: R_m = S R S, S = diag(-1, 1, 1)  (axis-angle (x,y,z) -> (x,-y,-z))
MIRROR_S = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


class Encoder:
    """영상(ImageNet 정규화, 224) -> 공간 특징 (B, D, g*g) 와 CLS (B, D). 모두 동결."""

    def __init__(self, name: str, dev):
        self.name, self.dev = name, dev
        if name == "dinov3":
            os.chdir(REPO)
            self.m = torch.hub.load(repo_or_dir="", model="dinov3_vitl16", source="local",
                                    pretrained=True, weights=DINOV3_WEIGHTS, check_hash=False)
            self.grid = 14
        elif name == "dinov2":
            from transformers import Dinov2Model
            self.m = Dinov2Model.from_pretrained("facebook/dinov2-large")
            self.grid = 16
        elif name == "clip":
            from transformers import CLIPVisionModel
            self.m = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            self.grid = 16
        else:
            raise SystemExit(f"unknown encoder {name}")
        for q in self.m.parameters():
            q.requires_grad_(False)
        self.m = self.m.to(dev).eval()

    @torch.no_grad()
    def __call__(self, img):
        b = img.shape[0]
        if self.name == "dinov3":
            # 모델과 같은 호출: 마지막 23블록 중 [-1] = 최종 블록, norm 적용
            out = self.m.get_intermediate_layers(img, n=23, reshape=True,
                                                 return_class_token=True, norm=True)[-1]
            return out[0].flatten(2), out[1]
        if self.name == "dinov2":
            h = self.m(pixel_values=img).last_hidden_state          # (B, 1+256, D), layernorm 적용됨
        else:
            raw = img * IMNET_STD.to(img) + IMNET_MEAN.to(img)
            x = (raw - CLIP_MEAN.to(img)) / CLIP_STD.to(img)
            h = self.m.vision_model.post_layernorm(self.m(pixel_values=x).last_hidden_state)
        cls, pt = h[:, 0], h[:, 1:]
        g = self.grid
        assert pt.shape[1] == g * g, (pt.shape, g)
        return pt.transpose(1, 2).reshape(b, DIM, g * g), cls


def rot_out(width: int) -> nn.Linear:
    lin = nn.Linear(width, 6)
    nn.init.zeros_(lin.weight)
    with torch.no_grad():
        lin.bias.copy_(torch.tensor(IDENTITY_6D))
    return lin


class PooledHead(nn.Module):
    def __init__(self, width=1024, dropout=0.5, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.mlp = nn.Sequential(nn.Linear(DIM, width), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):
        x = cls if self.use_cls else spatial.mean(-1)
        return self.rot(self.mlp(x))


class GridFlatHead(nn.Module):
    """2×2 평균으로 격자를 절반으로(14→7, 16→8) 줄이고 채널 투영 후 펼친다. attention 없음."""

    def __init__(self, grid, proj=96, width=1024, dropout=0.5):
        super().__init__()
        self.grid = grid
        self.down = nn.AvgPool2d(2)
        self.proj = nn.Conv2d(DIM, proj, 1)
        self.mlp = nn.Sequential(nn.Linear(proj * (grid // 2) ** 2, width), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):
        b = spatial.shape[0]
        x = self.proj(self.down(spatial.view(b, DIM, self.grid, self.grid)))
        return self.rot(self.mlp(x.flatten(1)))


class ConvPoolHead(nn.Module):
    def __init__(self, grid, ch=256, width=1024, dropout=0.5):
        super().__init__()
        self.grid = grid
        self.conv = nn.Sequential(nn.Conv2d(DIM, ch, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(ch, width), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(width)

    def forward(self, spatial, cls):
        b = spatial.shape[0]
        x = self.conv(spatial.view(b, DIM, self.grid, self.grid)).mean((2, 3))
        return self.rot(self.mlp(x))


class AttnHead(nn.Module):
    def __init__(self, grid, n_q=1, pos=False, nhead=8, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_q, 1, DIM))
        nn.init.normal_(self.query, std=0.02)
        self.pos = nn.Parameter(torch.zeros(grid * grid, 1, DIM)) if pos else None
        if pos:
            nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(DIM, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM * n_q, DIM), nn.GELU(), nn.Dropout(dropout))
        self.rot = rot_out(DIM)

    def forward(self, spatial, cls):
        mem = spatial.permute(2, 0, 1)
        key = mem + self.pos if self.pos is not None else mem
        q = self.query.expand(-1, mem.shape[1], -1)
        x, _ = self.attn(q, key, mem)
        x = self.norm(x).permute(1, 0, 2).flatten(1)
        return self.rot(self.mlp(x))


def make_heads(dev, grid) -> dict:
    heads = {"pooled_w1024": PooledHead(1024), "pooled_w2048": PooledHead(2048),
             "cls_w1024": PooledHead(1024, use_cls=True),
             "grid_flat": GridFlatHead(grid), "conv_pool": ConvPoolHead(grid),
             "attn_1q": AttnHead(grid, 1), "attn_1q_pos": AttnHead(grid, 1, pos=True),
             "attn_4q": AttnHead(grid, 4)}
    return {k: h.to(dev) for k, h in heads.items()}


def geodesic_deg(a, b):
    tr = (a.transpose(1, 2) @ b).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


@torch.no_grad()
def extract(enc, ds, idx, batch, workers, dev, tag, flip=False):
    """표본 idx 의 동결 특징 (fp16, CPU) + 회전행렬. flip=True 면 영상을 좌우반전해 넣는다."""
    from pytorch3d.transforms import axis_angle_to_matrix

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                                         batch_size=batch, shuffle=False,
                                         num_workers=workers, drop_last=False)
    feats, clss, rots = [], [], []
    t0 = time.time()
    for b, data in enumerate(loader):
        img = data["img"].to(dev)
        if flip:
            img = torch.flip(img, dims=[3])
        f, c = enc(img)
        feats.append(f.half().cpu())
        clss.append(c.half().cpu())
        rots.append(axis_angle_to_matrix(data["rotation"]).cpu())
        if b % 40 == 0:
            print(f"  [{tag}] {b * batch}/{len(idx)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    F, C, R = torch.cat(feats), torch.cat(clss), torch.cat(rots)
    print(f"  [{tag}] 특징 {tuple(F.shape)} CLS {tuple(C.shape)}", flush=True)
    return F, C, R


def cos_rows(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)


def flip_invariance(F, Ff, grid, tag) -> dict:
    """원본 F 와 반전 Ff (같은 표본) 의 특징 유사도. 기준선은 서로 다른 영상(한 칸 밀기)."""
    n = len(F)
    pool, poolf = F.float().mean(-1), Ff.float().mean(-1)
    mu = pool.mean(0, keepdim=True)                        # 이방성 제거용 데이터 평균
    g = F.float().view(n, DIM, grid, grid)
    gf = Ff.float().view(n, DIM, grid, grid)
    gf_back = torch.flip(gf, dims=[3])                     # 반전된 격자를 다시 되돌린다
    gmu = g.mean(0, keepdim=True)
    other = torch.roll(torch.arange(n), 1)
    r = {
        "pool_flip": cos_rows(pool, poolf).mean().item(),
        "pool_other": cos_rows(pool, pool[other]).mean().item(),
        "pool_flip_centered": cos_rows(pool - mu, poolf - mu).mean().item(),
        "pool_other_centered": cos_rows(pool - mu, pool[other] - mu).mean().item(),
        "grid_flipback": cos_rows((g - gmu).flatten(1), (gf_back - gmu).flatten(1)).mean().item(),
        "grid_noflipback": cos_rows((g - gmu).flatten(1), (gf - gmu).flatten(1)).mean().item(),
        "grid_other": cos_rows((g - gmu).flatten(1), (g[other] - gmu).flatten(1)).mean().item(),
    }
    print(f"[flip/{tag}] pooled cos: flip {r['pool_flip']:.4f} vs other {r['pool_other']:.4f} | "
          f"centered: flip {r['pool_flip_centered']:.4f} vs other {r['pool_other_centered']:.4f} | "
          f"grid(centered): flip-back {r['grid_flipback']:.4f}, no-flip-back {r['grid_noflipback']:.4f}, "
          f"other {r['grid_other']:.4f}", flush=True)
    print(" ".join(f"@{tag}_{k}={v:.4f}" for k, v in r.items()), flush=True)
    return r


@torch.no_grad()
def mirror_test(heads, F, C, R, Ff, Cf, dev, tag) -> dict:
    """원본/반전 영상에 대한 예측이 거울상만큼 달라지는가.

    gt_pair  = angle(R, S R S)         : 바뀌어야 할 양
    pred_pair= angle(P(x), P(flip x))  : 실제로 바뀐 양
    err_flip = angle(P(flip x), S R S) : 반전 영상에서의 오차
    """
    from pytorch3d.transforms import rotation_6d_to_matrix

    S = MIRROR_S
    Rm = S @ R @ S
    gt_pair = geodesic_deg(R, Rm)
    out = {}
    for k, h in heads.items():
        h.eval()
        ps, pf = [], []
        for i in range(0, len(F), 256):
            ps.append(rotation_6d_to_matrix(h(F[i:i + 256].to(dev).float(), C[i:i + 256].to(dev).float())).cpu())
            pf.append(rotation_6d_to_matrix(h(Ff[i:i + 256].to(dev).float(), Cf[i:i + 256].to(dev).float())).cpu())
        h.train()
        P, Pf = torch.cat(ps), torch.cat(pf)
        pred_pair = geodesic_deg(P, Pf)
        r = {"gt_pair": gt_pair.mean().item(), "pred_pair": pred_pair.mean().item(),
             "err_orig": geodesic_deg(P, R).mean().item(), "err_flip": geodesic_deg(Pf, Rm).mean().item(),
             "pred_vs_gt_pair_corr": float(np.corrcoef(pred_pair.numpy(), gt_pair.numpy())[0, 1])}
        out[k] = r
        print(f"[mirror/{tag}] {k:13s} gt-pair {r['gt_pair']:5.1f}°  pred-pair {r['pred_pair']:5.1f}°  "
              f"err orig {r['err_orig']:5.1f}°  err flip {r['err_flip']:5.1f}°  corr {r['pred_vs_gt_pair_corr']:.2f}",
              flush=True)
        print(f"@{tag}_{k}_mirror_gtpair={r['gt_pair']:.2f} @{tag}_{k}_mirror_predpair={r['pred_pair']:.2f} "
              f"@{tag}_{k}_mirror_errflip={r['err_flip']:.2f}", flush=True)
    return out


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


def train_seed(seed, F_tr, C_tr, R_tr, evalsets, p, dev, grid, tag):
    from pytorch3d.transforms import rotation_6d_to_matrix

    torch.manual_seed(seed)
    np.random.seed(seed)
    heads = make_heads(dev, grid)
    params = {k: sum(x.numel() for x in h.parameters()) for k, h in heads.items()}
    opts = {k: torch.optim.AdamW(h.parameters(), lr=float(p["lr"]), weight_decay=1e-3)
            for k, h in heads.items()}
    mse = nn.MSELoss()
    bs, t0 = int(p["batch"]), time.time()
    gen = torch.Generator().manual_seed(1000 + seed)
    print(f"\n@seed={seed}  [{tag}] params(M)=" + " ".join(f"{k}:{v / 1e6:.2f}" for k, v in params.items()),
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
        print(f"  [{tag}] {k:13s} train {tr[0]:5.1f}°  COCO-ho {ho[0]:5.1f}° (med {ho[1]:5.1f})  "
              f"3DPW {dv[0]:5.1f}° (med {dv[1]:5.1f})", flush=True)
        print(f"@{tag}_seed{seed}_{k}_train={tr[0]:.2f} @{tag}_seed{seed}_{k}_ho={ho[0]:.2f} "
              f"@{tag}_seed{seed}_{k}_3dpw={dv[0]:.2f}", flush=True)
    return heads, {"params": params, "res": res}


def main() -> int:
    p = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    enc_name = str(p["enc"])
    seeds = [int(s) for s in str(p["seeds"]).split(",")]
    print(f"params={p}", flush=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf

    from vq_hps import DatasetHMR

    OmegaConf.load("configs/config.yaml")
    dev = torch.device("cuda")
    enc = Encoder(enc_name, dev)
    os.chdir(REPO)
    print(f"[encoder] {enc_name} grid {enc.grid}x{enc.grid} dim {DIM}", flush=True)

    coco = DatasetHMR(dataset_file="datasets/coco/coco.npz", augment=False, proportion=1.0)
    dpw = DatasetHMR(dataset_file="datasets/3DPW/3DPW_validation.npz", augment=False, proportion=1.0)
    # probe_rotation(2).py 와 같은 시드 0 순열
    idx_c = torch.randperm(len(coco), generator=torch.Generator().manual_seed(0))[:int(p["n_train"])]
    idx_d = torch.randperm(len(dpw), generator=torch.Generator().manual_seed(0))[:int(p["n_val3dpw"])]
    n_ho = int(float(p["holdout"]) * len(idx_c))
    eb, wk = int(p["extract_batch"]), int(p["workers"])

    print("\n[1] 특징 추출", flush=True)
    Fc, Cc, Rc = extract(enc, coco, idx_c, eb, wk, dev, f"{enc_name}/coco")
    Fd, Cd, Rd = extract(enc, dpw, idx_d, eb, wk, dev, f"{enc_name}/3dpw")
    Ffl, Cfl, _ = extract(enc, coco, idx_c[:n_ho], eb, wk, dev, f"{enc_name}/coco_ho_flip", flip=True)
    grid = enc.grid
    del enc
    torch.cuda.empty_cache()

    print("\n[2] 좌우반전 불변성 (COCO 홀드아웃)", flush=True)
    flip_stats = flip_invariance(Fc[:n_ho], Ffl, grid, enc_name)

    evalsets = {"coco_ho": (Fc[:n_ho], Cc[:n_ho], Rc[:n_ho]), "3dpw_val": (Fd, Cd, Rd)}
    F_tr, C_tr, R_tr = Fc[n_ho:], Cc[n_ho:], Rc[n_ho:]
    print(f"학습 {len(F_tr)}  COCO 홀드아웃 {n_ho}  3DPW val {len(Fd)}", flush=True)

    print("\n[3] 헤드 학습 (시드별)", flush=True)
    allres, mirror = {}, None
    for s in seeds:
        heads, r = train_seed(s, F_tr, C_tr, R_tr, evalsets, p, dev, grid, enc_name)
        allres[s] = r
        if s == seeds[0]:
            print("\n[4] 거울상 구분 (시드 첫째 헤드, COCO 홀드아웃)", flush=True)
            mirror = mirror_test(heads, Fc[:n_ho], Cc[:n_ho], Rc[:n_ho], Ffl, Cfl, dev, enc_name)
        del heads
        torch.cuda.empty_cache()

    print("\n===== 시드 요약 (평균 ± 표준편차) =====", flush=True)
    summary = {}
    for k in allres[seeds[0]]["params"]:
        row = {"params_M": allres[seeds[0]]["params"][k] / 1e6}
        for split in ("train", "coco_ho", "3dpw_val"):
            for j, stat in enumerate(("mean", "median")):
                v = np.array([allres[s]["res"][f"{k}/{split}"][j] for s in seeds])
                row[f"{split}_{stat}"] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        summary[k] = row
        f = lambda key: f"{row[key][0]:5.2f}±{row[key][1]:4.2f}"  # noqa: E731
        print(f"[{enc_name}] {k:13s} {row['params_M']:5.2f}M  train {f('train_mean')}  "
              f"COCO-ho {f('coco_ho_mean')}  3DPW {f('3dpw_val_mean')}", flush=True)
    print("SUMMARY_JSON " + json.dumps({"enc": enc_name, "grid": grid, "summary": summary,
                                        "flip": flip_stats, "mirror": mirror}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
