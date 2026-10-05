"""val 이상 징후 진단 — GT 회전을 대입하면 왜 val 에서만 PVE 가 나빠지는가 (평가만).

eval_oracle.py: both_s0 val 94→98, unfrozen val 78→89 로 GT 회전 대입이 PVE 를
악화시켰고, test 에서는 정상(91.3→85.9)이었다. 표본별로 기록해 원인을 가른다.
  - npz 키(pose_cam/pose), gender, imgname 의 closeup 여부, 시퀀스
  - GT 내부 일관성: Kabsch(local_mesh_gt → mesh_gt) 와 R_gt 의 각도 (0 이어야 정상)
  - Kabsch(pred_canonical → mesh_gt) 와 R_gt·R_pred 의 각도
val 표본은 시드 고정(0)으로 1/10 을 뽑는다 (원래 코드는 random.sample 시드 미고정).
체크포인트 경로: CKPT_BOTH_S0, CKPT_UNFROZEN 환경변수 (eval_oracle.py 와 같은 규약).
CQ_PARAMS: models (쉼표), val_prop, test_n
"""

import json
import os
import random
import sys
from collections import defaultdict
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
    "both_s0": (ckpt("both_s0"), "token", "attn"),
    "unfrozen": (ckpt("unfrozen"), "token", "attn"),
}
DEFAULTS = dict(models="both_s0,unfrozen", val_prop=0.1, test_n=3500)
BATCH = 32
MM = 1000.0


def geodesic_deg(a, b):
    rel = torch.matmul(a.transpose(1, 2), b)
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def kabsch(src, dst):
    """중심화한 점군 src → dst 의 최적 회전 (스케일 없음). (B,N,3) → (B,3,3), dst ≈ R src."""
    s = src - src.mean(1, keepdim=True)
    d = dst - dst.mean(1, keepdim=True)
    h = s.transpose(1, 2) @ d
    u, _, vt = torch.linalg.svd(h)
    det = torch.det(vt.transpose(1, 2) @ u.transpose(1, 2))
    dmat = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], -1))
    return vt.transpose(1, 2) @ dmat @ u.transpose(1, 2)


def describe_npz(path):
    with np.load(path, allow_pickle=True) as d:
        keys = list(d.keys())
        names = [str(x) for x in d["imgname"]]
        info = {"n": len(names), "keys": keys, "has_pose_cam": "pose_cam" in keys,
                "closeup_frac": float(np.mean(["closeup" in x for x in names])),
                "genders": sorted(set(str(g) for g in d["gender"])) if "gender" in keys else ["(none)"]}
        seqs = sorted(set(x.split("/")[-2] for x in names))
        info["n_seq"] = len(seqs)
        info["seqs"] = seqs
    print(f"[npz] {path}: {json.dumps(info)}", flush=True)
    return info


def evaluate(trainer, loader, J, dev):
    from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix

    from vq_hps.utils.eval import v2v

    model = trainer.model
    model.eval()
    rows = []
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
            gt, pr = data["mesh"].float(), out["pred_mesh"].float().cpu()
            can = out["pred_mesh_canonical"].float().cpu()
            orc = (r_g @ can.transpose(2, 1)).transpose(2, 1)
            vv = (MM * v2v(gt, pr, reduction=False)).reshape(n, -1).mean(-1)
            vvo = (MM * v2v(gt, orc, reduction=False)).reshape(n, -1).mean(-1)
            g_int = geodesic_deg(kabsch(data["local_mesh"].float(), gt), r_g)
            r_can = kabsch(can, gt)
            g_can_gt = geodesic_deg(r_can, r_g)
            g_can_pred = geodesic_deg(r_can, r_p)
            rot = geodesic_deg(r_p, r_g)
            for i in range(n):
                name = str(data["imgname"][i])
                rows.append(dict(img=name, seq=name.split("/")[-2], closeup="closeup" in name,
                                 gender=str(data["gender"][i]), rot=float(rot[i]), pve=float(vv[i]),
                                 pve_gtrot=float(vvo[i]), gt_internal=float(g_int[i]),
                                 kabsch_can_vs_gt=float(g_can_gt[i]), kabsch_can_vs_pred=float(g_can_pred[i])))
    return rows


def agg(rs):
    a = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    return dict(n=len(rs), rot=a("rot"), pve=a("pve"), pve_gtrot=a("pve_gtrot"),
                worse_frac=float(np.mean([r["pve_gtrot"] > r["pve"] for r in rs])),
                gt_internal=a("gt_internal"), kabsch_can_vs_gt=a("kabsch_can_vs_gt"),
                kabsch_can_vs_pred=a("kabsch_can_vs_pred"))


def summarize(rows, tag):
    s = {"all": agg(rows)}
    for flag in (True, False):
        sub = [r for r in rows if r["closeup"] == flag]
        if sub:
            s[f"closeup={flag}"] = agg(sub)
    for g in sorted(set(r["gender"] for r in rows)):
        s[f"gender={g}"] = agg([r for r in rows if r["gender"] == g])
    byseq = defaultdict(list)
    for r in rows:
        byseq[r["seq"]].append(r)
    s["by_seq"] = {k: agg(v) for k, v in sorted(byseq.items())}
    a = s["all"]
    print(f"[{tag}] n={a['n']} rot {a['rot']:.2f}° PVE {a['pve']:.2f} → GT-rot {a['pve_gtrot']:.2f} "
          f"(worse in {100 * a['worse_frac']:.1f}%)  GT-internal {a['gt_internal']:.3f}°  "
          f"kabsch(can)→R_gt {a['kabsch_can_vs_gt']:.2f}°  →R_pred {a['kabsch_can_vs_pred']:.2f}°", flush=True)
    for k, v in s.items():
        if k in ("all", "by_seq"):
            continue
        print(f"   {k:16s} n={v['n']:5d} PVE {v['pve']:.1f}→{v['pve_gtrot']:.1f} worse {100 * v['worse_frac']:.0f}%  "
              f"GT-int {v['gt_internal']:.2f}°  kab→Rgt {v['kabsch_can_vs_gt']:.1f}°", flush=True)
    for k, v in s["by_seq"].items():
        print(f"   seq {k:30s} n={v['n']:4d} rot {v['rot']:5.1f}° PVE {v['pve']:6.1f}→{v['pve_gtrot']:6.1f} "
              f"worse {100 * v['worse_frac']:3.0f}%  GT-int {v['gt_internal']:.2f}°  kab→Rgt {v['kabsch_can_vs_gt']:.1f}°",
              flush=True)
    t = tag.replace("/", "_")
    print(f"@{t}_pve={a['pve']:.3f} @{t}_pve_gtrot={a['pve_gtrot']:.3f} @{t}_worse_frac={a['worse_frac']:.3f} "
          f"@{t}_gt_internal={a['gt_internal']:.4f}", flush=True)
    return s


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import tempfile

    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    from mesh_vq_vae import FullyConvAE, MeshVQVAE
    from vq_hps import DatasetHMR, MeshRegressor, MeshRegressor_Train
    from vq_hps.train.train_mesh_regressor import follow_up_regressor as FU

    def _scratch_dir(self):
        d = Path(tempfile.mkdtemp(prefix="eval_follow_"))
        self.path = self.path_samples = self.path_samples_train = d

    FU.Follow.create_directory = _scratch_dir

    p = {**DEFAULTS, **json.loads(os.environ.get("CQ_PARAMS") or "{}")}
    print(f"params={p}", flush=True)
    info = {"val": describe_npz("datasets/3DPW/3DPW_validation.npz"),
            "test": describe_npz("datasets/3DPW/3DPW_test.npz")}

    cfg = OmegaConf.load("configs/config.yaml")
    ref_bm = np.load("body_models/smplh/neutral/model.npz")
    conv = FullyConvAE(cfg.modelconv, test_mode=True)
    vq = MeshVQVAE(conv, **cfg.vqvaemesh)
    vq.load(path_model=VQVAE_DEFAULT)
    conv.init_test_mode()
    J = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()
    J24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()

    ds_val = DatasetHMR(dataset_file="datasets/3DPW/3DPW_validation.npz", augment=False, proportion=1.0)
    ds_test = DatasetHMR(dataset_file="datasets/3DPW/3DPW_test.npz", augment=False, proportion=1.0)
    rng = random.Random(0)
    vidx = sorted(rng.sample(range(len(ds_val)), int(float(p["val_prop"]) * len(ds_val))))
    tidx = sorted(rng.sample(range(len(ds_test)), int(p["test_n"])))
    subsets = {"val": Subset(ds_val, vidx), "test": Subset(ds_test, tidx)}

    out = {"npz": info}
    for name in str(p["models"]).split(","):
        ck, head_axis, rot_from = CONFIGS[name]
        cfg.regressor["head_axis"] = head_axis
        cfg.regressor["rot_from"] = rot_from
        model = MeshRegressor(**cfg.regressor)
        trainer = MeshRegressor_Train(model, ds_val, ds_val, cfg.train,
                                      faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
                                      vqvae_mesh=vq, joints_regressor=J, joints_regressor_smpl=J24)
        state = torch.load(str(ck), map_location="cpu")
        trainer.model.load_state_dict(state["model"], strict=True)
        print(f"\n[{name}] {ck} epoch={state.get('epoch')}", flush=True)
        for split, ds in subsets.items():
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=12, drop_last=True)
            rows = evaluate(trainer, loader, J, trainer.device)
            out[f"{name}/{split}"] = summarize(rows, f"{name}/{split}")
        del trainer, model, state
        torch.cuda.empty_cache()
    print("RESULTS_JSON " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
