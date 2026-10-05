# Global Average Pooling Loses Orientation

Code for the paper **"Global Average Pooling Loses Orientation: Frozen Encoders in Token-Based Human Mesh Recovery"** (Changmin Lee, HyeokHyeon Kwon; under review).

The system under study is a token-based human mesh recovery model in the style of [VQ-HPS](https://github.com/g-fiche/VQ-HPS): a frozen DINOv3 ViT-L encoder, a frozen Mesh VQ-VAE codebook (54 tokens, 512 entries), a transformer that classifies the 54 mesh tokens, and a separate head for the global orientation of the decoded mesh. This repository contains the code that produces the paper's measurements:

- the two readout defects and their fixes (orientation head reading a globally average-pooled vector; a causal mask in the token head that acts on the batch axis),
- the 2x2 training ablation, seed replication and encoder unfreezing,
- the orientation probes on cached frozen DINOv3, DINOv2 and CLIP features,
- the rotation-injection sweep that shows PA-MPJPE is blind to global rotation,
- the ground-truth-orientation substitution, the Mesh VQ-VAE reconstruction floor, and the mirror-invariance test.

It does **not** contain datasets, body models, model weights or third-party predictions; see [Data preparation](#data-preparation).

## Repository layout

```
patches/
  01_vqhps_to_dinov3mesh.patch        # VQ-HPS (commit cf74370) -> our DINOv3 port
  02_transformer_predictor_and_eval_mode.patch
  LICENSE-VQ-HPS.md                   # MIT licence of the upstream code the patches modify
scripts/
  train/apply_model_edits.py          # adds the two fixes as switches (head_axis, rot_from)
  train/train.py                      # 30-epoch runs of the 2x2 ablation (seed 0)
  train/train_unfreeze.py             # seed replication, frozen and unfrozen-encoder runs
  probe/probe_rotation2.py            # DINOv3 orientation probe, 8 readouts x 3 seeds
  probe/probe_enc2.py                 # same probe on DINOv2 / CLIP + mirror invariance
  eval/eval_oracle.py                 # orientation error, per-axis split, GT-orientation substitution
  eval/eval_orient.py                 # orientation error of the fixed checkpoints
  eval/eval_valanom.py                # why GT substitution raises PVE on the validation subset
  eval/rotation_blindness.py          # injected global rotation vs PA-MPJPE / MPJPE
  eval/vertex_radius.py               # joint -> vertex conversion factor for the rotation sweep
  eval/vqvae_oracle.py                # Mesh VQ-VAE reconstruction floor on 3DPW test
  eval/diag_batchmix.py               # batch-position dependence of the as-received token head
figures/
  make_system_fig.py                  # Fig. 1 (system diagram)
  make_ga.py                          # graphical abstract
results/                              # the numbers reported in the paper, as CSV
```

## Environment

```bash
uv sync                      # Python 3.11, torch 2.9 (CUDA 12.8 wheels)
```

PyTorch3D has no PyPI wheel. Build it against the installed torch and CUDA, then install it into the same environment, for example:

```bash
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
```

All scripts read their parameters from a JSON string in the `CQ_PARAMS` environment variable (an artefact of the experiment runner we used; an empty value means "defaults"). Paths are taken from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `VQHPS_ROOT` | patched VQ-HPS working tree (see below) | `third_party/VQ-HPS` |
| `DINOV3_WEIGHTS` | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| `MESH_VQVAE_DIR` | released Mesh VQ-VAE checkpoint `mesh_vqvae_54` | `checkpoints/mesh_vqvae_54` |
| `CKPT_<NAME>` | trained regressor checkpoints for the evaluation scripts (`CKPT_ASRECV_S1`, `CKPT_BOTH_S0`, `CKPT_UNFROZEN`, `CKPT_POOLED_E103`, `CKPT_BOTH`, `CKPT_BATCHMIX`, ...) | `$VQHPS_ROOT/checkpoint/MESH_REGRESSOR/<name>/model_checkpoint` |
| `PRED_DIR` | third-party per-frame predictions on 3DPW test (rotation sweep only) | `data/predictions` |
| `TOKENHMR_3DPW_NPZ`, `SMPL_DIR` | 3DPW test annotation file in TokenHMR's format and SMPL-H body models (VQ-VAE floor, vertex radius) | `data/tokenhmr/datasets/3dpw_test.npz`, `data/body_models` |

## Building the model code

The model is a port of VQ-HPS (MIT licence) to a DINOv3 backbone. We ship it as patches against the upstream repository rather than as a copy:

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/g-fiche/VQ-HPS.git && cd VQ-HPS
git checkout cf7437050e1d291c5135c07a97d5ec4db183ce50
git apply ../../patches/01_vqhps_to_dinov3mesh.patch
git apply ../../patches/02_transformer_predictor_and_eval_mode.patch
cd ../.. && VQHPS_ROOT=third_party/VQ-HPS uv run python scripts/train/apply_model_edits.py
```

`apply_model_edits.py` adds two constructor switches whose defaults reproduce the as-received behaviour:

- `head_axis="batch"` (as received) / `"token"` (fixed): the token head receives a `(54, B, C)` tensor that its causal self-attention reads as `(B, T, C)`, so the mask runs over the batch. `"token"` transposes it and removes the mask. The same pattern is present in the public VQ-HPS code at the commit above (`vq_hps/model/mesh_regressor/transformer.py`).
- `rot_from="pooled"` (as received) / `"attn"` (fixed): the orientation and camera are regressed from the globally average-pooled DINOv3 vector, or by a learned query cross-attending to the spatial features.

The DINOv3 backbone is loaded through `torch.hub` from the working tree. Copy the `dinov3/` package and `hubconf.py` from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) into `third_party/VQ-HPS/`, and obtain the ViT-L/16 weights from Meta under the DINOv3 License (they are gated and not redistributed here).

The Mesh VQ-VAE package (`mesh_vq_vae/`), its released checkpoint and the `body_models/` folder (SMPL-H, connection matrices, joint regressors, T-pose) are set up as described in the [VQ-HPS](https://github.com/g-fiche/VQ-HPS) and [Mesh-VQ-VAE](https://github.com/g-fiche/Mesh-VQ-VAE) READMEs.

## Data preparation

None of the datasets may be redistributed. Download them from the official pages, accepting their licences, and convert them with the VQ-HPS preprocessing scripts (`preprocess_data/`, extended by patch 01 for MPI-INF-3DHP frame extraction):

| Dataset | Use | Official page |
|---|---|---|
| 3DPW | validation / test (and train split for the VQ-VAE fine-tune) | https://virtualhumans.mpi-inf.mpg.de/3DPW/ |
| COCO (with SMPL pseudo ground truth as in VQ-HPS) | training, probe images | https://cocodataset.org/ |
| MPII | training | http://human-pose.mpi-inf.mpg.de/ |
| MPI-INF-3DHP | training | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ |
| AMASS | Mesh VQ-VAE training (upstream) | https://amass.is.tue.mpg.de/ |
| SMPL / SMPL-H body models | meshes | https://smpl.is.tue.mpg.de/, https://mano.is.tue.mpg.de/ |

The resulting files are expected under `$VQHPS_ROOT/datasets/` (`coco/coco.npz`, `3DPW/3DPW_validation.npz`, `3DPW/3DPW_test.npz`, and the training list in `configs/full_train.txt`).

The rotation-injection sweep (Table "inject") uses the per-frame 3DPW test predictions of TokenHMR, HMR 2.0 and SPIN. We do not redistribute them; the scripts that regenerate them from the official model releases are in the companion repository [`hmr-holdout-refinement`](https://github.com/playidea-lab/hmr-holdout-refinement) (`scripts/predictions/`). Place the resulting `tokenhmr_3dpw_cuda.pkl`, `4dhumans_3dpw_cuda.pkl` and `spin_3dpw_14j.pkl` in `$PRED_DIR`.

## Reproducing the paper

Run from the repository root with `VQHPS_ROOT` and the other variables set. Training uses one GPU; a 30-epoch run takes about half a day on an RTX 5080-class card, data loading being the bottleneck.

| Paper item | Script | Command / parameters |
|---|---|---|
| Table "configs" (2x2), Table "factorial", Fig. "factorial" — seed-0 rows | `scripts/train/train.py` | `CQ_PARAMS='{"epochs":30}'` (as received); add `"head_axis":"token"`, `"rot_from":"attn"` or both for the other three cells |
| Table "configs" — rows with three seeds | `scripts/train/train_unfreeze.py` | `CQ_PARAMS='{"seed":S,"epochs":30,"unfreeze_blocks":0,"head_axis":"batch","rot_from":"pooled"}'` (as received) or `"head_axis":"token","rot_from":"attn"` (both fixes), `S` in 1, 2 |
| Table "configs" last row, Table "unfreeze" | `scripts/train/train_unfreeze.py` | `CQ_PARAMS='{"epochs":60,"unfreeze_blocks":4,"backbone_lr_mult":0.1,"lr_step":45}'` |
| Table "probe" (DINOv3, 8 readouts, 3 seeds) | `scripts/probe/probe_rotation2.py` | `CQ_PARAMS='{"seeds":"0,1,2"}'` |
| Table "probe3" (DINOv2, CLIP), Table "flip" | `scripts/probe/probe_enc2.py` | `CQ_PARAMS='{"enc":"dinov2"}'`, `'{"enc":"clip"}'`, and `'{"enc":"dinov3","seeds":"0"}'` for the DINOv3 mirror test |
| Table "inject", Fig. "rotblind" | `scripts/eval/rotation_blindness.py` | needs `$PRED_DIR`; writes `results/rot_blind/results.json` |
| Joint-to-vertex conversion used in the rotation discussion | `scripts/eval/vertex_radius.py` | needs `$PRED_DIR`, `TOKENHMR_3DPW_NPZ`, `SMPL_DIR` |
| Table "oracleR" (orientation error, axis split, GT substitution) | `scripts/eval/eval_oracle.py` | set `CKPT_*` for the evaluated checkpoints |
| Orientation error of the fixed checkpoints | `scripts/eval/eval_orient.py` | `CKPT_BOTH`, `CKPT_UNFROZEN` |
| Validation-subset anomaly (decoder not fully canonical) | `scripts/eval/eval_valanom.py` | `CKPT_BOTH_S0`, `CKPT_UNFROZEN` |
| Table "oracle" (Mesh VQ-VAE floor) | `scripts/eval/vqvae_oracle.py` | `--out_dir outputs/vqvae_oracle --batch 64` |
| Batch-axis mask in the token head (logit change up to 1.006) | `scripts/eval/diag_batchmix.py` | `CKPT_BATCHMIX` = an as-received checkpoint |
| Fig. 1, graphical abstract | `figures/make_system_fig.py`, `figures/make_ga.py` | `uv run python figures/make_system_fig.py` |

The numbers printed by these scripts are collected in `results/*.csv` exactly as they appear in the paper (Fig. "factorial" and Fig. "rotblind" plot `results/tab_configs_2x2.csv` and `results/tab_rotation_injection.csv`; regenerate both with `uv run python figures/make_figs.py`, output in `$FIG_DIR`, default `figures/out`).

### Notes on what was changed for release

- Hard-coded machine paths were replaced by the environment variables above; checkpoint paths in the evaluation scripts were dated run folders and are now `CKPT_<NAME>` variables. No numerical code was changed.
- `apply_model_edits.py`: the "already applied" check now tests only for the new string, because one replacement contains its own original and the old check re-applied it on a second run. A single run (the one used for the paper) is unaffected.
- `vqvae_oracle.py` originally imported a verbatim copy of the Mesh-VQ-VAE model files; it now imports the same files from `mesh_vq_vae` in the patched tree. `diag_batchmix.py` now contains the three helper functions it previously imported from a sibling diagnostic script.

## Licence

Our code is released under the Apache License 2.0 (`LICENSE`). The patches in `patches/` modify VQ-HPS, which is MIT-licensed (`patches/LICENSE-VQ-HPS.md`); the modified portions remain under that licence. DINOv3 code and weights are subject to Meta's DINOv3 License and are not included. Datasets and body models are subject to their own licences.

## Citation

```bibtex
@unpublished{lee2026gap_orientation,
  title  = {Global Average Pooling Loses Orientation: Frozen Encoders in Token-Based Human Mesh Recovery},
  author = {Lee, Changmin and Kwon, HyeokHyeon},
  note   = {Under review},
  year   = {2026}
}
```
