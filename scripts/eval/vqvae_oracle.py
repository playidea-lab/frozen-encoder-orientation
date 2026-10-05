#!/usr/bin/env python3
"""Mesh VQ-VAE의 재구성 상한을 3DPW test 정답 메시로 측정한다.

혁현 논문의 구조는 코드북과 디코더를 동결한다. 따라서 GT 메시를 그대로 통과시킨
오차(encode -> quantize -> decode)가 이 구조가 도달할 수 있는 하한이고, 보고된
PVE 중 얼마가 코드북 탓이고 얼마가 매핑 탓인지를 가른다.

학습은 없다. 3DPW test의 SMPL 파라미터로 SMPL-H 메시를 만들고 왕복시킨다.

원 실행은 Mesh-VQ-VAE 의 model 하위 파일(내용 동일)을 복사한 패키지를 썼다. 공개판은
패치된 VQ-HPS 작업 트리(VQHPS_ROOT) 안의 mesh_vq_vae 를 그대로 쓰고, modelconv 설정은
VQ-HPS configs/config.yaml 의 것(원 실행과 같은 값)을 읽는다.
경로: VQHPS_ROOT, TOKENHMR_3DPW_NPZ, SMPL_DIR 환경변수.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve()
CKPT = Path("checkpoint/MESH_VQVAE/mesh_vqvae_54")
CONNECTION_FOLDER = "body_models/ConnectionMatrices/"
ANN = str(Path(os.environ.get("TOKENHMR_3DPW_NPZ", "data/tokenhmr/datasets/3dpw_test.npz")).resolve())
BODY_MODELS = str(Path(os.environ.get("SMPL_DIR", "data/body_models")).resolve())


def build_model(batch):
    from omegaconf import OmegaConf

    from mesh_vq_vae import FullyConvAE, MeshVQVAE

    cfg = OmegaConf.load("configs/config.yaml")
    mc = cfg.modelconv
    mc.connection_folder = CONNECTION_FOLDER
    mc.initial_connection_fn = CONNECTION_FOLDER + "_pool0.npy"
    # graphAE는 test_mode에서 배치 크기를 고정해 잔차 텐서를 만든다.
    # 실행 배치와 다르면 forward에서 크기가 어긋난다.
    mc.batch = batch
    conv_ae = FullyConvAE(mc, test_mode=True)
    model = MeshVQVAE(conv_ae, num_embeddings=512, embedding_dim=9)
    return model, conv_ae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, conv_ae = build_model(args.batch)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"체크포인트: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(dev).eval()
    conv_ae.init_test_mode()

    ann = np.load(ANN, allow_pickle=True)
    pose = ann["body_pose"].astype(np.float32)      # (N, 72)
    betas = ann["betas"].astype(np.float32)         # (N, 10)
    n = len(pose) if args.limit == 0 else min(args.limit, len(pose))
    # 모델이 배치를 고정하므로 잔여분은 버린다 (35,515 중 최대 batch-1개)
    n = (n // args.batch) * args.batch
    print(f"3DPW test: {n} samples (배치 {args.batch}의 배수로 절삭)", flush=True)

    import smplx
    smplh = smplx.create(BODY_MODELS, model_type="smplh", gender="neutral",
                         use_pca=False, batch_size=args.batch).to(dev)

    v2v_all, tok_match = [], []
    with torch.no_grad():
        for i in range(0, n, args.batch):
            b = args.batch
            bp = torch.from_numpy(pose[i:i + b, 3:66]).to(dev)
            be = torch.from_numpy(betas[i:i + b]).to(dev)
            # VQ-VAE는 정준 메시를 다룬다. 모델 코드가 디코딩 결과에 전역 회전을
            # 따로 곱하므로(rotmat @ mesh_canonical), 전역 방향을 0으로 둔다.
            go = torch.zeros(b, 3, device=dev)
            gt = smplh(betas=be, body_pose=bp, global_orient=go).vertices
            gt_c = gt - gt.mean(dim=1, keepdim=True)

            idx = model.encode(gt_c)                 # (b, 54) 코드북 인덱스
            rec = model.decode(idx)                  # (b, 6890, 3)
            rec_c = rec - rec.mean(dim=1, keepdim=True)

            v2v = torch.norm(rec_c - gt_c, dim=-1).mean(dim=-1) * 1000
            v2v_all += v2v.cpu().tolist()

            # 왕복 후 다시 인코딩했을 때 같은 토큰이 나오는 비율 (양자화 안정성)
            idx2 = model.encode(rec_c)
            tok_match += (idx2 == idx).float().mean(dim=-1).cpu().tolist()

            if (i // args.batch) % 50 == 0:
                print(f"  {i + b}/{n}  PVE {np.mean(v2v_all):.2f} mm", flush=True)

    v2v_all = np.array(v2v_all)
    res = {
        "n_samples": int(len(v2v_all)),
        "oracle_pve_mm": float(v2v_all.mean()),
        "oracle_v2v_median_mm": float(np.median(v2v_all)),
        "oracle_v2v_p95_mm": float(np.percentile(v2v_all, 95)),
        "token_stability": float(np.mean(tok_match)),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== VQ-VAE 재구성 상한 ===", flush=True)
    print(f"  PVE  평균 {res['oracle_pve_mm']:.2f} mm  "
          f"중앙값 {res['oracle_v2v_median_mm']:.2f}  "
          f"p95 {res['oracle_v2v_p95_mm']:.2f}", flush=True)
    print(f"  토큰 안정성 {res['token_stability']:.4f}", flush=True)
    print(f"  @oracle_pve_mm={res['oracle_pve_mm']:.4f} "
          f"@oracle_v2v_median_mm={res['oracle_v2v_median_mm']:.4f} "
          f"@n_samples={res['n_samples']}", flush=True)


if __name__ == "__main__":
    main()
