"""레포의 모델 파일 두 개에 rotfix 변경을 넣는다 (Worker_edit 이 절대경로를 못 다뤄 대체).

각 치환은 정확히 한 번만 일치해야 한다 — 아니면 아무것도 쓰지 않고 실패한다.
이미 적용돼 있으면(새 문자열이 있으면) 건너뛴다.

두 변경 모두 기본값은 원본 동작이다(head_axis="batch", rot_from="pooled").
  - head_axis="token": 토큰 헤드의 causal self-attention 이 배치 축에 걸리던 결함을 고친다.
  - rot_from="attn"  : 전역 평균풀링 벡터 대신 공간 특징에 cross-attention 해서 회전을 낸다.
"""

import os
import sys
from pathlib import Path

# VQHPS_ROOT = 패치(patches/01, 02)를 적용한 VQ-HPS 작업 트리
REPO = Path(os.environ.get("VQHPS_ROOT", "third_party/VQ-HPS")).resolve() / "vq_hps/model/mesh_regressor_dino"

TRANSFORMER = [
    (
        "    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):\n"
        "        super().__init__()\n"
        "        assert embed_dim % 8 == 0",
        "    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, causal=True):\n"
        "        super().__init__()\n"
        "        self.causal = causal  # False 면 마스크 없이 전체 self-attention\n"
        "        assert embed_dim % 8 == 0",
    ),
    (
        '        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))',
        "        if self.causal:\n"
        '            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))',
    ),
    (
        "        drop_out_rate=0.1,\n        feedforward_dim=2048,\n    ):\n        super().__init__()\n"
        "        self.ln1 = nn.LayerNorm(embed_dim)\n        self.ln2 = nn.LayerNorm(embed_dim)\n"
        "        self.attn = CausalCrossConditionalSelfAttention(\n"
        "            embed_dim, block_size, n_head, drop_out_rate\n        )",
        "        drop_out_rate=0.1,\n        feedforward_dim=2048,\n        causal=True,\n    ):\n        super().__init__()\n"
        "        self.ln1 = nn.LayerNorm(embed_dim)\n        self.ln2 = nn.LayerNorm(embed_dim)\n"
        "        self.attn = CausalCrossConditionalSelfAttention(\n"
        "            embed_dim, block_size, n_head, drop_out_rate, causal\n        )",
    ),
    (
        "        num_layers=2,\n        n_head=8,\n        drop_out_rate=0.1,\n        feedforward_dim=2048,\n    ):",
        "        num_layers=2,\n        n_head=8,\n        drop_out_rate=0.1,\n        feedforward_dim=2048,\n"
        "        causal=True,\n    ):",
    ),
    (
        "                Block(embed_dim, block_size, n_head, drop_out_rate, feedforward_dim)\n"
        "                for _ in range(num_layers)",
        "                Block(embed_dim, block_size, n_head, drop_out_rate, feedforward_dim, causal)\n"
        "                for _ in range(num_layers)",
    ),
    (
        "        block_size=103,\n        autoreg=True,\n    ):",
        '        block_size=103,\n        autoreg=True,\n        head_axis="batch",\n    ):',
    ),
    (
        "        self.autoreg = autoreg\n        if autoreg:\n            self.dec_head = CrossCondTransHead(\n"
        "                dict_size,\n                model_dim,\n                block_size,\n"
        "                num_dec_layers,\n                nhead,\n                dropout,\n"
        "                feedforward_dim,\n            )",
        "        self.autoreg = autoreg\n"
        '        # "batch": 원본 — 헤드가 (num_mesh, B, C)를 그대로 받아 causal attention이 배치 축에 걸린다\n'
        "        #          (샘플 i가 같은 배치의 샘플 0..i-1을 본다. 실측: 자리만 바꿔도 토큰 1~3개 변함).\n"
        '        # "token": (B, num_mesh, C)로 넘겨 54개 토큰 특징 위에서 마스크 없이 self-attention 한다.\n'
        "        self.head_axis = head_axis\n"
        "        if autoreg:\n            self.dec_head = CrossCondTransHead(\n"
        "                dict_size,\n                model_dim,\n                block_size,\n"
        "                num_dec_layers,\n                nhead,\n                dropout,\n"
        '                feedforward_dim,\n                causal=(head_axis == "batch"),\n            )',
    ),
    (
        "        mesh_logits = self.dec_head(mesh_features)\n\n        return enc_img_features, mesh_logits",
        '        if self.head_axis == "token":\n'
        "            mesh_logits = self.dec_head(mesh_features.transpose(0, 1)).transpose(0, 1)\n"
        "        else:\n            mesh_logits = self.dec_head(mesh_features)\n\n"
        "        return enc_img_features, mesh_logits",
    ),
    (
        "def build_transformer(transformer_config, autoreg=True):",
        'def build_transformer(transformer_config, autoreg=True, head_axis="batch"):',
    ),
    (
        '        block_size=transformer_config["block_size"],\n        autoreg=autoreg,\n    )',
        '        block_size=transformer_config["block_size"],\n        autoreg=autoreg,\n'
        "        head_axis=head_axis,\n    )",
    ),
]

ROTCAM_CLASS = '''class RotCamHead(nn.Module):
    """전역 회전(6D)·약원근 카메라 헤드.

    원본은 DINOv3 특징을 전역 평균풀링한 벡터에서 회전을 회귀했는데, 풀링이 공간 배치를
    지워 회전이 학습되지 않았다(학습 데이터에서도 평균 28°). 여기서는 학습 쿼리 하나가
    인코더의 공간 특징(위치 인코딩 포함)에 cross-attention 해서 회전·카메라를 낸다.
    """

    def __init__(self, dim: int, nhead: int = 8, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.rot = nn.Linear(dim, 6)
        self.cam = nn.Linear(dim, 3)
        # 6D 표현이 0 근처면 정규화가 불안정하다 — 항등 회전에서 시작한다
        nn.init.zeros_(self.rot.weight)
        with torch.no_grad():
            self.rot.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, memory, pos):
        """memory, pos: (hw, b, dim) 시퀀스-우선. 반환 rot (b, 6), cam (b, 3)."""
        q = self.query.expand(-1, memory.shape[1], -1)
        x, _ = self.attn(q, memory + pos, memory)
        x = self.mlp(self.norm(x[0]))
        return self.rot(x), self.cam(x)


class MeshRegressor(nn.Module):
    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""'''

PREDICTOR = [
    (
        "class MeshRegressor(nn.Module):\n"
        '    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""',
        ROTCAM_CLASS,
    ),
    (
        "        seq_len=49,\n        autoreg=True,\n    ):",
        '        seq_len=49,\n        autoreg=True,\n        head_axis="batch",\n        rot_from="pooled",\n    ):',
    ),
    (
        "        self.transformer = build_transformer(\n            self.transformer_config, autoreg=self.autoreg\n        )",
        "        self.transformer = build_transformer(\n"
        "            self.transformer_config, autoreg=self.autoreg, head_axis=head_axis\n        )",
    ),
    (
        "        self.rot_predictor = nn.Linear(1024, 6)\n        self.cam_predictor = nn.Linear(1024, 3)",
        "        self.rot_predictor = nn.Linear(1024, 6)\n        self.cam_predictor = nn.Linear(1024, 3)\n"
        '        # "pooled": 원본 — 전역 평균풀링 벡터 + T-pose 상수에서 회전·카메라를 회귀한다.\n'
        '        # "attn":   인코더의 공간 특징에 학습 쿼리로 cross-attention 한다 (공간 배치 유지).\n'
        "        self.rot_from = rot_from\n"
        '        if rot_from == "attn":\n'
        "            self.rotcam_head = RotCamHead(\n"
        '                self.transformer_config["model_dim"],\n'
        '                self.transformer_config["nhead"],\n'
        '                self.transformer_config["dropout"],\n'
        "            )",
    ),
    (
        "        img_enc = img_enc.view(img_enc.size(0), -1)\n\n"
        "        xc = torch.cat([img_enc, pose_init], 1)\n\n"
        "        xc = self.fc1(xc)\n        xc = self.drop1(xc)\n        xc = self.fc2(xc)\n        xc = self.drop2(xc)\n\n"
        "        pred_rot = self.rot_predictor(xc) # b, 6\n"
        "        pred_cam = self.cam_predictor(xc).view(-1, 3).cpu()  # b, 3",
        "        img_enc = img_enc.view(img_enc.size(0), -1)\n\n"
        '        if self.rot_from == "pooled":\n'
        "            xc = torch.cat([img_enc, pose_init], 1)\n\n"
        "            xc = self.fc1(xc)\n            xc = self.drop1(xc)\n            xc = self.fc2(xc)\n            xc = self.drop2(xc)\n\n"
        "            pred_rot = self.rot_predictor(xc) # b, 6\n"
        "            pred_cam = self.cam_predictor(xc).view(-1, 3).cpu()  # b, 3",
    ),
    (
        "        _, mesh_features = self.transformer(img_features_pose, mesh_tokens, pos_enc)   # transformer_dim, b, hxw",
        "        enc_img_features, mesh_features = self.transformer(img_features_pose, mesh_tokens, pos_enc)   # transformer_dim, b, hxw\n"
        '        if self.rot_from == "attn":\n'
        "            pred_rot, pred_cam = self.rotcam_head(enc_img_features, pos_enc)\n"
        "            pred_cam = pred_cam.cpu()",
    ),
]


def apply(path: Path, edits: list[tuple[str, str]]) -> None:
    src = path.read_text()
    done = 0
    for i, (old, new) in enumerate(edits):
        # 새 문자열이 옛 문자열을 포함하는 치환(예: 7번)이 있어 'old not in src' 는 재실행 때 오탐한다
        if new in src:
            print(f"  [{i}] 이미 적용됨", flush=True)
            continue
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{path.name} 치환 {i}: 일치 {n}회 (1회여야 함)\n---\n{old[:200]}")
        src = src.replace(old, new)
        done += 1
    path.write_text(src)
    print(f"{path.name}: {done}/{len(edits)} 적용", flush=True)


def main() -> int:
    apply(REPO / "transformer.py", TRANSFORMER)
    apply(REPO / "mesh_predictor_transformer.py", PREDICTOR)
    import py_compile
    for f in ("transformer.py", "mesh_predictor_transformer.py"):
        py_compile.compile(str(REPO / f), doraise=True)
    print("@edits_ok=1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
