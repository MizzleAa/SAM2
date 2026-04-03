"""
ONNX 모델 내보내기

■ 역할:
  SAM2 모델을 ONNX 형식으로 내보내어 Windows 배포에 사용합니다.
  3개 모듈을 개별 ONNX 파일로 분리합니다.

■ 내보내기 함수 (분리):
  1. export_image_encoder()  — 이미지 인코더 → ONNX
  2. export_prompt_encoder()  — 프롬프트 인코더 → ONNX
  3. export_mask_decoder()    — 마스크 디코더 → ONNX
  4. export_all()             — 전체 내보내기
  5. verify_onnx()            — ONNX 검증

■ 사용법:
  python hvs/scripts/export_model.py --model_size tiny --checkpoint path/to/ckpt
  python hvs/scripts/export_model.py --model_size tiny --output_dir exports/

■ 출력:
  exports/
  ├── sam2_image_encoder.onnx
  ├── sam2_prompt_encoder.onnx
  ├── sam2_mask_decoder.onnx
  └── config.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hvs.models.build import build_sam2_image_model
from hvs.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# 래퍼 클래스 (ONNX 호환)
# ─────────────────────────────────────────────────

class ImageEncoderWrapper(nn.Module):
    """이미지 인코더 ONNX 래퍼"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor):
        enc_out = self.encoder(image)
        fpn = enc_out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


class PromptEncoderWrapper(nn.Module):
    """프롬프트 인코더 ONNX 래퍼"""
    def __init__(self, prompt_encoder):
        super().__init__()
        self.pe = prompt_encoder

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse, dense = self.pe(
            points=(point_coords, point_labels),
            boxes=None, masks=None,
        )
        image_pe = self.pe.get_dense_pe()
        return sparse, dense, image_pe


class MaskDecoderWrapper(nn.Module):
    """마스크 디코더 ONNX 래퍼"""
    def __init__(self, mask_decoder):
        super().__init__()
        self.md = mask_decoder

    def forward(
        self,
        image_embed: torch.Tensor,
        image_pe: torch.Tensor,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
    ):
        masks, iou_pred, _, _ = self.md(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
        )
        return masks, iou_pred


# ─────────────────────────────────────────────────
# 1. 이미지 인코더 내보내기
# ─────────────────────────────────────────────────

def export_image_encoder(
    model_parts: dict,
    output_path: str,
    image_size: int = 1024,
    opset_version: int = 17,
) -> str:
    """
    이미지 인코더 → ONNX

    Args:
        model_parts: build_sam2_image_model() 반환값
        output_path: ONNX 저장 경로
        image_size: 입력 이미지 크기
        opset_version: ONNX opset 버전
    """
    wrapper = ImageEncoderWrapper(model_parts["image_encoder"]).eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        wrapper, dummy, output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["fpn_0", "fpn_1", "fpn_2"],
        dynamic_axes={"image": {0: "batch"}},
    )
    logger.info(f"Image encoder exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────
# 2. 프롬프트 인코더 내보내기
# ─────────────────────────────────────────────────

def export_prompt_encoder(
    model_parts: dict,
    output_path: str,
    opset_version: int = 17,
) -> str:
    """프롬프트 인코더 → ONNX"""
    wrapper = PromptEncoderWrapper(model_parts["prompt_encoder"]).eval()
    dummy_coords = torch.randn(1, 1, 2)
    dummy_labels = torch.ones(1, 1, dtype=torch.int32)

    torch.onnx.export(
        wrapper, (dummy_coords, dummy_labels), output_path,
        opset_version=opset_version,
        input_names=["point_coords", "point_labels"],
        output_names=["sparse_embeddings", "dense_embeddings", "image_pe"],
        dynamic_axes={
            "point_coords": {0: "batch", 1: "num_points"},
            "point_labels": {0: "batch", 1: "num_points"},
        },
    )
    logger.info(f"Prompt encoder exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────
# 3. 마스크 디코더 내보내기
# ─────────────────────────────────────────────────

def export_mask_decoder(
    model_parts: dict,
    output_path: str,
    image_size: int = 1024,
    opset_version: int = 17,
) -> str:
    """마스크 디코더 → ONNX"""
    md = model_parts["mask_decoder"]
    pe = model_parts["prompt_encoder"]
    wrapper = MaskDecoderWrapper(md).eval()

    # 더미 입력 생성
    feat_size = image_size // 16
    high0_ch = md.conv_s0.out_channels if hasattr(md, 'conv_s0') else 256
    high1_ch = md.conv_s1.out_channels if hasattr(md, 'conv_s1') else 256
    config = model_parts.get("config", {})
    d_model = config.get("d_model", 256)

    dummy_embed = torch.randn(1, d_model, feat_size, feat_size)
    dummy_pe = torch.randn(1, d_model, feat_size, feat_size)
    dummy_sparse = torch.randn(1, 2, d_model)
    dummy_dense = torch.randn(1, d_model, feat_size, feat_size)
    dummy_hr0 = torch.randn(1, high0_ch, feat_size * 4, feat_size * 4)
    dummy_hr1 = torch.randn(1, high1_ch, feat_size * 2, feat_size * 2)

    torch.onnx.export(
        wrapper,
        (dummy_embed, dummy_pe, dummy_sparse, dummy_dense, dummy_hr0, dummy_hr1),
        output_path,
        opset_version=opset_version,
        input_names=["image_embed", "image_pe", "sparse", "dense", "high_res_0", "high_res_1"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={"image_embed": {0: "batch"}},
    )
    logger.info(f"Mask decoder exported: {output_path}")
    return output_path


# ─────────────────────────────────────────────────
# 4. 전체 내보내기
# ─────────────────────────────────────────────────

def export_all(
    model_size: str = "tiny",
    image_size: int = 1024,
    checkpoint_path: str = None,
    output_dir: str = "./exports",
    opset_version: int = 17,
) -> dict:
    """
    전체 모델 ONNX 내보내기

    Returns:
        {image_encoder: path, prompt_encoder: path, mask_decoder: path, config: path}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 모델 빌드
    model_parts = build_sam2_image_model(model_size, image_size)

    # 체크포인트 로드
    if checkpoint_path:
        load_checkpoint(model_parts, checkpoint_path, mode="finetune", strict=False)

    paths = {}

    # 1. 이미지 인코더
    paths["image_encoder"] = export_image_encoder(
        model_parts,
        os.path.join(output_dir, "sam2_image_encoder.onnx"),
        image_size, opset_version,
    )

    # 2. 프롬프트 인코더
    paths["prompt_encoder"] = export_prompt_encoder(
        model_parts,
        os.path.join(output_dir, "sam2_prompt_encoder.onnx"),
        opset_version,
    )

    # 3. 마스크 디코더
    paths["mask_decoder"] = export_mask_decoder(
        model_parts,
        os.path.join(output_dir, "sam2_mask_decoder.onnx"),
        image_size, opset_version,
    )

    # 4. 설정 파일
    config = {
        "model_size": model_size,
        "image_size": image_size,
        "checkpoint": checkpoint_path,
        "opset_version": opset_version,
        "files": {
            "image_encoder": "sam2_image_encoder.onnx",
            "prompt_encoder": "sam2_prompt_encoder.onnx",
            "mask_decoder": "sam2_mask_decoder.onnx",
        },
        "input_format": {
            "image": {"shape": [1, 3, image_size, image_size], "dtype": "float32"},
            "point_coords": {"shape": [1, "N", 2], "dtype": "float32"},
            "point_labels": {"shape": [1, "N"], "dtype": "int32"},
        },
        "output_format": {
            "masks": {"shape": [1, 1, "H", "W"], "dtype": "float32"},
            "iou_predictions": {"shape": [1, 1], "dtype": "float32"},
        },
        "normalization": {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
        },
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    paths["config"] = config_path

    print(f"\n=== Export Complete ===")
    for k, v in paths.items():
        size_mb = os.path.getsize(v) / 1024 / 1024 if os.path.exists(v) else 0
        print(f"  {k}: {v} ({size_mb:.1f} MB)")

    return paths


# ─────────────────────────────────────────────────
# 5. ONNX 검증
# ─────────────────────────────────────────────────

def verify_onnx(onnx_path: str) -> bool:
    """ONNX 파일 검증"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  [OK] ONNX valid: {onnx_path}")
        return True
    except ImportError:
        print(f"  [SKIP] onnx not installed")
        return True
    except Exception as e:
        print(f"  [FAIL] {onnx_path}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 ONNX Export")
    parser.add_argument("--model_size", default="tiny")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="./exports")
    args = parser.parse_args()

    export_all(
        model_size=args.model_size,
        image_size=args.image_size,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
