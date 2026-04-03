"""
Test 02: Head 모듈 (Prompt Encoder + Mask Decoder) + 전체 파이프라인 검증

 목표:
  1. PromptEncoder 빌드 + forward shape 검증
  2. MaskDecoder 빌드 + forward shape 검증
  3. 전체 파이프라인 (Image Encoder -> Prompt Encoder -> Mask Decoder) 검증
  4. 다중 마스크 / 단일 마스크 출력 모드 검증

 이 테스트가 통과되면:
  -> Phase 0 Overfitting 테스트로 진행 가능
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from hvs.models.build import (
    build_prompt_encoder,
    build_mask_decoder,
    build_sam2_image_model,
)


class TestPromptEncoder:
    """Prompt Encoder forward test"""

    def test_point_prompt(self):
        """point prompt embedding shape"""
        pe = build_prompt_encoder(d_model=256, image_size=256, backbone_stride=16)
        pe.eval()

        # 전경 점 1개
        coords = torch.rand(1, 1, 2) * 256  # (B, N, 2)
        labels = torch.ones(1, 1, dtype=torch.int32)  # label=1 (전경)

        with torch.no_grad():
            sparse, dense = pe(points=(coords, labels), boxes=None, masks=None)

        # sparse: (B, N+1, C) -- +1은 패딩 점
        assert sparse.shape == (1, 2, 256), f"Expected (1, 2, 256), got {sparse.shape}"
        # dense: (B, C, H, W) -- no_mask_embed 확장
        assert dense.shape[0] == 1 and dense.shape[1] == 256
        print(f"  [OK] Point prompt: sparse={sparse.shape}, dense={dense.shape}")

    def test_no_prompt(self):
        """No prompt (empty sparse)"""
        pe = build_prompt_encoder(d_model=256, image_size=256, backbone_stride=16)
        pe.eval()
        with torch.no_grad():
            sparse, dense = pe(points=None, boxes=None, masks=None)
        assert sparse.shape == (1, 0, 256)
        print(f"  [OK] No prompt: sparse={sparse.shape}")

    def test_dense_pe(self):
        """get_dense_pe shape"""
        pe = build_prompt_encoder(d_model=256, image_size=256, backbone_stride=16)
        dpe = pe.get_dense_pe()
        # (1, C, H, W) = (1, 256, 16, 16)
        assert dpe.shape == (1, 256, 16, 16), f"Expected (1, 256, 16, 16), got {dpe.shape}"
        print(f"  [OK] Dense PE: {dpe.shape}")


class TestMaskDecoder:
    """Mask Decoder forward test"""

    def test_mask_decoder_single_mask(self):
        """Single mask output"""
        md = build_mask_decoder(d_model=256, use_high_res_features=False)
        md.eval()

        B, C, H, W = 1, 256, 16, 16
        image_emb = torch.randn(B, C, H, W)
        image_pe = torch.randn(1, C, H, W)
        sparse = torch.randn(B, 2, C)  # 2 prompt tokens
        dense = torch.randn(B, C, H, W)

        with torch.no_grad():
            masks, iou, tokens, obj_scores = md(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
            )

        assert masks.shape == (1, 1, H * 4, W * 4), f"Mask shape: {masks.shape}"
        assert iou.shape == (1, 1), f"IoU shape: {iou.shape}"
        print(f"  [OK] Single mask: masks={masks.shape}, iou={iou.shape}")

    def test_mask_decoder_multi_mask(self):
        """Multi mask output (3 masks)"""
        md = build_mask_decoder(d_model=256, use_high_res_features=False)
        md.eval()

        B, C, H, W = 1, 256, 16, 16
        image_emb = torch.randn(B, C, H, W)
        image_pe = torch.randn(1, C, H, W)
        sparse = torch.randn(B, 2, C)
        dense = torch.randn(B, C, H, W)

        with torch.no_grad():
            masks, iou, tokens, obj_scores = md(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
            )

        assert masks.shape == (1, 3, H * 4, W * 4), f"Mask shape: {masks.shape}"
        assert iou.shape == (1, 3), f"IoU shape: {iou.shape}"
        print(f"  [OK] Multi mask: masks={masks.shape}, iou={iou.shape}")


class TestFullPipeline:
    """End-to-end: Image Encoder -> Prompt Encoder -> Mask Decoder"""

    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_full_pipeline(self, size):
        """Full pipeline forward pass"""
        image_size = 256  # small for test
        model = build_sam2_image_model(size, image_size=image_size)

        ie = model["image_encoder"]
        pe = model["prompt_encoder"]
        md = model["mask_decoder"]
        ie.eval(); pe.eval(); md.eval()

        # 1) Image Encoding
        image = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            enc_out = ie(image)

        fpn = enc_out["backbone_fpn"]
        pos = enc_out["vision_pos_enc"]
        # backbone_features = 최고 해상도 특징 (맨 뒤)
        backbone_features = fpn[-1]

        # high_res_features (scalp=1 이므로 3개 FPN중 앞 2개)
        high_res_features = [
            md.conv_s0(fpn[0]),   # level 0
            md.conv_s1(fpn[1]),   # level 1
        ]

        # 2) Prompt Encoding (전경 점 1개)
        point_coords = torch.tensor([[[128.0, 128.0]]])  # 이미지 중심
        point_labels = torch.tensor([[1]], dtype=torch.int32)  # 전경
        with torch.no_grad():
            sparse, dense = pe(
                points=(point_coords, point_labels),
                boxes=None, masks=None,
            )

        # 3) Mask Decoding
        image_pe = pe.get_dense_pe()
        with torch.no_grad():
            masks, iou_pred, sam_tokens, obj_scores = md(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

        # shape 검증
        expected_h = backbone_features.shape[2] * 4
        expected_w = backbone_features.shape[3] * 4
        assert masks.shape[0] == 1
        assert masks.shape[1] == 3  # multimask
        assert masks.shape[2] == expected_h
        assert masks.shape[3] == expected_w

        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [ie, pe, md]
        )
        print(f"\n  [{size}] Full pipeline:")
        print(f"    Image: {image.shape}")
        print(f"    FPN levels: {len(fpn)}")
        print(f"    Backbone features: {backbone_features.shape}")
        print(f"    Masks: {masks.shape}")
        print(f"    IoU pred: {iou_pred.shape}")
        print(f"    Object scores: {obj_scores.shape}")
        print(f"    Total params: {total_params:,}")
        print(f"  [OK] {size} full pipeline forward pass")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
