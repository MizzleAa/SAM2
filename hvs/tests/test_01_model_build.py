"""
Test 01: 모델 빌드 + Forward Shape 검증

■ 목표:
  1. 4가지 크기(Tiny/Small/Base+/Large) 모델이 정상 빌드되는지 확인
  2. Forward pass 시 출력 shape이 올바른지 검증
  3. 모델 정보 조회 API 동작 확인

■ 이 테스트가 통과되면:
  → Backbone(Hiera) + Neck(FPN) 구현이 정상
  → Phase 0의 overfitting 테스트로 진행 가능
"""

import sys
import os

# hvs 패키지 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from hvs.models.build import (
    build_backbone,
    build_image_encoder,
    get_model_info,
    list_available_models,
    MODEL_CONFIGS,
    _resolve_size,
)


class TestModelBuild:
    """모델 빌드 기본 테스트"""

    def test_list_available_models(self):
        """사용 가능한 모델 목록 확인"""
        models = list_available_models()
        assert "tiny" in models
        assert "small" in models
        assert "base_plus" in models
        assert "large" in models
        print(f"[OK] 사용 가능한 모델: {models}")

    def test_size_aliases(self):
        """별칭 매핑 테스트"""
        assert _resolve_size("t") == "tiny"
        assert _resolve_size("s") == "small"
        assert _resolve_size("b+") == "base_plus"
        assert _resolve_size("l") == "large"
        assert _resolve_size("hiera_t") == "tiny"
        assert _resolve_size("hiera_b+") == "base_plus"
        print("[OK] 모델 크기 별칭 매핑 정상")

    def test_invalid_size_raises(self):
        """잘못된 크기 입력 시 에러 확인"""
        with pytest.raises(ValueError, match="Unknown model size"):
            _resolve_size("xxl")
        print("[OK] 잘못된 크기 입력 시 ValueError 발생")

    def test_get_model_info(self):
        """모델 정보 조회"""
        for size in ["tiny", "small", "base_plus", "large"]:
            info = get_model_info(size)
            assert "size" in info
            assert "stages" in info
            assert "channels" in info
            print(f"  {size}: embed_dim={info['embed_dim']}, "
                  f"stages={info['stages']}, blocks={info['total_blocks']}")
        print("[OK] 모델 정보 조회 정상")


class TestBackboneBuild:
    """Backbone(Hiera) 빌드 + Forward 테스트"""

    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backbone_build_and_forward(self, size):
        """
        Backbone 빌드 후 forward pass shape 검증
        - 입력: (1, 3, 1024, 1024)
        - 출력: 4개 스케일 특징맵 리스트
        """
        backbone = build_backbone(size)
        backbone.eval()

        # 작은 입력으로 테스트 (메모리 절약)
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            outputs = backbone(x)

        assert isinstance(outputs, list)
        assert len(outputs) == 4, f"Expected 4 stages, got {len(outputs)}"

        info = get_model_info(size)
        print(f"\n  [{size}] Backbone output shapes:")
        for i, feat in enumerate(outputs):
            print(f"    Stage {i+1}: {feat.shape}")
            assert feat.dim() == 4, "Output should be 4D (B, C, H, W)"
            assert feat.shape[0] == 1, "Batch size should be 1"

        print(f"  [OK] {size} Backbone forward pass 정상")


class TestImageEncoderBuild:
    """Image Encoder (Backbone + FPN Neck) 빌드 + Forward 테스트"""

    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_image_encoder_build_and_forward(self, size):
        """
        Image Encoder 빌드 후 forward pass shape 검증
        - 입력: (1, 3, 256, 256) — 메모리 절약을 위해 작은 입력
        - 출력 dict: vision_features, vision_pos_enc, backbone_fpn
        """
        encoder = build_image_encoder(size)
        encoder.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = encoder(x)

        # 출력 딕셔너리 키 확인
        assert "vision_features" in output
        assert "vision_pos_enc" in output
        assert "backbone_fpn" in output

        # vision_features shape 확인
        vf = output["vision_features"]
        assert vf.dim() == 4
        d_model = MODEL_CONFIGS[_resolve_size(size)]["neck"]["d_model"]
        assert vf.shape[1] == d_model, \
            f"vision_features channel should be {d_model}, got {vf.shape[1]}"

        # backbone_fpn: scalp 적용 후 특징맵 수 확인
        fpn = output["backbone_fpn"]
        scalp = MODEL_CONFIGS[_resolve_size(size)]["image_encoder"]["scalp"]
        expected_levels = 4 - scalp
        assert len(fpn) == expected_levels, \
            f"Expected {expected_levels} FPN levels (scalp={scalp}), got {len(fpn)}"

        # 모든 FPN 출력이 d_model 채널인지 확인
        for i, feat in enumerate(fpn):
            assert feat.shape[1] == d_model, \
                f"FPN level {i} channel should be {d_model}, got {feat.shape[1]}"

        # position encoding 수 확인
        pos = output["vision_pos_enc"]
        assert len(pos) == len(fpn)

        print(f"\n  [{size}] Image Encoder output:")
        print(f"    vision_features: {vf.shape}")
        for i, (f, p) in enumerate(zip(fpn, pos)):
            print(f"    FPN level {i}: feat={f.shape}, pos={p.shape}")
        print(f"  [OK] {size} Image Encoder forward pass 정상")

    def test_all_sizes_build(self):
        """모든 크기의 Image Encoder 빌드 성공 확인 (forward는 생략)"""
        for size in ["tiny", "small", "base_plus", "large"]:
            encoder = build_image_encoder(size)
            params = sum(p.numel() for p in encoder.parameters())
            print(f"  {size}: {params:,} parameters")
        print("[OK] 모든 크기 모델 빌드 성공")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
