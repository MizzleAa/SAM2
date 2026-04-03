"""
Test 04: Memory 모듈 + 전체 SAM2 모델(이미지+비디오) 빌드 검증

 목표:
  1. MemoryEncoder forward shape 검증
  2. MemoryAttention forward shape 검증
  3. build_sam2_full_model 빌드 + 모듈 구성 확인
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from hvs.models.build import (
    build_memory_encoder,
    build_memory_attention,
    build_sam2_full_model,
)


class TestMemoryEncoder:
    """Memory Encoder forward 테스트"""

    def test_memory_encoder_forward(self):
        """MemoryEncoder: pix_feat + mask -> memory features"""
        me = build_memory_encoder(d_model=256, memory_dim=64)
        me.eval()

        B, C, H, W = 1, 256, 16, 16
        pix_feat = torch.randn(B, C, H, W)
        masks = torch.randn(B, 1, H * 16, W * 16)  # 원본 해상도 마스크

        with torch.no_grad():
            out = me(pix_feat, masks)

        assert "vision_features" in out
        assert "vision_pos_enc" in out
        vf = out["vision_features"]
        assert vf.shape == (B, 64, H, W), f"Expected (1, 64, 16, 16), got {vf.shape}"
        assert len(out["vision_pos_enc"]) == 1
        print(f"  [OK] MemoryEncoder: {vf.shape}")


class TestMemoryAttention:
    """Memory Attention forward 테스트"""

    def test_memory_attention_forward(self):
        """MemoryAttention: current + past memory -> context-aware features"""
        ma = build_memory_attention(d_model=256, num_layers=4, memory_dim=64)
        ma.eval()

        B, N, C = 1, 64, 256
        M = 128
        mem_dim = 64
        # (N, B, C) seq-first format
        curr = torch.randn(N, B, C)
        memory = torch.randn(M, B, mem_dim)  # memory는 64차원
        curr_pos = torch.randn(N, B, C)
        memory_pos = torch.randn(M, B, mem_dim)

        with torch.no_grad():
            out = ma(curr, memory, curr_pos, memory_pos)

        assert out.shape == (N, B, C), f"Expected ({N}, {B}, {C}), got {out.shape}"
        print(f"  [OK] MemoryAttention: {out.shape}")


class TestFullSAM2Model:
    """전체 SAM2 모델 빌드 테스트"""

    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_full_model_build(self, size):
        """build_sam2_full_model: 모든 모듈 존재 확인"""
        model = build_sam2_full_model(size, image_size=256)

        # 모든 핵심 모듈 존재 확인
        assert "image_encoder" in model
        assert "prompt_encoder" in model
        assert "mask_decoder" in model
        assert "memory_encoder" in model
        assert "memory_attention" in model
        assert "config" in model

        # config 확인
        cfg = model["config"]
        assert cfg["model_size"] == size
        assert cfg["d_model"] == 256
        assert cfg["memory_dim"] == 64

        # 총 파라미터 수
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for k, m in model.items()
            if isinstance(m, torch.nn.Module)
        )
        print(f"  [{size}] Full SAM2: {total_params:,} params, config={cfg}")

    def test_full_model_sizes(self):
        """모든 크기 모델 빌드 확인"""
        for size in ["tiny", "small", "base_plus", "large"]:
            model = build_sam2_full_model(size, image_size=256)
            total_params = sum(
                sum(p.numel() for p in m.parameters())
                for k, m in model.items()
                if isinstance(m, torch.nn.Module)
            )
            print(f"  {size}: {total_params:,} params")
        print("  [OK] All sizes built successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
