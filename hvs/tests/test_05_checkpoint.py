"""
Test 05: 체크포인트 로더 검증

 목표:
  1. key 리매핑 로직 정상 동작 확인
  2. scratch 모드: 체크포인트 없이 모델 빌드
  3. 가짜 체크포인트로 load/backbone_only 모드 검증
  4. build.py 통합 확인
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from hvs.models.build import build_sam2_full_model, build_sam2_image_model
from hvs.utils.checkpoint import (
    _remap_checkpoint_keys,
    load_checkpoint,
    CHECKPOINT_URLS,
    HF_MODEL_IDS,
)


class TestKeyRemapping:
    """Key 리매핑 테스트"""

    def test_remap_prompt_encoder(self):
        """sam_prompt_encoder -> prompt_encoder"""
        state_dict = {
            "sam_prompt_encoder.pe_layer.weight": torch.randn(1),
            "sam_prompt_encoder.point_embeddings.0.weight": torch.randn(1),
        }
        remapped = _remap_checkpoint_keys(state_dict)
        assert "prompt_encoder.pe_layer.weight" in remapped
        assert "prompt_encoder.point_embeddings.0.weight" in remapped
        print("  [OK] sam_prompt_encoder -> prompt_encoder")

    def test_remap_mask_decoder(self):
        """sam_mask_decoder -> mask_decoder"""
        state_dict = {
            "sam_mask_decoder.iou_token.weight": torch.randn(1),
            "sam_mask_decoder.mask_tokens.weight": torch.randn(1),
        }
        remapped = _remap_checkpoint_keys(state_dict)
        assert "mask_decoder.iou_token.weight" in remapped
        assert "mask_decoder.mask_tokens.weight" in remapped
        print("  [OK] sam_mask_decoder -> mask_decoder")

    def test_remap_keep_others(self):
        """image_encoder, memory_* 등은 유지"""
        state_dict = {
            "image_encoder.trunk.blocks.0.attn.qkv.weight": torch.randn(1),
            "memory_encoder.fuser.layers.0.dwconv.weight": torch.randn(1),
            "memory_attention.layers.0.self_attn.q_proj.weight": torch.randn(1),
        }
        remapped = _remap_checkpoint_keys(state_dict)
        # 변경 없이 유지
        for key in state_dict:
            assert key in remapped
        print("  [OK] image_encoder/memory keys unchanged")


class TestScratchMode:
    """Scratch 모드 테스트"""

    def test_scratch_mode(self):
        """scratch 모드: 체크포인트 로드 생략"""
        model = build_sam2_image_model("tiny", image_size=256)
        result = load_checkpoint(model, "nonexistent.pt", mode="scratch")
        assert result["loaded_keys"] == 0
        assert result["mode"] == "scratch"
        print("  [OK] Scratch mode: no checkpoint loaded")


class TestFakeCheckpoint:
    """가짜 체크포인트 로드 테스트"""

    def _create_fake_checkpoint(self, model_parts, tmpdir):
        """전체 모델의 state_dict를 Facebook 형식으로 저장"""
        state_dict = {}

        # image_encoder
        if "image_encoder" in model_parts:
            for k, v in model_parts["image_encoder"].state_dict().items():
                state_dict[f"image_encoder.{k}"] = v

        # prompt_encoder -> sam_prompt_encoder (Facebook 형식)
        if "prompt_encoder" in model_parts:
            for k, v in model_parts["prompt_encoder"].state_dict().items():
                state_dict[f"sam_prompt_encoder.{k}"] = v

        # mask_decoder -> sam_mask_decoder (Facebook 형식)
        if "mask_decoder" in model_parts:
            for k, v in model_parts["mask_decoder"].state_dict().items():
                state_dict[f"sam_mask_decoder.{k}"] = v

        # memory modules
        for module_name in ["memory_encoder", "memory_attention"]:
            if module_name in model_parts and hasattr(model_parts[module_name], 'state_dict'):
                for k, v in model_parts[module_name].state_dict().items():
                    state_dict[f"{module_name}.{k}"] = v

        ckpt_path = os.path.join(tmpdir, "fake_checkpoint.pt")
        torch.save({"model": state_dict}, ckpt_path)
        return ckpt_path

    def test_finetune_mode(self):
        """finetune 모드: 전체 가중치 로드"""
        model = build_sam2_full_model("tiny", image_size=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._create_fake_checkpoint(model, tmpdir)
            result = load_checkpoint(model, ckpt_path, mode="finetune")

        assert result["loaded_keys"] > 0
        assert result["mode"] == "finetune"
        print(f"  [OK] Finetune mode: {result['loaded_keys']} keys loaded, "
              f"{len(result['missing_keys'])} missing")

    def test_backbone_only_mode(self):
        """backbone_only 모드: image_encoder만 로드"""
        model = build_sam2_full_model("tiny", image_size=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._create_fake_checkpoint(model, tmpdir)
            result = load_checkpoint(model, ckpt_path, mode="backbone_only")

        assert result["loaded_keys"] > 0
        assert result["mode"] == "backbone_only"
        # backbone_only는 image_encoder만 로드하므로 나머지는 missing
        print(f"  [OK] Backbone-only mode: {result['loaded_keys']} keys loaded")


class TestCheckpointURLs:
    """체크포인트 URL/ID 정상 확인"""

    def test_urls_exist(self):
        """모든 크기에 대해 URL이 존재하는지"""
        for size in ["tiny", "small", "base_plus", "large"]:
            assert size in CHECKPOINT_URLS
            assert CHECKPOINT_URLS[size].endswith(".pt")
        print("  [OK] All checkpoint URLs defined")

    def test_hf_ids_exist(self):
        """HuggingFace 모델 ID 존재"""
        for size in ["tiny", "small", "base_plus", "large"]:
            assert size in HF_MODEL_IDS
            assert HF_MODEL_IDS[size].startswith("facebook/")
        print("  [OK] All HF model IDs defined")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
