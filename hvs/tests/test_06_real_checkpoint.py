"""
Test 06: 실제 Facebook 체크포인트 vs hvs 모델 key 비교 검증

 목표:
  1. Facebook 체크포인트의 key 구조 분석
  2. hvs 모델의 key와 매핑 가능한지 검증
  3. 실제 가중치 로드 후 forward pass 정상 동작 확인
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"


@pytest.fixture
def checkpoint():
    """실제 체크포인트 로드"""
    if not os.path.exists(CKPT_PATH):
        pytest.skip("Checkpoint not found")
    raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    return raw["model"]


class TestCheckpointAnalysis:
    """Facebook 체크포인트 분석"""

    def test_checkpoint_keys_structure(self, checkpoint):
        """체크포인트 key 구조 분석"""
        modules = {}
        for key in checkpoint.keys():
            module = key.split(".")[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(key)

        print(f"\n  Facebook Tiny checkpoint: {len(checkpoint)} total keys")
        for mod, keys in sorted(modules.items()):
            print(f"    {mod}: {len(keys)} keys")
        print(f"  [OK] Checkpoint structure analyzed")

    def test_key_remapping(self, checkpoint):
        """hvs key 매핑 후 모듈 분포 확인"""
        from hvs.utils.checkpoint import _remap_checkpoint_keys
        remapped = _remap_checkpoint_keys(checkpoint)

        modules = {}
        for key in remapped.keys():
            module = key.split(".")[0]
            modules[module] = modules.get(module, 0) + 1

        print(f"\n  Remapped keys: {len(remapped)}")
        for mod, count in sorted(modules.items()):
            print(f"    {mod}: {count} keys")

        # 핵심 모듈 존재 확인
        assert "image_encoder" in modules, "image_encoder keys missing"
        assert "prompt_encoder" in modules or "mask_decoder" in modules, \
            "Head keys missing"
        print(f"  [OK] Key remapping verified")


class TestRealCheckpointLoad:
    """실제 체크포인트 로드 + forward 검증"""

    def test_finetune_load_tiny(self, checkpoint):
        """Tiny 체크포인트 finetune 모드 로드"""
        from hvs.models.build import build_sam2_full_model
        from hvs.utils.checkpoint import load_checkpoint

        model = build_sam2_full_model("tiny", image_size=1024)
        result = load_checkpoint(model, CKPT_PATH, mode="finetune", strict=False)

        print(f"\n  Finetune load result:")
        print(f"    Loaded: {result['loaded_keys']} keys")
        print(f"    Missing: {len(result['missing_keys'])} keys")
        print(f"    Unexpected: {len(result['unexpected_keys'])} keys")

        if result['missing_keys']:
            print(f"    Sample missing (first 5):")
            for k in result['missing_keys'][:5]:
                print(f"      - {k}")

        if result['unexpected_keys']:
            print(f"    Sample unexpected (first 5):")
            for k in result['unexpected_keys'][:5]:
                print(f"      - {k}")

        assert result['loaded_keys'] > 0, "No keys loaded"
        print(f"  [OK] Finetune load: {result['loaded_keys']} keys")

    def test_forward_after_load(self, checkpoint):
        """체크포인트 로드 후 forward pass 성공 확인"""
        from hvs.models.build import build_sam2_image_model
        from hvs.utils.checkpoint import load_checkpoint

        model = build_sam2_image_model("tiny", image_size=256)
        load_checkpoint(model, CKPT_PATH, mode="finetune", strict=False)

        ie = model["image_encoder"]
        pe = model["prompt_encoder"]
        md = model["mask_decoder"]
        ie.eval(); pe.eval(); md.eval()

        image = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            enc_out = ie(image)
            fpn = enc_out["backbone_fpn"]
            backbone_features = fpn[-1]
            high_res = [md.conv_s0(fpn[0]), md.conv_s1(fpn[1])]

            point_coords = torch.tensor([[[128.0, 128.0]]])
            point_labels = torch.tensor([[1]], dtype=torch.int32)
            sparse, dense = pe(points=(point_coords, point_labels),
                               boxes=None, masks=None)
            image_pe = pe.get_dense_pe()

            masks, iou, _, _ = md(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res,
            )

        print(f"\n  Forward after checkpoint load:")
        print(f"    Masks: {masks.shape}")
        print(f"    IoU: {iou.shape}")
        print(f"    Masks range: [{masks.min().item():.3f}, {masks.max().item():.3f}]")
        print(f"  [OK] Forward pass after checkpoint load successful")

    def test_backbone_only_load(self, checkpoint):
        """backbone_only 모드: image_encoder만 로드"""
        from hvs.models.build import build_sam2_full_model
        from hvs.utils.checkpoint import load_checkpoint

        model = build_sam2_full_model("tiny", image_size=1024)
        result = load_checkpoint(model, CKPT_PATH, mode="backbone_only", strict=False)

        print(f"\n  Backbone-only load:")
        print(f"    Loaded: {result['loaded_keys']} keys (image_encoder only)")
        assert result['loaded_keys'] > 0
        print(f"  [OK] Backbone-only mode verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
