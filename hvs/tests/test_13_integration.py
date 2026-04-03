"""
Test 13: 통합 검증 — 미완료 항목 일괄 검증

■ 검증 항목:
  1. requirements.txt 존재
  2. utils/transforms.py 함수 동작
  3. utils/visualization.py 함수 동작
  4. predictor/utils.py 후처리 함수 동작
  5. scripts/export_model.py ONNX 내보내기
  6. scripts/deploy_windows.py 패키지 생성
  7. 전체 모델 크기별 빌드 검증 (T/S/B+/L)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest
import shutil


class TestRequirements:
    """requirements 파일 검증"""

    def test_requirements_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "torch" in content
        assert "numpy" in content
        assert "hydra" not in content.lower().split("#")[0]  # hydra 제거 확인
        print(f"  [OK] requirements.txt (hydra-free)")


class TestTransforms:
    """전처리 함수 검증"""

    def test_resize_image(self):
        from hvs.utils.transforms import resize_image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized, orig = resize_image(img, 256)
        assert orig == (480, 640)
        assert max(resized.shape[:2]) == 256
        print(f"  resize: {img.shape[:2]} → {resized.shape[:2]}")
        print(f"  [OK] resize_image")

    def test_normalize_denormalize(self):
        from hvs.utils.transforms import normalize_image, denormalize_image
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        norm = normalize_image(img)
        denorm = denormalize_image(norm)
        diff = np.abs(img.astype(float) - denorm.astype(float)).mean()
        assert diff < 1.0, f"Round-trip diff too high: {diff}"
        print(f"  [OK] normalize/denormalize (diff={diff:.2f})")

    def test_preprocess_image(self):
        from hvs.utils.transforms import preprocess_image
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensor, orig = preprocess_image(img, target_size=256)
        assert tensor.shape == (1, 3, 256, 256)
        assert orig == (300, 400)
        print(f"  [OK] preprocess_image {img.shape[:2]} → {tensor.shape}")


class TestPredictorUtils:
    """후처리 함수 검증"""

    def test_mask_to_bbox(self):
        from hvs.predictor.utils import mask_to_bbox
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:50, 30:70] = True
        bbox = mask_to_bbox(mask)
        assert bbox[0] == 30 and bbox[1] == 20  # x1, y1
        assert bbox[2] == 70 and bbox[3] == 50  # x2, y2
        print(f"  [OK] mask_to_bbox: {bbox}")

    def test_calculate_iou(self):
        from hvs.predictor.utils import calculate_iou
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[20:60, 20:60] = True
        m2[40:80, 40:80] = True
        iou = calculate_iou(m1, m2)
        assert 0 < iou < 1
        print(f"  [OK] calculate_iou: {iou:.4f}")

    def test_postprocess_masks(self):
        from hvs.predictor.utils import postprocess_masks
        logits = np.random.randn(3, 64, 64).astype(np.float32)
        masks = postprocess_masks(logits, threshold=0.0, min_area=0)
        assert masks.shape == (3, 64, 64)
        assert masks.dtype == bool
        print(f"  [OK] postprocess_masks: {masks.shape}")

    def test_select_best_mask(self):
        from hvs.predictor.utils import select_best_mask
        masks = np.random.rand(3, 64, 64) > 0.5
        scores = np.array([0.8, 0.95, 0.7])
        best, score = select_best_mask(masks, scores)
        assert score == 0.95
        print(f"  [OK] select_best_mask: score={score}")


class TestExportDeploy:
    """ONNX 내보내기 + 배포 검증"""

    def test_export_onnx(self):
        """ONNX 내보내기 (Tiny, 256)"""
        try:
            import onnxscript
        except ImportError:
            pytest.skip("onnxscript not installed")

        from hvs.scripts.export_model import export_all

        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "_test_exports")
        try:
            paths = export_all(
                model_size="tiny",
                image_size=256,
                output_dir=output_dir,
            )
            assert os.path.exists(paths["image_encoder"])
            assert os.path.exists(paths["prompt_encoder"])
            assert os.path.exists(paths["mask_decoder"])
            assert os.path.exists(paths["config"])

            # 크기 확인
            for name, path in paths.items():
                size = os.path.getsize(path) / 1024 / 1024
                print(f"    {name}: {size:.1f} MB")

            print(f"  [OK] ONNX export")
        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_deploy_package(self):
        """Windows 배포 패키지 생성"""
        try:
            import onnxscript
        except ImportError:
            pytest.skip("onnxscript not installed")

        from hvs.scripts.deploy_windows import create_deploy_package, test_deploy_package

        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "_test_deploy")
        try:
            create_deploy_package(
                model_size="tiny",
                image_size=256,
                output_dir=output_dir,
            )
            ok = test_deploy_package(output_dir)
            assert ok, "Deploy package verification failed"
            print(f"  [OK] Deploy package")
        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)


class TestModelSizes:
    """전 모델 크기 빌드 검증"""

    @pytest.mark.parametrize("size", ["tiny", "small", "base_plus", "large"])
    def test_build_all_sizes(self, size):
        from hvs.models.build import build_sam2_image_model
        model_parts = build_sam2_image_model(size, image_size=256)
        ie = model_parts["image_encoder"]
        total = sum(p.numel() for p in ie.parameters())
        dummy = torch.randn(1, 3, 256, 256)
        out = ie(dummy)
        assert "backbone_fpn" in out
        print(f"  {size}: {total:,} params, fpn={len(out['backbone_fpn'])} levels")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
