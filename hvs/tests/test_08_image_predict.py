"""
Test 08: Image Predictor 검증

 목표:
  1. ImagePredictor 초기화 (scratch/finetune)
  2. set_image + predict 패턴 검증
  3. 점/박스 프롬프트 정상 동작
  4. 출력 shape 및 범위 검증
  5. 실제 체크포인트로 의미 있는 예측 확인
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from hvs.predictor.image_predictor import ImagePredictor

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"


class TestImagePredictorBasic:
    """기본 동작 검증 (scratch 모드)"""

    def test_init_scratch(self):
        """Scratch 모드 초기화"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        info = predictor.model_info
        print(f"\n  Predictor: {info['total_params']:,} params on {info['device']}")
        assert info["total_params"] > 0
        print(f"  [OK] Scratch init")

    def test_set_image_numpy(self):
        """numpy 이미지로 set_image"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)
        assert predictor._is_image_set
        assert predictor._orig_hw == (480, 640)
        print(f"  [OK] set_image (numpy)")

    def test_predict_point(self):
        """점 프롬프트로 예측"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        assert masks.shape == (3, 480, 640), f"Expected (3, 480, 640), got {masks.shape}"
        assert scores.shape == (3,), f"Expected (3,), got {scores.shape}"
        print(f"\n  Masks: {masks.shape}, Scores: {scores}")
        print(f"  [OK] Point predict (multimask)")

    def test_predict_single_mask(self):
        """단일 마스크 출력"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

        assert masks.shape == (1, 480, 640), f"Expected (1, 480, 640), got {masks.shape}"
        assert scores.shape == (1,)
        print(f"  Masks: {masks.shape}, Scores: {scores}")
        print(f"  [OK] Point predict (single)")

    def test_predict_box(self):
        """박스 프롬프트로 예측"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            box=np.array([[100, 100, 400, 300]]),
            multimask_output=False,
        )

        assert masks.shape == (1, 480, 640)
        print(f"  [OK] Box predict")

    def test_return_logits(self):
        """로짓 반환 모드"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)

        masks, _, _ = predictor.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
            return_logits=True,
        )

        # 로짓이면 float 값 (0/1 이진이 아님)
        assert masks.dtype == np.float32 or masks.dtype == np.float64
        print(f"  Logits range: [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"  [OK] Return logits")

    def test_reset(self):
        """예측기 리셋"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256, device="cpu",
            init_mode="scratch",
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(image)
        assert predictor._is_image_set

        predictor.reset()
        assert not predictor._is_image_set
        print(f"  [OK] Reset")


class TestImagePredictorCheckpoint:
    """실제 체크포인트로 검증"""

    def test_finetune_predict(self):
        """Finetune 모드 + 예측"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        predictor = ImagePredictor(
            model_size="tiny", image_size=1024, device="cpu",
            checkpoint_path=CKPT_PATH, init_mode="finetune",
        )

        # 원형 마스크가 있는 합성 이미지 생성
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[100:400, 100:400] = 200  # 밝은 정사각형 영역

        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[250, 250]]),  # 중앙
            point_labels=np.array([1]),
            multimask_output=True,
        )

        print(f"\n  Finetune prediction:")
        print(f"    Masks: {masks.shape}")
        print(f"    Scores: {scores}")
        print(f"    Best IoU score: {scores.max():.4f}")
        assert masks.shape == (3, 512, 512)
        print(f"  [OK] Finetune predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
