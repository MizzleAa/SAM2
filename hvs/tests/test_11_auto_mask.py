"""
Test 11: AutoMaskGenerator 검증

 목표:
  1. AutoMaskGenerator 초기화
  2. 합성 이미지에서 마스크 자동 생성
  3. 출력 형식 검증
  4. 실제 체크포인트 + COCO 이미지로 자동 마스크
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest
from PIL import Image

from hvs.predictor.auto_mask_generator import AutoMaskGenerator

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"
COCO_IMAGES = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")


class TestAutoMaskBasic:
    """기본 동작 검증"""

    def test_init(self):
        """초기화"""
        gen = AutoMaskGenerator(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
            points_per_side=4,
        )
        assert gen.point_grid.shape == (16, 2)
        print(f"  Grid: {gen.point_grid.shape[0]} points")
        print(f"  [OK] Init")

    def test_generate_synthetic(self):
        """합성 이미지에서 마스크 생성"""
        gen = AutoMaskGenerator(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
            points_per_side=4,
            pred_iou_thresh=0.0,
            stability_score_thresh=0.0,
        )
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[50:200, 50:200] = 180  # 밝은 영역

        masks = gen.generate(image)
        print(f"\n  Generated {len(masks)} masks")
        assert isinstance(masks, list)
        if len(masks) > 0:
            m = masks[0]
            assert "segmentation" in m
            assert "area" in m
            assert "bbox" in m
            assert "predicted_iou" in m
            assert "stability_score" in m
            assert "point_coords" in m
            assert m["segmentation"].shape == (256, 256)
            print(f"  First mask: area={m['area']}, iou={m['predicted_iou']:.3f}")
        print(f"  [OK] Generate (scratch)")

    def test_output_format(self):
        """출력 형식 검증"""
        gen = AutoMaskGenerator(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
            points_per_side=2,
            pred_iou_thresh=0.0,
            stability_score_thresh=0.0,
        )
        image = np.random.randint(0, 255, (128, 192, 3), dtype=np.uint8)
        masks = gen.generate(image)

        for m in masks:
            # segmentation은 원본 크기
            assert m["segmentation"].shape == (128, 192)
            # bbox는 [x, y, w, h]
            assert len(m["bbox"]) == 4
            # area는 정수
            assert isinstance(m["area"], int)
            # iou는 float
            assert isinstance(m["predicted_iou"], float)

        print(f"  {len(masks)} masks, all correct format")
        print(f"  [OK] Output format")


class TestAutoMaskCheckpoint:
    """실제 체크포인트로 검증"""

    def test_generate_coco(self):
        """COCO 이미지에서 자동 마스크 생성"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")
        if not os.path.exists(COCO_IMAGES):
            pytest.skip("COCO images not found")

        gen = AutoMaskGenerator(
            model_size="tiny", image_size=1024,
            device="cpu",
            checkpoint_path=CKPT_PATH,
            init_mode="finetune",
            points_per_side=8,  # 64 points (테스트 속도)
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
        )

        # 첫 번째 COCO 이미지
        img_files = sorted([f for f in os.listdir(COCO_IMAGES) if f.endswith(".jpg")])
        assert len(img_files) > 0
        image = np.array(Image.open(os.path.join(COCO_IMAGES, img_files[0])).convert("RGB"))

        masks = gen.generate(image)
        print(f"\n  COCO image: {image.shape} ({img_files[0]})")
        print(f"  Generated {len(masks)} masks")

        assert len(masks) > 0, "체크포인트 모델이 마스크를 생성하지 못함"

        # IoU 기준 상위 5개 출력
        masks_sorted = sorted(masks, key=lambda m: m["predicted_iou"], reverse=True)
        for i, m in enumerate(masks_sorted[:5]):
            print(f"    Mask {i}: area={m['area']}, iou={m['predicted_iou']:.3f}, "
                  f"stability={m['stability_score']:.3f}")

        print(f"  [OK] Auto mask generation on COCO")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
