"""
Test 10: 실제 COCO 데이터셋 학습 검증

 목표:
  1. COCO 미니 데이터셋 로드 확인
  2. SAM2Dataset으로 변환 검증
  3. 실제 데이터로 1 에폭 학습
  4. 학습 전후 Loss/IoU 비교
  5. Finetune 모드 + 실제 체크포인트 + 실제 데이터 학습
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from hvs.training.sam2_dataset import SAM2Dataset
from hvs.training.trainer import Trainer

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANN_FILE = os.path.join(DATASET_DIR, "annotations", "instances.json")
CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"


class TestCOCODataset:
    """COCO 데이터셋 로드 검증"""

    def test_dataset_load(self):
        """COCO 미니 데이터셋 로드"""
        if not os.path.exists(ANN_FILE):
            pytest.skip("COCO mini dataset not found")

        ds = SAM2Dataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANN_FILE,
            image_size=256,
            num_points=1,
        )

        assert len(ds) > 0
        print(f"\n  Dataset: {len(ds)} samples")
        print(f"  [OK] COCO mini loaded")

    def test_dataset_getitem(self):
        """단일 샘플 로드 + shape 확인"""
        if not os.path.exists(ANN_FILE):
            pytest.skip("COCO mini dataset not found")

        ds = SAM2Dataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANN_FILE,
            image_size=256,
            num_points=1,
        )

        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "point_coords" in sample
        assert "point_labels" in sample

        print(f"\n  image: {sample['image'].shape}")
        print(f"  mask: {sample['mask'].shape}")
        print(f"  point_coords: {sample['point_coords'].shape}")
        print(f"  point_labels: {sample['point_labels'].shape}")
        print(f"  mask sum: {sample['mask'].sum().item():.0f} pixels")
        assert sample["image"].shape == (3, 256, 256)
        assert sample["mask"].shape == (1, 256, 256)
        print(f"  [OK] Sample shapes correct")

    def test_dataloader(self):
        """DataLoader 배치 로드"""
        if not os.path.exists(ANN_FILE):
            pytest.skip("COCO mini dataset not found")

        ds = SAM2Dataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANN_FILE,
            image_size=256,
            num_points=1,
        )
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(dl))

        assert batch["image"].shape[0] == 4
        assert batch["mask"].shape[0] == 4
        print(f"\n  Batch image: {batch['image'].shape}")
        print(f"  Batch mask: {batch['mask'].shape}")
        print(f"  [OK] DataLoader batch")


class TestRealTraining:
    """실제 데이터 학습 검증"""

    def test_scratch_train(self):
        """Scratch 모드: 실제 COCO 데이터로 1 에폭 학습"""
        if not os.path.exists(ANN_FILE):
            pytest.skip("COCO mini dataset not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu",
            init_mode="scratch",
        )

        ds = SAM2Dataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANN_FILE,
            image_size=256,
            num_points=1,
        )
        dl = DataLoader(ds, batch_size=2, shuffle=True)

        # 학습 전 검증
        val_before = trainer.validate(dl)

        # 1 에폭 학습
        losses = trainer.train_epoch(dl, epoch=0)

        # 학습 후 검증
        val_after = trainer.validate(dl)

        print(f"\n  Scratch training on COCO mini:")
        print(f"    Loss: {losses['total']:.4f}")
        print(f"    IoU before: {val_before['iou']:.4f}")
        print(f"    IoU after:  {val_after['iou']:.4f}")
        assert losses["total"] > 0
        print(f"  [OK] Scratch training on real data")

    def test_finetune_train(self):
        """Finetune 모드: 체크포인트 + 실제 데이터 학습"""
        if not os.path.exists(ANN_FILE):
            pytest.skip("COCO mini dataset not found")
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-5, device="cpu",
            init_mode="finetune",
            checkpoint_path=CKPT_PATH,
        )

        ds = SAM2Dataset(
            image_dir=IMAGES_DIR,
            annotation_file=ANN_FILE,
            image_size=256,
            num_points=1,
        )
        dl = DataLoader(ds, batch_size=2, shuffle=True)

        # 학습 전 검증
        val_before = trainer.validate(dl)

        # 1 에폭 학습
        losses = trainer.train_epoch(dl, epoch=0)

        # 학습 후 검증
        val_after = trainer.validate(dl)

        print(f"\n  Finetune training on COCO mini:")
        print(f"    Loss: {losses['total']:.4f}")
        print(f"    IoU before: {val_before['iou']:.4f}")
        print(f"    IoU after:  {val_after['iou']:.4f}")
        assert losses["total"] > 0
        print(f"  [OK] Finetune training on real data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
