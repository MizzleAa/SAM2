"""
Test 07: 학습 루프 통합 테스트

 목표:
  1. Trainer 초기화 (scratch 모드)
  2. 합성 데이터로 학습 1 에폭
  3. 검증 IoU 계산
  4. 체크포인트 저장/복원
  5. Finetune 모드 초기화 (실제 체크포인트)
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from torch.utils.data import DataLoader

from hvs.training.trainer import Trainer
from hvs.training.sam2_dataset import SAM2Dataset

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"


class TestTrainer:
    """Trainer 기본 테스트"""

    def test_trainer_init_scratch(self):
        """Trainer scratch 모드 초기화"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-4, device="cpu", init_mode="scratch",
        )
        params = trainer.count_parameters()
        print(f"\n  Scratch mode:")
        print(f"    image_encoder: {params['image_encoder']:,}")
        print(f"    prompt_encoder: {params['prompt_encoder']:,}")
        print(f"    mask_decoder: {params['mask_decoder']:,}")
        print(f"    total: {params['total']:,}")
        print(f"    trainable: {params['trainable']:,}")
        assert params["total"] > 0
        print(f"  [OK] Trainer init (scratch)")

    def test_train_one_epoch(self):
        """합성 데이터 1 에폭 학습"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        dataset = SAM2Dataset(image_size=256, num_points=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        losses = trainer.train_epoch(dataloader, epoch=0)
        print(f"\n  Train 1 epoch:")
        print(f"    total: {losses['total']:.4f}")
        print(f"    focal: {losses['focal']:.4f}")
        print(f"    dice: {losses['dice']:.4f}")
        assert losses["total"] > 0
        print(f"  [OK] 1 epoch training")

    def test_validate(self):
        """검증 IoU 계산"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        dataset = SAM2Dataset(image_size=256, num_points=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        metrics = trainer.validate(dataloader)
        print(f"\n  Validation IoU: {metrics['iou']:.4f}")
        assert 0 <= metrics["iou"] <= 1
        print(f"  [OK] Validation")

    def test_checkpoint_save_load(self):
        """체크포인트 저장 + 복원"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model_size="tiny", image_size=256,
                lr=1e-3, device="cpu", init_mode="scratch",
                save_dir=tmpdir,
            )

            # 학습 1스텝
            dataset = SAM2Dataset(image_size=256)
            dl = DataLoader(dataset, batch_size=1)
            trainer.train_epoch(dl, epoch=0)

            # 저장
            ckpt_path = trainer.save_checkpoint(epoch=0)
            assert os.path.exists(ckpt_path)

            # 새 trainer로 복원
            trainer2 = Trainer(
                model_size="tiny", image_size=256,
                lr=1e-3, device="cpu", init_mode="scratch",
            )
            epoch = trainer2.load_training_checkpoint(ckpt_path)
            assert epoch == 0
            assert trainer2.global_step == trainer.global_step

            print(f"  [OK] Checkpoint save/load (step={trainer.global_step})")


class TestTrainerFinetune:
    """Finetune 모드 테스트 (실제 체크포인트)"""

    def test_finetune_init(self):
        """Finetune 모드 초기화 (실제 Facebook 체크포인트)"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-5, device="cpu",
            init_mode="finetune",
            checkpoint_path=CKPT_PATH,
        )
        params = trainer.count_parameters()
        print(f"\n  Finetune mode:")
        print(f"    Total params: {params['total']:,}")
        print(f"  [OK] Finetune init with real checkpoint")

    def test_finetune_train(self):
        """Finetune 모드 학습 1스텝"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-5, device="cpu",
            init_mode="finetune",
            checkpoint_path=CKPT_PATH,
        )
        dataset = SAM2Dataset(image_size=256)
        dl = DataLoader(dataset, batch_size=1)

        losses = trainer.train_epoch(dl, epoch=0)
        print(f"\n  Finetune train 1 epoch: total={losses['total']:.4f}")
        assert losses["total"] > 0
        print(f"  [OK] Finetune training")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
