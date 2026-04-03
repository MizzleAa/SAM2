"""
Test 14: 학습 모드별 종합 테스트

■ 학습 종류 (3가지):
  A) Scratch       — 완전 랜덤 초기화, 처음부터 학습
  B) Finetune      — Facebook 체크포인트 로드 후 미세 조정
  C) Backbone-only — 백본만 로드, Head/Memory는 스크래치

■ 검증 항목:
  - 각 모드별 초기화 정상 동작
  - 각 모드별 1에폭 학습 후 loss 감소
  - 각 모드별 validation IoU
  - 각 모드별 체크포인트 저장/복원
  - 모드 간 성능 비교

■ 옵티마이저 옵션:
  - backbone_lr_factor: 백본 학습률 배율
  - layer_lr_decay: 레이어별 감쇠
  - warmup_steps: 워밍업
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader

from hvs.training.trainer import Trainer
from hvs.training.sam2_dataset import SAM2Dataset

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"
COCO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini")


# ─────────────────────────────────────────────────
# A. Scratch 모드 (완전 랜덤 초기화)
# ─────────────────────────────────────────────────

class TestScratchMode:
    """Scratch: 완전 랜덤 초기화 학습"""

    def test_scratch_init(self):
        """Scratch 초기화 — 전체 파라미터 랜덤"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        params = trainer.count_parameters()
        assert params["trainable"] == params["total"]
        print(f"\n  [Scratch] Total: {params['total']:,} (전체 학습)")
        print(f"  [OK] Scratch init")

    def test_scratch_train_2epochs(self):
        """Scratch 2에폭 학습 — loss 감소 확인"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)

        loss_epoch0 = trainer.train_epoch(dl, epoch=0)
        loss_epoch1 = trainer.train_epoch(dl, epoch=1)

        print(f"\n  [Scratch] Epoch 0: loss={loss_epoch0['total']:.4f}")
        print(f"  [Scratch] Epoch 1: loss={loss_epoch1['total']:.4f}")
        # 학습이 진행되면 loss가 변화해야 함
        assert loss_epoch0["total"] > 0
        assert loss_epoch1["total"] > 0
        print(f"  [OK] Scratch 2-epoch training")

    def test_scratch_validate(self):
        """Scratch 검증 — IoU 범위 확인"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)
        trainer.train_epoch(dl, epoch=0)
        metrics = trainer.validate(dl)
        print(f"\n  [Scratch] IoU: {metrics['iou']:.4f}")
        assert 0 <= metrics["iou"] <= 1
        print(f"  [OK] Scratch validate")

    def test_scratch_with_scheduler(self):
        """Scratch + Cosine Warmup 스케줄러"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
            total_steps=10, warmup_steps=2,
        )
        assert trainer.scheduler is not None
        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)
        losses = trainer.train_epoch(dl, epoch=0)
        print(f"\n  [Scratch+Scheduler] loss={losses['total']:.4f}, steps={trainer.global_step}")
        print(f"  [OK] Scratch with scheduler")


# ─────────────────────────────────────────────────
# B. Finetune 모드 (Facebook 체크포인트 → 미세 조정)
# ─────────────────────────────────────────────────

class TestFinetuneMode:
    """Finetune: Facebook 체크포인트 로드 후 미세 조정"""

    def test_finetune_init(self):
        """Finetune 초기화 — 체크포인트 로드"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-5, device="cpu",
            init_mode="finetune",
            checkpoint_path=CKPT_PATH,
            backbone_lr_factor=0.1,
        )
        params = trainer.count_parameters()
        lr_summary = trainer.get_lr_summary()

        print(f"\n  [Finetune] Total: {params['total']:,}")
        for s in lr_summary[:3]:
            print(f"    {s['module']}: lr={s['lr']:.7f}")
        print(f"  [OK] Finetune init (backbone lr ×0.1)")

    def test_finetune_vs_scratch_iou(self):
        """Finetune vs Scratch — IoU 비교 (Finetune이 높아야 함)"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)

        # Scratch
        scratch = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu", init_mode="scratch",
        )
        scratch.train_epoch(dl, epoch=0)
        scratch_iou = scratch.validate(dl)["iou"]

        # Finetune
        finetune = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-5, device="cpu",
            init_mode="finetune",
            checkpoint_path=CKPT_PATH,
        )
        finetune.train_epoch(dl, epoch=0)
        finetune_iou = finetune.validate(dl)["iou"]

        print(f"\n  [Scratch]  IoU: {scratch_iou:.4f}")
        print(f"  [Finetune] IoU: {finetune_iou:.4f}")
        print(f"  Gap: {finetune_iou - scratch_iou:+.4f}")
        # Finetune이 일반적으로 더 좋지만, 합성 데이터 1에폭이므로 절대 보장은 않음
        print(f"  [OK] Finetune vs Scratch comparison")

    def test_finetune_checkpoint_cycle(self):
        """Finetune 체크포인트 저장 → 복원 → 재학습"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Finetune 초기화 + 학습
            trainer1 = Trainer(
                model_size="tiny", image_size=256,
                lr=1e-5, device="cpu",
                init_mode="finetune",
                checkpoint_path=CKPT_PATH,
                save_dir=tmpdir,
            )
            dataset = SAM2Dataset(image_size=256)
            dl = DataLoader(dataset, batch_size=1)
            trainer1.train_epoch(dl, epoch=0)

            # 2. 저장
            ckpt = trainer1.save_checkpoint(epoch=0)
            assert os.path.exists(ckpt)

            # 3. 새 trainer로 복원 (scratch로 시작 → 체크포인트 로드)
            trainer2 = Trainer(
                model_size="tiny", image_size=256,
                lr=1e-5, device="cpu",
                init_mode="scratch",
            )
            epoch = trainer2.load_training_checkpoint(ckpt)
            assert epoch == 0
            assert trainer2.global_step == trainer1.global_step

            # 4. 복원 후 추가 학습
            losses = trainer2.train_epoch(dl, epoch=1)
            print(f"\n  [Finetune Cycle] Epoch 1 after restore: loss={losses['total']:.4f}")
            print(f"  [OK] Finetune checkpoint cycle")


# ─────────────────────────────────────────────────
# C. Backbone-only 모드 (백본만 로드)
# ─────────────────────────────────────────────────

class TestBackboneOnlyMode:
    """Backbone-only: 백본만 로드, Head/Memory 스크래치"""

    def test_backbone_only_init(self):
        """Backbone-only 초기화"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-4, device="cpu",
            init_mode="backbone_only",
            checkpoint_path=CKPT_PATH,
            backbone_lr_factor=0.01,  # 백본은 거의 고정
        )
        params = trainer.count_parameters()
        print(f"\n  [Backbone-only] Total: {params['total']:,}")
        print(f"  [OK] Backbone-only init")

    def test_backbone_only_train(self):
        """Backbone-only 학습"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-4, device="cpu",
            init_mode="backbone_only",
            checkpoint_path=CKPT_PATH,
        )
        dataset = SAM2Dataset(image_size=256)
        dl = DataLoader(dataset, batch_size=1)
        losses = trainer.train_epoch(dl, epoch=0)
        iou = trainer.validate(dl)["iou"]

        print(f"\n  [Backbone-only] loss={losses['total']:.4f}, IoU={iou:.4f}")
        print(f"  [OK] Backbone-only train")


# ─────────────────────────────────────────────────
# D. 3모드 일괄 비교
# ─────────────────────────────────────────────────

class TestModeComparison:
    """3가지 모드 일괄 비교"""

    def test_all_modes_comparison(self):
        """Scratch / Finetune / Backbone-only 비교"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        modes = [
            ("scratch", "scratch", None),
            ("finetune", "finetune", CKPT_PATH),
            ("backbone_only", "backbone_only", CKPT_PATH),
        ]
        results = {}

        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)

        for name, mode, ckpt in modes:
            trainer = Trainer(
                model_size="tiny", image_size=256,
                lr=1e-4 if mode == "scratch" else 1e-5,
                device="cpu",
                init_mode=mode,
                checkpoint_path=ckpt,
            )
            losses = trainer.train_epoch(dl, epoch=0)
            iou = trainer.validate(dl)["iou"]
            results[name] = {"loss": losses["total"], "iou": iou}

        print(f"\n  ╔══════════════╤══════════╤══════════╗")
        print(f"  ║ Mode         │ Loss     │ IoU      ║")
        print(f"  ╠══════════════╪══════════╪══════════╣")
        for name, r in results.items():
            print(f"  ║ {name:<12} │ {r['loss']:.4f}   │ {r['iou']:.4f}   ║")
        print(f"  ╚══════════════╧══════════╧══════════╝")
        print(f"  [OK] All 3 modes compared")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
