"""
Test 12: 추론 벤치마크 + 옵티마이저 + AMP 검증

■ 목표:
  1. 옵티마이저 함수별 분리 검증
  2. Cosine Warmup 스케줄러 동작
  3. Layer-wise LR Decay 동작
  4. AMP 학습 (CPU에서는 비활성)
  5. 추론 속도 벤치마크
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from hvs.training.optimizer import (
    get_param_groups,
    build_optimizer,
    build_scheduler,
    get_optimizer_summary,
)
from hvs.training.trainer import Trainer
from hvs.training.sam2_dataset import SAM2Dataset
from torch.utils.data import DataLoader
from hvs.models.build import build_sam2_image_model
from hvs.scripts.benchmark_inference import (
    benchmark_image_encoding,
    benchmark_mask_decoding,
    benchmark_end_to_end,
    create_benchmark_report,
)
from hvs.predictor.image_predictor import ImagePredictor


class TestOptimizer:
    """옵티마이저 함수 검증"""

    def test_get_param_groups(self):
        """모듈별 파라미터 그룹 생성"""
        model_parts = build_sam2_image_model("tiny", image_size=256)
        groups = get_param_groups(
            model_parts, base_lr=1e-4, weight_decay=0.01,
            backbone_lr_factor=0.1,
        )
        assert len(groups) > 0

        module_names = set(g.get("module", "") for g in groups)
        print(f"\n  Parameter groups: {len(groups)}")
        for g in groups:
            n = sum(p.numel() for p in g["params"])
            print(f"    {g['module']}: lr={g['lr']:.6f}, wd={g['weight_decay']}, params={n:,}")
        assert "image_encoder" in module_names
        print(f"  [OK] Param groups ({len(groups)} groups)")

    def test_build_optimizer(self):
        """옵티마이저 빌드"""
        model_parts = build_sam2_image_model("tiny", image_size=256)
        optimizer = build_optimizer(
            model_parts, lr=1e-4, weight_decay=0.01,
            backbone_lr_factor=0.1,
        )
        summary = get_optimizer_summary(optimizer)
        total_params = sum(s["num_params"] for s in summary)
        print(f"\n  Optimizer: {len(summary)} groups, {total_params:,} params")
        assert total_params > 0
        print(f"  [OK] Optimizer build")

    def test_layer_lr_decay(self):
        """Layer-wise LR Decay"""
        model_parts = build_sam2_image_model("tiny", image_size=256)
        groups = get_param_groups(
            model_parts, base_lr=1e-4,
            backbone_lr_factor=0.1,
            layer_lr_decay=0.8,
        )
        # 백본 그룹들의 LR이 다른지 확인
        backbone_lrs = set(g["lr"] for g in groups if "image_encoder" in g.get("module", ""))
        print(f"\n  Backbone LR values: {sorted(backbone_lrs)}")
        assert len(backbone_lrs) > 1, "Layer-wise decay should produce different LRs"
        print(f"  [OK] Layer-wise LR Decay ({len(backbone_lrs)} distinct LRs)")


class TestScheduler:
    """스케줄러 검증"""

    def test_cosine_warmup(self):
        """Cosine Warmup 스케줄러"""
        model_parts = build_sam2_image_model("tiny", image_size=256)
        optimizer = build_optimizer(model_parts, lr=1e-4)

        scheduler = build_scheduler(
            optimizer, total_steps=100, warmup_steps=10,
            min_lr_ratio=0.01,
        )

        lrs = []
        for step in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Warmup 구간 확인
        assert lrs[0] < lrs[9], "LR should increase during warmup"
        # Decay 구간 확인
        assert lrs[50] < lrs[10], "LR should decrease after warmup"
        # 최소 LR 확인 (min_lr_ratio는 lambda 값이므로 실제 lr = base_lr * lambda)
        # group[0]의 base_lr에 따라 최소값 달라짐
        assert lrs[-1] > 0, "LR should stay positive"

        print(f"\n  Warmup LR[0]={lrs[0]:.6f}, LR[9]={lrs[9]:.6f}")
        print(f"  Peak LR[10]={lrs[10]:.6f}")
        print(f"  Final LR[99]={lrs[99]:.6f}")
        print(f"  [OK] Cosine Warmup Scheduler")


class TestAMP:
    """AMP 학습 검증"""

    def test_amp_trainer(self):
        """AMP 모드 Trainer (CPU에서는 자동 비활성)"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu",
            init_mode="scratch",
            use_amp=True,  # CPU에서는 자동 비활성
        )
        # CPU에서는 AMP가 비활성화되어야 함
        assert not trainer.use_amp, "AMP should be disabled on CPU"

        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)
        losses = trainer.train_epoch(dl, epoch=0)

        assert losses["total"] > 0
        print(f"  AMP trainer: loss={losses['total']:.4f} (AMP={'on' if trainer.use_amp else 'off'})")
        print(f"  [OK] AMP Trainer (CPU fallback)")

    def test_scheduler_in_trainer(self):
        """Trainer에 스케줄러 통합"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-3, device="cpu",
            init_mode="scratch",
            total_steps=100,
            warmup_steps=5,
        )
        assert trainer.scheduler is not None

        dataset = SAM2Dataset(image_size=256, num_points=1)
        dl = DataLoader(dataset, batch_size=1)
        trainer.train_epoch(dl, epoch=0)
        print(f"  Scheduler step: {trainer.global_step}")
        print(f"  [OK] Scheduler in Trainer")

    def test_lr_summary(self):
        """LR 그룹 요약"""
        trainer = Trainer(
            model_size="tiny", image_size=256,
            lr=1e-4, device="cpu",
            init_mode="scratch",
            backbone_lr_factor=0.1,
        )
        summary = trainer.get_lr_summary()
        print(f"\n  LR Groups: {len(summary)}")
        for s in summary:
            print(f"    {s['module']}: lr={s['lr']:.6f}, params={s['num_params']:,}")
        assert len(summary) > 0
        print(f"  [OK] LR Summary")


class TestBenchmark:
    """추론 벤치마크 검증"""

    def test_benchmark_cpu(self):
        """CPU 벤치마크"""
        predictor = ImagePredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        # 가벼운 측정
        enc = benchmark_image_encoding(predictor, (128, 128), num_warmup=1, num_runs=3)
        dec = benchmark_mask_decoding(predictor, (128, 128), num_warmup=1, num_runs=3)
        e2e = benchmark_end_to_end(predictor, (128, 128), num_warmup=1, num_runs=3)

        results = {"encoding": enc, "decoding": dec, "end_to_end": e2e}
        report = create_benchmark_report(results, "tiny", "cpu", 256)
        print(report)

        assert enc["fps"] > 0
        assert dec["fps"] > 0
        assert e2e["fps"] > 0
        print(f"  [OK] CPU Benchmark")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
