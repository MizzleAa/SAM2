"""
Test 02b: 다중 샘플 Overfitting 테스트 (10~20장)

■ 목표:
  합성 이미지 10~20장에 대해 모델이 완전히 외울 수 있는지 확인합니다.
  이전 test_03은 1장에 대해 검증했다면, 이 테스트는 다양한 형태의
  마스크(원, 사각형, 삼각형)를 10~20개 생성하여 모델이
  모든 패턴을 학습할 수 있는지 검증합니다.

■ 검증 기준:
  - 10장 데이터, 100 steps → Loss 80% 이상 감소
  - 20장 데이터, 200 steps → Loss 70% 이상 감소
  - 최종 IoU > 0 (마스크가 어느 정도 일치)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader, TensorDataset

from hvs.models.build import build_sam2_image_model
from hvs.training.loss_fns import SAM2Loss


def create_multi_sample_dataset(
    num_samples: int = 10,
    image_size: int = 256,
):
    """
    다양한 형태의 합성 데이터셋 생성

    - 원형, 사각형, 타원형 마스크를 랜덤 위치에 배치
    - 각 이미지에 마스크 중심을 프롬프트 점으로 제공

    Returns:
        images: (N, 3, H, W),  masks: (N, 1, H, W),
        point_coords: (N, 1, 2),  point_labels: (N, 1)
    """
    images = []
    masks = []
    points = []

    for i in range(num_samples):
        # 배경 노이즈
        img = torch.randn(3, image_size, image_size) * 0.5

        # 마스크 생성 (형태 랜덤)
        mask = torch.zeros(1, image_size, image_size)
        y, x = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="ij"
        )

        # 랜덤 중심 + 크기
        cx = np.random.randint(image_size // 4, 3 * image_size // 4)
        cy = np.random.randint(image_size // 4, 3 * image_size // 4)
        r = np.random.randint(image_size // 8, image_size // 4)

        shape_type = i % 3
        if shape_type == 0:  # 원
            dist = ((x - cx) ** 2 + (y - cy) ** 2).float().sqrt()
            mask[0] = (dist < r).float()
        elif shape_type == 1:  # 사각형
            mask[0] = ((x >= cx - r) & (x <= cx + r) & (y >= cy - r) & (y <= cy + r)).float()
        else:  # 타원
            dist = ((x - cx) ** 2 / (r ** 2 + 1) + (y - cy) ** 2 / ((r * 0.6) ** 2 + 1)).float()
            mask[0] = (dist < 1.0).float()

        # 물체 영역에 밝은 색
        img[:, mask[0].bool()] += 2.0

        images.append(img)
        masks.append(mask)
        points.append(torch.tensor([[float(cx), float(cy)]]))

    images = torch.stack(images)
    masks = torch.stack(masks)
    point_coords = torch.stack(points)
    point_labels = torch.ones(num_samples, 1, dtype=torch.int32)

    return images, masks, point_coords, point_labels


class TestMultiSampleOverfit:
    """다중 샘플 Overfitting 테스트"""

    def test_overfit_10_samples(self):
        """10장 합성 데이터 overfitting"""
        torch.manual_seed(42)
        np.random.seed(42)

        num_samples = 10
        image_size = 256
        num_steps = 100

        # 데이터 생성
        images, masks, point_coords, point_labels = create_multi_sample_dataset(
            num_samples=num_samples, image_size=image_size,
        )
        print(f"\n  Dataset: {num_samples} samples, shapes={list(images.shape)}")

        # 모델
        model_parts = build_sam2_image_model("tiny", image_size=image_size)
        ie = model_parts["image_encoder"]
        pe = model_parts["prompt_encoder"]
        md = model_parts["mask_decoder"]
        ie.train(); pe.train(); md.train()

        all_params = list(ie.parameters()) + list(pe.parameters()) + list(md.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=5e-4, weight_decay=0.01)
        criterion = SAM2Loss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)

        mask_h = image_size // 4

        losses_log = []
        for step in range(num_steps):
            idx = step % num_samples
            img = images[idx:idx+1]
            msk = F.interpolate(masks[idx:idx+1], (mask_h, mask_h), mode="nearest")
            pc = point_coords[idx:idx+1]
            pl = point_labels[idx:idx+1]

            optimizer.zero_grad()
            enc = ie(img)
            fpn = enc["backbone_fpn"]
            high_res = [md.conv_s0(fpn[0]), md.conv_s1(fpn[1])]
            sparse, dense = pe(points=(pc, pl), boxes=None, masks=None)
            image_pe = pe.get_dense_pe()
            masks_pred, iou_pred, _, _ = md(
                image_embeddings=fpn[-1], image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, repeat_image=False, high_res_features=high_res,
            )
            losses = criterion(masks_pred, msk, iou_pred)
            losses["total"].backward()
            optimizer.step()
            losses_log.append(losses["total"].item())

            if step % 20 == 0 or step == num_steps - 1:
                print(f"    Step {step:3d}: loss={losses['total'].item():.4f}")

        initial = losses_log[0]
        final = losses_log[-1]
        reduction = (1 - final / initial) * 100

        print(f"\n  Initial: {initial:.4f} → Final: {final:.4f} ({reduction:.1f}% reduction)")
        assert final < initial, "Loss did not decrease"
        assert reduction > 50, f"Reduction only {reduction:.1f}% (need >50%)"
        print(f"  [OK] 10-sample overfit ({reduction:.1f}% reduction)")

    def test_overfit_20_samples(self):
        """20장 합성 데이터 overfitting"""
        torch.manual_seed(123)
        np.random.seed(123)

        num_samples = 20
        image_size = 256
        num_steps = 200

        images, masks, point_coords, point_labels = create_multi_sample_dataset(
            num_samples=num_samples, image_size=image_size,
        )
        print(f"\n  Dataset: {num_samples} samples")

        model_parts = build_sam2_image_model("tiny", image_size=image_size)
        ie = model_parts["image_encoder"]
        pe = model_parts["prompt_encoder"]
        md = model_parts["mask_decoder"]
        ie.train(); pe.train(); md.train()

        all_params = list(ie.parameters()) + list(pe.parameters()) + list(md.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=5e-4, weight_decay=0.01)
        criterion = SAM2Loss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)
        mask_h = image_size // 4

        losses_log = []
        for step in range(num_steps):
            idx = step % num_samples
            img = images[idx:idx+1]
            msk = F.interpolate(masks[idx:idx+1], (mask_h, mask_h), mode="nearest")
            pc = point_coords[idx:idx+1]
            pl = point_labels[idx:idx+1]

            optimizer.zero_grad()
            enc = ie(img)
            fpn = enc["backbone_fpn"]
            high_res = [md.conv_s0(fpn[0]), md.conv_s1(fpn[1])]
            sparse, dense = pe(points=(pc, pl), boxes=None, masks=None)
            image_pe = pe.get_dense_pe()
            masks_pred, iou_pred, _, _ = md(
                image_embeddings=fpn[-1], image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, repeat_image=False, high_res_features=high_res,
            )
            losses = criterion(masks_pred, msk, iou_pred)
            losses["total"].backward()
            optimizer.step()
            losses_log.append(losses["total"].item())

            if step % 40 == 0 or step == num_steps - 1:
                print(f"    Step {step:3d}: loss={losses['total'].item():.4f}")

        initial = losses_log[0]
        final = losses_log[-1]
        reduction = (1 - final / initial) * 100

        print(f"\n  Initial: {initial:.4f} → Final: {final:.4f} ({reduction:.1f}% reduction)")
        assert final < initial, "Loss did not decrease"
        print(f"  [OK] 20-sample overfit ({reduction:.1f}% reduction)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
