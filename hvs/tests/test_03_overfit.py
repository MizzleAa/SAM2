"""
Test 03: Overfitting 테스트 (Loss -> 0 수렴 검증)

 목표:
  합성 데이터 1장에 대해 모델이 "외울" 수 있는지 확인합니다.
  Loss가 안정적으로 감소하면 -> 학습 파이프라인이 정상적으로 동작.
  Loss가 감소하지 않으면 -> 모델 구조 또는 손실 함수에 문제.

 과적합(Overfitting) 테스트란?
  일부러 모델을 과적합시키는 것이 목적입니다.
  소량 데이터에 대해 Loss가 0에 수렴하면:
  1) 모델이 충분한 표현력을 가지고 있고
  2) 학습 루프(forward -> loss -> backward -> update)가 정상 동작함

 Phase 0의 핵심 검증:
  "이 모델도 학습이 되는가?" -> Yes 확인 후 본격 개발 진행
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest

from hvs.models.build import build_sam2_image_model
from hvs.training.loss_fns import SAM2Loss


def create_synthetic_data(
    batch_size: int = 1,
    image_size: int = 256,
    device: str = "cpu",
):
    """
    합성 학습 데이터 생성

    원 모양의 마스크를 생성하여 모델이 학습할 수 있는지 검증합니다.
    실제 산업 데이터 없이도 학습 파이프라인을 검증할 수 있습니다.

    Returns:
        image: (B, 3, H, W) 합성 이미지
        mask: (B, 1, H, W) 원형 마스크 (0 또는 1)
        point_coords: (B, 1, 2) 마스크 중심 좌표
        point_labels: (B, 1) 전경 레이블
    """
    # 랜덤 이미지 (물체 영역에 밝은 값)
    image = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # 원형 마스크 생성
    mask = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    cx, cy = image_size // 2, image_size // 2
    radius = image_size // 6
    y, x = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing="ij",
    )
    circle = ((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2
    mask[:, 0] = circle.float()

    # 물체 영역에 밝은 색 추가 (학습을 약간 더 쉽게)
    for b in range(batch_size):
        image[b, :, circle] += 2.0

    # 프롬프트: 마스크 중심 점
    point_coords = torch.tensor(
        [[[float(cx), float(cy)]]], device=device
    ).expand(batch_size, -1, -1)
    point_labels = torch.ones(batch_size, 1, dtype=torch.int32, device=device)

    return image, mask, point_coords, point_labels


class TestSAM2Loss:
    """Loss 함수 기본 테스트"""

    def test_focal_loss(self):
        """Focal Loss shape 및 양수 확인"""
        from hvs.training.loss_fns import sigmoid_focal_loss
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randint(0, 2, (1, 1, 16, 16)).float()
        loss = sigmoid_focal_loss(pred, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        print(f"  [OK] Focal Loss: {loss.item():.4f}")

    def test_dice_loss(self):
        """Dice Loss shape 및 양수 확인"""
        from hvs.training.loss_fns import dice_loss
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randint(0, 2, (1, 1, 16, 16)).float()
        loss = dice_loss(pred, target)
        assert loss.ndim == 0
        assert 0 <= loss.item() <= 1.0
        print(f"  [OK] Dice Loss: {loss.item():.4f}")

    def test_sam2_loss_combined(self):
        """SAM2Loss 통합 손실"""
        criterion = SAM2Loss()
        pred_masks = torch.randn(1, 3, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        pred_iou = torch.rand(1, 3)

        losses = criterion(pred_masks, target, pred_iou)
        assert "total" in losses
        assert "focal" in losses
        assert "dice" in losses
        assert "iou" in losses
        assert losses["total"].item() > 0
        print(f"  [OK] SAM2Loss: total={losses['total'].item():.4f}")


class TestOverfitting:
    """Overfitting 테스트 - 핵심 검증"""

    def test_overfit_tiny(self):
        """
        Tiny 모델 overfitting 테스트

        검증 기준:
        1. 학습 시작 시 Loss > 학습 종료 시 Loss (감소 확인)
        2. 최종 Loss < 초기 Loss * 0.5 (50% 이상 감소)
        """
        torch.manual_seed(42)
        image_size = 256

        # 모델 빌드
        model_parts = build_sam2_image_model("tiny", image_size=image_size)
        ie = model_parts["image_encoder"]
        pe = model_parts["prompt_encoder"]
        md = model_parts["mask_decoder"]

        # 학습 모드 설정
        ie.train(); pe.train(); md.train()

        # 옵티마이저: 모든 파라미터를 함께 학습
        all_params = list(ie.parameters()) + list(pe.parameters()) + list(md.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
        criterion = SAM2Loss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)

        # 합성 데이터 생성 (고정, 매번 같은 데이터)
        image, mask, point_coords, point_labels = create_synthetic_data(
            batch_size=1, image_size=image_size
        )

        # 마스크를 Decoder 출력 해상도에 맞게 리사이즈
        # Decoder 출력은 backbone의 최고해상도 특징의 4배
        # 256px -> stride 16,8,4 -> fpn[-1] = 16x16 -> decoder = 64x64
        mask_target_size = image_size // 4
        mask_resized = nn.functional.interpolate(
            mask, size=(mask_target_size, mask_target_size),
            mode="nearest"
        )

        losses_log = []
        num_steps = 50
        print(f"\n  Overfitting test: {num_steps} steps, image_size={image_size}")
        print(f"  Mask target size: {mask_resized.shape}")

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass
            enc_out = ie(image)
            fpn = enc_out["backbone_fpn"]
            backbone_features = fpn[-1]

            # high_res_features
            high_res_features = [
                md.conv_s0(fpn[0]),
                md.conv_s1(fpn[1]),
            ]

            sparse, dense = pe(
                points=(point_coords, point_labels),
                boxes=None, masks=None,
            )
            image_pe = pe.get_dense_pe()

            masks_pred, iou_pred, _, _ = md(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,  # 단일 마스크
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # Loss 계산
            losses = criterion(masks_pred, mask_resized, iou_pred)
            total_loss = losses["total"]

            # Backward + Update
            total_loss.backward()
            optimizer.step()

            losses_log.append(total_loss.item())

            if step % 10 == 0 or step == num_steps - 1:
                print(f"    Step {step:3d}: total={total_loss.item():.4f}, "
                      f"focal={losses['focal'].item():.4f}, "
                      f"dice={losses['dice'].item():.4f}")

        # 검증: Loss 감소 확인
        initial_loss = losses_log[0]
        final_loss = losses_log[-1]
        min_loss = min(losses_log)

        print(f"\n  Results:")
        print(f"    Initial loss: {initial_loss:.4f}")
        print(f"    Final loss:   {final_loss:.4f}")
        print(f"    Min loss:     {min_loss:.4f}")
        print(f"    Reduction:    {(1 - final_loss/initial_loss)*100:.1f}%")

        # 핵심 검증: Loss가 감소해야 함
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

        # 추가 검증: 50% 이상 감소
        assert final_loss < initial_loss * 0.8, \
            f"Loss decreased less than 20%: {initial_loss:.4f} -> {final_loss:.4f}"

        print(f"  [OK] Overfitting test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
