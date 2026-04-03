"""
손실 함수 (Loss Functions)

■ SAM2 학습에 사용하는 3가지 핵심 손실 함수:

  1. Focal Loss (세그멘테이션 마스크용)
     - Binary Cross Entropy의 개선판
     - "쉬운 픽셀"의 가중치를 줄이고 "어려운 픽셀"에 집중
     - 비유: 시험에서 이미 아는 문제는 무시하고 틀린 문제만 복습

  2. Dice Loss (마스크 영역 겹침 최적화)
     - 예측 마스크와 정답 마스크의 겹침 정도(IoU와 유사)를 직접 최적화
     - 비유: 두 종이를 겹쳤을 때 겹치는 면적을 최대화

  3. IoU Loss (마스크 품질 점수용)
     - 모델이 예측한 IoU 점수와 실제 IoU의 차이를 줄임
     - "이 마스크가 얼마나 정확한가?"를 모델 스스로 판단하게 학습

■ 최종 손실 = focal_weight * Focal + dice_weight * Dice + iou_weight * IoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal Loss (Sigmoid 기반)

    ■ 핵심 아이디어:
      CE Loss에 (1-p_t)^gamma 가중치를 곱함
      → 쉬운 샘플(p_t 높음): 가중치 작음 → 무시
      → 어려운 샘플(p_t 낮음): 가중치 큼 → 집중

    ■ 산업 결함 검출에서 중요한 이유:
      배경 픽셀이 압도적으로 많고, 결함 픽셀이 매우 적음
      → 일반 CE Loss: 배경만 잘 맞추면 loss가 낮아짐
      → Focal Loss: 결함 픽셀(어려운 샘플)에 더 집중

    Args:
        pred: (B, 1, H, W) 마스크 로짓 (sigmoid 전의 원시 출력)
        target: (B, 1, H, W) 정답 마스크 (0 또는 1)
        alpha: 전경/배경 가중치 (0.25 = 배경에 0.75, 전경에 0.25)
        gamma: focusing 파라미터 (클수록 쉬운 샘플 무시)
    """
    prob = torch.sigmoid(pred)
    # Binary Cross Entropy 계산 (로짓 → sigmoid → CE)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    # p_t: 정답 클래스에 대한 확률
    p_t = prob * target + (1 - prob) * (1 - target)

    # Focal 가중치: 쉬운 샘플 억제
    focal_weight = (1 - p_t) ** gamma

    # Alpha 균형: 전경/배경 비율 보정
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dice Loss

    ■ 핵심 아이디어:
      Dice 계수 = 2 * |A ∩ B| / (|A| + |B|)
      → 1이면 완벽한 겹침, 0이면 전혀 겹치지 않음
      Loss = 1 - Dice → 겹침이 클수록 loss 감소

    ■ Focal Loss와의 차이:
      - Focal: 픽셀 단위 분류 (각 픽셀이 맞았는지/틀렸는지)
      - Dice: 영역 단위 겹침 (마스크 전체의 겹침 비율)
      → 두 loss를 함께 쓰면 보완적 효과

    Args:
        pred: (B, 1, H, W) 마스크 로짓
        target: (B, 1, H, W) 정답 마스크 (0 또는 1)
        smooth: 분모가 0이 되는 것을 방지 (라플라스 스무딩)
    """
    pred = torch.sigmoid(pred)

    # 배치 차원 유지, 나머지 평탄화
    pred_flat = pred.flatten(1)      # (B, H*W)
    target_flat = target.flatten(1)  # (B, H*W)

    # Dice 계수 계산
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def iou_loss(
    pred_iou: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    IoU 예측 손실

    ■ 역할:
      모델이 예측한 IoU 점수가 실제 IoU와 일치하도록 학습.
      "이 마스크가 얼마나 정확한지" 스스로 판단하는 능력을 기름.

    Args:
        pred_iou: (B, M) 모델이 예측한 IoU 점수
        pred_mask: (B, M, H, W) 예측 마스크 로짓
        target_mask: (B, 1, H, W) 정답 마스크 (0 또는 1)
    """
    # 예측 마스크를 이진화
    pred_binary = (pred_mask > 0).float()

    # target을 M개 마스크에 맞춰 확장
    if target_mask.shape[1] != pred_binary.shape[1]:
        target_expanded = target_mask.expand_as(pred_binary)
    else:
        target_expanded = target_mask

    # 실제 IoU 계산
    intersection = (pred_binary * target_expanded).flatten(2).sum(dim=2)
    union = (pred_binary + target_expanded).clamp(0, 1).flatten(2).sum(dim=2)
    actual_iou = intersection / (union + 1e-6)

    # MSE Loss: 예측 IoU vs 실제 IoU
    loss = F.mse_loss(pred_iou, actual_iou, reduction=reduction)
    return loss


class SAM2Loss(nn.Module):
    """
    SAM2 통합 손실 함수

    ■ 구성:
      total_loss = focal_weight * Focal + dice_weight * Dice + iou_weight * IoU

    ■ 기본 가중치 (Facebook SAM2 학습 설정):
      - Focal: 20.0 (주요 손실)
      - Dice: 1.0 (보조 손실)
      - IoU: 1.0 (품질 예측 손실)

    Args:
        focal_weight: Focal Loss 가중치
        dice_weight: Dice Loss 가중치
        iou_weight: IoU Loss 가중치
        focal_alpha: Focal Loss의 alpha
        focal_gamma: Focal Loss의 gamma
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_iou: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            pred_masks: (B, M, H, W) 예측 마스크 로짓
            target_masks: (B, 1, H, W) 정답 마스크 (0 또는 1)
            pred_iou: (B, M) 예측 IoU 점수 (None이면 IoU loss 제외)

        Returns:
            dict: {
                "total": 총 손실,
                "focal": Focal Loss 값,
                "dice": Dice Loss 값,
                "iou": IoU Loss 값 (있을 때만),
            }
        """
        # target을 pred와 같은 마스크 수로 확장
        if target_masks.shape[1] != pred_masks.shape[1]:
            target_expanded = target_masks.expand_as(pred_masks)
        else:
            target_expanded = target_masks

        # 각 손실 계산
        focal = sigmoid_focal_loss(
            pred_masks, target_expanded,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
        )
        dice = dice_loss(pred_masks, target_expanded)

        total = self.focal_weight * focal + self.dice_weight * dice

        result = {
            "total": total,
            "focal": focal,
            "dice": dice,
        }

        # IoU Loss (선택적)
        if pred_iou is not None:
            iou = iou_loss(pred_iou, pred_masks, target_masks)
            total = total + self.iou_weight * iou
            result["iou"] = iou
            result["total"] = total

        return result
