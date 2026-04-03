"""
SAM2 옵티마이저 + 스케줄러

■ 구성 (함수별 분리):
  1. build_optimizer()     — AdamW + Layer-wise LR Decay
  2. build_scheduler()     — Cosine Annealing + Linear Warmup
  3. get_param_groups()    — 모듈별 파라미터 그룹 생성
  4. get_layer_lr_decay()  — 레이어별 학습률 감쇠 계수 계산

■ Layer-wise LR Decay:
  SAM2의 Hiera 백본은 깊은 레이어일수록 높은 학습률 적용.
  얕은 레이어(일반적 시각 특징)는 적게 변경하고
  깊은 레이어(작업-특화 특징)는 많이 조정합니다.

  예: decay=0.8, 4레이어 → [0.512, 0.64, 0.8, 1.0] × base_lr
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ─────────────────────────────────────────────────
# 1. 파라미터 그룹 구성
# ─────────────────────────────────────────────────

def get_param_groups(
    model_parts: Dict[str, nn.Module],
    base_lr: float = 1e-4,
    weight_decay: float = 0.01,
    backbone_lr_factor: float = 0.1,
    layer_lr_decay: float = 1.0,
    no_decay_keywords: Tuple[str, ...] = ("bias", "norm", "pos_embed", "pe_layer"),
) -> List[dict]:
    """
    모듈별 파라미터 그룹 생성

    ■ 구분 전략:
      - image_encoder: backbone_lr_factor × base_lr (사전학습 보존)
      - prompt_encoder: base_lr
      - mask_decoder: base_lr
      - memory_encoder: base_lr
      - memory_attention: base_lr

    ■ Weight Decay 제외:
      bias, normalization, position embedding은 decay 미적용

    Args:
        model_parts: {"image_encoder": ..., "prompt_encoder": ..., ...}
        base_lr: 기본 학습률
        weight_decay: 가중치 감쇠
        backbone_lr_factor: 백본 학습률 배율
        layer_lr_decay: 레이어별 학습률 감쇠율
        no_decay_keywords: decay 제외 파라미터 키워드

    Returns:
        list of param_group dicts
    """
    param_groups = []

    for module_name, module in model_parts.items():
        if module is None:
            continue
        if not isinstance(module, nn.Module):
            continue

        # 모듈별 기본 학습률
        if module_name == "image_encoder":
            module_lr = base_lr * backbone_lr_factor
        else:
            module_lr = base_lr

        # Layer-wise decay (백본에만 적용)
        if module_name == "image_encoder" and layer_lr_decay < 1.0:
            layer_groups = _get_backbone_layer_groups(
                module, module_lr, weight_decay,
                layer_lr_decay, no_decay_keywords,
                prefix=module_name,
            )
            param_groups.extend(layer_groups)
        else:
            # 일반 모듈: decay/no-decay 2그룹
            decay_params = []
            no_decay_params = []

            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(kw in name for kw in no_decay_keywords):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": module_lr,
                    "weight_decay": weight_decay,
                    "module": module_name,
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": module_lr,
                    "weight_decay": 0.0,
                    "module": module_name,
                })

    return param_groups


def _get_backbone_layer_groups(
    backbone: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay_rate: float,
    no_decay_keywords: Tuple[str, ...],
    prefix: str = "backbone",
) -> List[dict]:
    """
    백본 레이어별 LR Decay 그룹 생성

    ■ Hiera 구조:
      - patch_embed (Stage 0): 가장 낮은 LR
      - blocks (Stage 1~N): 점진적 증가
      - 기타: 최종 레이어 LR
    """
    # 레이어 깊이 맵 생성
    layer_params = {}  # {depth: [(name, param), ...]}

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue

        # 깊이 결정
        depth = _get_param_depth(name)
        if depth not in layer_params:
            layer_params[depth] = []
        layer_params[depth].append((name, param))

    # 총 깊이 수
    max_depth = max(layer_params.keys()) if layer_params else 0

    groups = []
    for depth, params_list in sorted(layer_params.items()):
        # decay_rate^(max_depth - depth) → 깊은 레이어가 높은 LR
        lr_scale = decay_rate ** (max_depth - depth)
        layer_lr = base_lr * lr_scale

        decay_params = []
        no_decay_params = []
        for name, param in params_list:
            if any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": layer_lr,
                "weight_decay": weight_decay,
                "module": f"{prefix}_layer{depth}",
            })
        if no_decay_params:
            groups.append({
                "params": no_decay_params,
                "lr": layer_lr,
                "weight_decay": 0.0,
                "module": f"{prefix}_layer{depth}_nd",
            })

    return groups


def _get_param_depth(param_name: str) -> int:
    """파라미터명에서 레이어 깊이 추출"""
    # "blocks.0.xxx" → depth 1
    # "blocks.5.xxx" → depth 6
    # "patch_embed.xxx" → depth 0
    # "neck.xxx" → max depth (가장 높은 LR)
    parts = param_name.split(".")
    if "patch_embed" in param_name or "pos_embed" in param_name:
        return 0
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1]) + 1
            except ValueError:
                pass
    return 99  # 알 수 없는 위치는 최상위


# ─────────────────────────────────────────────────
# 2. 옵티마이저 빌드
# ─────────────────────────────────────────────────

def build_optimizer(
    model_parts: Dict[str, nn.Module],
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    backbone_lr_factor: float = 0.1,
    layer_lr_decay: float = 1.0,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """
    AdamW 옵티마이저 빌드

    Args:
        model_parts: 모델 구성요소 딕셔너리
        lr: 기본 학습률
        weight_decay: 가중치 감쇠
        backbone_lr_factor: 백본 LR 배율
        layer_lr_decay: 레이어별 LR 감쇠율
        betas: Adam betas

    Returns:
        AdamW optimizer
    """
    param_groups = get_param_groups(
        model_parts, lr, weight_decay,
        backbone_lr_factor, layer_lr_decay,
    )

    optimizer = AdamW(param_groups, lr=lr, betas=betas)
    return optimizer


# ─────────────────────────────────────────────────
# 3. 스케줄러 빌드
# ─────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    scheduler_type: str = "cosine",
) -> LambdaLR:
    """
    학습률 스케줄러 빌드

    ■ Cosine Annealing with Linear Warmup:
      Step 0~warmup: 0 → base_lr (선형 증가)
      Step warmup~total: base_lr → min_lr (코사인 감쇠)

    Args:
        optimizer: 옵티마이저
        total_steps: 총 학습 스텝 수
        warmup_steps: 워밍업 스텝 수
        min_lr_ratio: 최소 학습률 비율 (base_lr 대비)
        scheduler_type: "cosine" 또는 "linear"

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup 구간
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Decay 구간
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(progress, 1.0)

        if scheduler_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            decay = 1.0 - progress
        else:
            decay = 1.0

        return max(min_lr_ratio, decay)

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────
# 4. 유틸리티
# ─────────────────────────────────────────────────

def get_optimizer_summary(optimizer: torch.optim.Optimizer) -> List[dict]:
    """
    옵티마이저 파라미터 그룹 요약

    Returns:
        list of {module, lr, weight_decay, num_params}
    """
    summary = []
    for group in optimizer.param_groups:
        num_params = sum(p.numel() for p in group["params"])
        summary.append({
            "module": group.get("module", "unknown"),
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "num_params": num_params,
        })
    return summary
