"""
공통 모델 유틸리티 — DropPath, MLP, LayerNorm2d

■ DropPath (Stochastic Depth):
  학습 중 랜덤하게 일부 블록을 건너뛰어 과적합을 방지.
  비유: 학습 중 가끔 "쉬는 시간"을 가져서 모든 길에 의존하지 않게 하는 것.

■ MLP (Multi-Layer Perceptron):
  특징을 비선형 변환하는 완전연결 계층 묶음.
  Transformer 블록에서 Attention 후 특징을 추가 변환할 때 사용.

■ LayerNorm2d:
  2D 특징맵(이미지 형태)에 Layer Normalization 적용.
  학습 안정화에 필수적인 정규화 기법.
"""

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Stochastic Depth (확률적 깊이)
    
    ■ 역할:
      학습 중 랜덤하게 블록의 출력을 0으로 만들어 건너뜁니다.
      이렇게 하면 모델이 특정 경로에만 의존하지 않고 다양한 특징을 학습합니다.
    
    ■ 동작:
      - 학습 중: drop_prob 확률로 출력을 0으로 → residual만 남음
      - 추론 중: 항상 원래 출력 사용 (드롭 없음)
    
    ■ 비유:
      팀 프로젝트에서 랜덤으로 한 명을 빼고 훈련하면
      나머지 멤버들이 더 독립적으로 일하는 능력을 키우는 것
    
    Args:
        drop_prob: 드롭 확률 (0.0 = 드롭 안 함, 0.1 = 10% 확률로 드롭)
        scale_by_keep: 드롭 안 된 샘플 스케일링 여부
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 배치 차원만 랜덤, 나머지 차원은 공유
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    """
    다층 퍼셉트론 (Multi-Layer Perceptron)
    
    ■ 역할:
      입력 특징을 비선형 변환하여 더 복잡한 패턴을 표현.
      Transformer 블록에서 Self-Attention 후 특징을 추가 변환할 때 사용.
    
    ■ 동작:
      Linear → Activation → Linear → Activation → ... → Linear
      (마지막 레이어에는 활성화 함수를 적용하지 않음)
    
    ■ 비유:
      원재료(입력)를 여러 공정(레이어)을 거쳐 완제품(출력)으로 가공
    
    Args:
        input_dim: 입력 차원
        hidden_dim: 은닉 차원 (보통 input_dim * 4)
        output_dim: 출력 차원
        num_layers: 레이어 수 (최소 2)
        activation: 활성화 함수 클래스 (기본: ReLU)
        sigmoid_output: 출력에 sigmoid 적용 여부
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization
    
    ■ 역할:
      이미지 형태(BCHW)의 특징맵에 Layer Normalization 적용.
      각 채널의 평균/분산을 0/1로 정규화하여 학습을 안정화.
    
    ■ 표준 LayerNorm과의 차이:
      - LayerNorm: (B, N, C) 형태의 시퀀스 데이터용
      - LayerNorm2d: (B, C, H, W) 형태의 이미지 데이터용
    
    Args:
        num_channels: 채널 수
        eps: 분산을 나눌 때 0 방지용 작은 값
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_activation_fn(activation: str):
    """문자열로 활성화 함수 반환"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """주어진 모듈을 N개 깊은 복사"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
