"""
메모리 인코더 (Memory Encoder)

■ 역할:
  현재 프레임의 예측 결과(마스크 + 이미지 특징)를 "메모리"로 압축합니다.
  이 메모리가 다음 프레임 처리 시 과거 정보로 활용됩니다.

■ 핵심 과정:
  1. 예측 마스크 → MaskDownSampler로 다운샘플링
  2. 이미지 특징 + 다운샘플된 마스크 → 합산
  3. Fuser(ConvNeXt 블록)로 특징 정제
  4. 위치 인코딩 추가

■ 비유:
  수업(현재 프레임)이 끝나면 중요 내용을 노트(메모리)에 정리해두고,
  다음 수업(다음 프레임)에서 그 노트를 참고하는 것

■ 산업 결함 검출에서의 역할:
  시간 순서대로 촬영된 이미지에서 결함의 위치/크기 변화를 추적할 때,
  이전 프레임의 결함 정보를 메모리로 저장하여 연속성 유지
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvs.models.model_utils import DropPath, LayerNorm2d, get_clones


class MaskDownSampler(nn.Module):
    """
    마스크 다운샘플러 (Mask Down-Sampler)

    ■ 역할:
      예측 마스크를 Progressive하게 축소합니다.
      (B, 1, H, W) → Conv → Conv → ... → (B, embed_dim, H/stride, W/stride)

    ■ 예 (total_stride=16, stride=4):
      Layer 1: (1, H, W) → Conv(4×4, stride=4) → (16, H/4, W/4)
      Layer 2: (16, H/4, W/4) → Conv(4×4, stride=4) → (256, H/16, W/16)
      Final:   (256, H/16, W/16) → Conv(1×1) → (embed_dim, H/16, W/16)

    Args:
        embed_dim: 출력 채널 (보통 256)
        kernel_size: 다운샘플 컨볼루션 커널 크기
        stride: 각 단계의 stride
        total_stride: 총 다운샘플 배율
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride ** 2)
            self.encoder.append(
                nn.Conv2d(mask_in_chans, mask_out_chans,
                          kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CXBlock(nn.Module):
    """
    ConvNeXt 블록

    ■ 역할:
      메모리 인코더에서 마스크+이미지 특징을 정제하는 잔차 블록.
      Depthwise Conv → LayerNorm → 1×1 Conv → GELU → 1×1 Conv 구조.

    ■ ConvNeXt 선택 이유:
      - Transformer 블록보다 빠르면서 유사한 성능
      - 지역적 정보(convolution)와 채널 간 정보(1×1)를 효율적으로 결합

    Args:
        dim: 채널 수
        kernel_size: Depthwise Conv 커널 크기 (기본 7)
        drop_path: DropPath 확률
        layer_scale_init_value: Layer Scale 초기값
        use_dwconv: Depthwise Conv 사용 여부
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) → (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        x = residual + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """
    특징 융합기 (Fuser)

    ■ 역할:
      여러 CXBlock을 쌓아서 마스크+이미지 특징을 정제.
      선택적으로 입력 프로젝션(1×1 Conv)도 적용.

    Args:
        layer: 복제할 기본 블록 (CXBlock)
        num_layers: 블록 수
        dim: 채널 수 (input_projection 사용 시 필요)
        input_projection: 입력에 1×1 Conv 적용 여부
    """

    def __init__(self, layer: nn.Module, num_layers: int,
                 dim: int = None, input_projection: bool = False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)
        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """
    메모리 인코더 (Memory Encoder)

    ■ 전체 파이프라인:
      예측 마스크 (B, 1, H, W)
        → sigmoid (로짓 → 확률)
        → MaskDownSampler → (B, 256, H/16, W/16)
      이미지 특징 (B, 256, H/16, W/16)
        → pix_feat_proj (1×1 Conv)
        → + 다운샘플된 마스크
        → Fuser (CXBlock ×2)
        → out_proj
      + 위치 인코딩

    Args:
        out_dim: 출력 차원 (메모리 차원, 보통 64)
        mask_downsampler: MaskDownSampler 인스턴스
        fuser: Fuser 인스턴스
        position_encoding: 위치 인코딩 모듈
        in_dim: 이미지 특징 차원 (보통 256)
    """

    def __init__(
        self,
        out_dim: int,
        mask_downsampler: nn.Module,
        fuser: nn.Module,
        position_encoding: nn.Module,
        in_dim: int = 256,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict:
        """
        Args:
            pix_feat: (B, C, H, W) 이미지 특징맵 (FPN 출력)
            masks: (B, 1, H, W) 예측 마스크 로짓
            skip_mask_sigmoid: True이면 sigmoid 생략 (이미 확률일 때)
        Returns:
            dict: {"vision_features": (B, out_dim, H', W'),
                   "vision_pos_enc": [(B, out_dim, H', W')]}
        """
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}
