"""
위치 인코딩 (Position Encoding)

■ 왜 필요한가?
  Transformer는 순서/위치 정보를 자체적으로 알 수 없습니다.
  "이 패치가 이미지의 왼쪽 위에 있는지, 오른쪽 아래에 있는지" 모릅니다.
  → 위치 인코딩을 명시적으로 추가해서 "공간 좌표"를 알려줘야 합니다.

■ PositionEmbeddingSine (사인 위치 인코딩):
  - sin/cos 함수의 주기를 다르게 하여 각 위치마다 고유한 벡터 생성
  - 학습 불필요 (수학적으로 고정됨)
  - SAM2에서 FPN Neck의 출력에 추가하는 데 사용

■ PositionEmbeddingRandom (랜덤 위치 인코딩):
  - 랜덤 주파수를 사용한 푸리에 특징 기반 인코딩
  - SAM의 Prompt Encoder에서 점/박스 좌표를 인코딩할 때 사용
"""

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    사인 기반 2D 위치 인코딩
    
    ■ 역할:
      이미지 특징맵의 각 위치 (h, w)에 고유한 위치 벡터를 생성합니다.
      "Attention Is All You Need" 논문의 위치 인코딩을 2D 이미지로 확장한 것.
    
    ■ 동작:
      1. y축 / x축 각각의 좌표를 정규화 (0~1)
      2. 다양한 주파수의 sin/cos 함수를 적용
      3. y축 인코딩 + x축 인코딩을 결합 → 2D 위치 벡터
    
    ■ 비유:
      GPS 좌표처럼 각 위치에 고유한 "주소"를 부여하는 것.
      다만 단순한 숫자가 아니라, sin/cos로 변환하여
      Transformer가 위치 정보를 더 잘 활용할 수 있게 합니다.
    
    Args:
        num_pos_feats: 위치 인코딩 차원 (보통 d_model과 동일, 예: 256)
        temperature: 주파수 스케일링 (클수록 낮은 주파수)
        normalize: 좌표를 [0, 2π] 범위로 정규화할지
        scale: 정규화 시 스케일 (기본 2π)
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "num_pos_feats must be even"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}

    @torch.no_grad()
    def _encode_xy(self, x, y):
        """x, y 좌표를 sin/cos로 인코딩"""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """점 좌표를 인코딩 (Prompt Encoder용)"""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        """위치 인코딩 생성 (캐시 사용)"""
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        # y, x 좌표 그리드 생성
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        # 정규화: [0, 2π] 범위로 변환
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # sin/cos 인코딩
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # y + x 결합 → (B, num_pos_feats, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 특징맵
        Returns:
            (B, num_pos_feats, H, W) 위치 인코딩
        """
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)


class PositionEmbeddingRandom(nn.Module):
    """
    랜덤 기반 위치 인코딩
    
    ■ 역할:
      랜덤 주파수를 가진 가우시안 행렬을 사용하여
      좌표를 고차원 푸리에 특징으로 변환합니다.
      SAM의 Prompt Encoder에서 점/박스 좌표를 인코딩할 때 사용.
    
    ■ 왜 랜덤?
      - 좌표를 그대로 사용하면 모델이 위치 정보를 잘 활용 못함
      - 랜덤 푸리에 특징으로 변환하면 더 표현력이 풍부해짐
    
    Args:
        num_pos_feats: 출력 차원의 절반 (sin + cos → 2배)
        scale: 랜덤 행렬의 분산 스케일
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """[0,1] 범위의 좌표를 sin/cos 인코딩"""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        지정 크기의 그리드에 대한 위치 인코딩 생성
        
        Args:
            size: (H, W) 그리드 크기
        Returns:
            (C, H, W) 위치 인코딩
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # (C, H, W)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        정규화되지 않은 좌표를 인코딩
        
        Args:
            coords_input: (B, N, 2) 좌표 (x, y 순서, 픽셀 단위)
            image_size: (H, W) 이미지 크기
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # (B, N, C)
