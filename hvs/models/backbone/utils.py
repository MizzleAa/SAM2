"""
백본 유틸리티 — 패치 임베딩과 윈도우 파티션 연산

■ PatchEmbed: 이미지를 작은 패치 단위로 나누어 벡터로 변환
   - 비유: 큰 사진을 작은 타일(퍼즐 조각)로 나누는 것
   - 예: 1024×1024 이미지 → 7×7 커널, stride 4 → 256×256 패치, 각 embed_dim 차원

■ window_partition / window_unpartition:
   - Transformer의 Self-Attention은 모든 패치 쌍을 계산 → 매우 느림
   - Window Attention: 이미지를 작은 창(window)으로 나눠서 창 내부끼리만 계산 → 빠름
   - 비유: 전 세계 사람이 서로 대화(Global) vs 같은 교실 안에서만 대화(Window)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int):
    """
    이미지를 겹치지 않는 윈도우(창)로 분할합니다. 필요하면 패딩 추가.

    ■ 동작 원리:
      입력 (B, H, W, C) → 창 크기로 분할 → (B*num_windows, ws, ws, C)
      
    ■ 비유:
      큰 격자 종이를 작은 정사각형 블록으로 잘라내는 것

    Args:
        x: 입력 텐서, shape = (B, H, W, C)
           B: 배치 크기, H/W: 높이/너비, C: 채널(특징 차원)
        window_size: 각 윈도우의 한 변 크기 (예: 8이면 8×8 창)

    Returns:
        windows: (B * num_windows, window_size, window_size, C) 분할된 윈도우들
        (Hp, Wp): 패딩 후의 높이/너비 (원본 복원에 필요)
    """
    B, H, W, C = x.shape

    # window_size로 나누어 떨어지지 않으면 패딩 추가
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # (B, H, W, C) → (B, H//ws, ws, W//ws, ws, C) → (B*nw, ws, ws, C)
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    분할된 윈도우를 원래 이미지 형태로 복원합니다. 패딩도 제거합니다.

    Args:
        windows: (B * num_windows, window_size, window_size, C) 분할된 윈도우
        window_size: 윈도우 한 변 크기
        pad_hw: 패딩 후의 (Hp, Wp)
        hw: 원본 (H, W) — 패딩 전 크기

    Returns:
        x: (B, H, W, C) 복원된 텐서
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    # 패딩 영역 제거
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    """
    패치 임베딩 (Patch Embedding)
    
    ■ 역할:
      입력 이미지를 작은 패치(조각)로 나누고, 각 패치를 고정 차원 벡터로 변환.
      이것이 Transformer의 "토큰" 역할을 합니다.
    
    ■ 동작:
      Conv2d로 구현 — 커널이 패치를 잘라내면서 동시에 임베딩(벡터 변환)
      (B, 3, H, W) → Conv2d → (B, embed_dim, H//stride, W//stride) → permute → (B, H', W', C)
    
    ■ 비유:
      사진을 격자로 나누고, 각 칸의 색상/패턴 정보를 요약한 "신분증"을 만드는 것
    
    Args:
        kernel_size: 패치를 자르는 필터 크기 (기본 7×7)
        stride: 패치 간 간격 (기본 4 → 4배 다운샘플)
        padding: 경계 패딩
        in_chans: 입력 이미지 채널 (RGB=3)
        embed_dim: 출력 임베딩 차원 (패치 하나의 벡터 길이)
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 입력 이미지
        Returns:
            (B, H', W', embed_dim) — 패치별 임베딩 벡터
        """
        x = self.proj(x)
        # (B, C, H, W) → (B, H, W, C) : Transformer 표준 형식
        x = x.permute(0, 2, 3, 1)
        return x
