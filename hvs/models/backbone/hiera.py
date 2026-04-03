"""
Hiera 백본 (Hierarchical Vision Transformer)

■ 역할:
  입력 이미지에서 다중 스케일 특징(feature)을 추출하는 "눈" 역할.
  SAM2의 핵심 백본으로, 계층적 구조를 가진 Vision Transformer입니다.

■ 핵심 개념 — 왜 Hiera인가?
  - 기존 ViT: 모든 패치를 한 해상도에서 처리 → 느리고 메모리 많이 사용
  - Hiera: 4개 Stage로 나눠서 해상도를 점진적으로 줄임 → 효율적
  - 각 Stage에서 해상도 절반 축소 + 채널 수 2배 증가
  
■ 4단계 처리 과정 (Tiny 기준):
  Stage 1: 256×256, 96ch  → [Block ×1]  → 가장 세밀한 특징
  Stage 2: 128×128, 192ch → [Block ×2]  → 중간 특징
  Stage 3:  64×64, 384ch  → [Block ×7]  → 핵심 특징 (블록 가장 많음)
  Stage 4:  32×32, 768ch  → [Block ×2]  → 전체 맥락 특징
  
■ Window Attention vs Global Attention:
  - Window: 창(window) 내부에서만 Attention 계산 → 빠르지만 범위 제한
  - Global: 전체 이미지에서 Attention 계산 → 느리지만 전체 맥락 파악
  - Hiera는 대부분 Window, 핵심 위치에서만 Global 사용 → 효율적

■ 입출력:
  입력: (B, 3, H, W) — RGB 이미지
  출력: [Stage4, Stage3, Stage2, Stage1] — 4개 스케일 특징맵 리스트
        각각 (B, C, H', W') 형태
"""

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvs.models.backbone.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from hvs.models.model_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """
    해상도 축소 풀링
    
    ■ 역할:
      Stage 전환 시 Query의 해상도를 절반으로 줄이는 연산.
      (B, H, W, C) → pool → (B, H/2, W/2, C)
    
    ■ 왜 필요한가?
      계층적 구조에서 해상도를 줄이면서 채널 수를 늘리면
      넓은 영역의 문맥 정보를 효율적으로 압축할 수 있습니다.
    """
    if pool is None:
        return x
    # (B, H, W, C) → (B, C, H, W) : Pool 연산은 spatial 차원에 적용
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') → (B, H', W', C) : 다시 원래 형태로
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    """
    멀티스케일 어텐션 (Multi-Scale Attention)
    
    ■ 역할:
      표준 Self-Attention에 Q-풀링을 추가하여
      Stage 전환 시 해상도를 줄이면서 Attention을 수행합니다.
    
    ■ 동작 원리:
      1. 입력(x)에서 Q(질문), K(답변 후보), V(실제 정보) 생성
      2. Q에만 MaxPool 적용 → Q의 해상도가 절반으로 줄어듦
      3. Q, K, V로 Scaled Dot-Product Attention 계산
      4. 결과: 줄어든 해상도의 특징맵
    
    ■ Q-풀링의 핵심:
      - K, V는 원래 해상도 유지 (세밀한 정보 보존)
      - Q만 다운샘플 → 출력 해상도 = Q 해상도 (절반)
      - 이를 통해 Stage 간 자연스러운 해상도 전환
    
    Args:
        dim: 입력 차원
        dim_out: 출력 차원
        num_heads: 어텐션 헤드 수 (병렬 관점 수)
        q_pool: Q에 적용할 풀링 모듈 (None이면 일반 Attention)
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        # Q, K, V를 한 번에 생성하는 선형 변환
        self.qkv = nn.Linear(dim, dim_out * 3)
        # 출력 프로젝션
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) 입력 특징맵
        Returns:
            (B, H', W', dim_out) — H'은 Q-풀링이 있으면 H//2
        """
        B, H, W, _ = x.shape
        # QKV 생성: (B, H*W, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)  # 각각 (B, H*W, num_heads, head_dim)

        # Q-풀링: Stage 전환 시 Q의 해상도만 줄임
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # 줄어든 해상도
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Scaled Dot-Product Attention (PyTorch 내장 최적화 버전)
        # (B, num_heads, N, head_dim) 형태로 변환 후 계산
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # (B, num_heads, N, head_dim) → (B, H, W, dim_out)
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):
    """
    멀티스케일 트랜스포머 블록
    
    ■ 역할:
      Hiera의 기본 구성 블록. 다음을 순차적으로 수행:
      1. LayerNorm → Attention → Residual 연결
      2. LayerNorm → MLP → Residual 연결
    
    ■ Residual 연결이란?
      블록의 출력 = 블록의 입력 + 블록이 학습한 것
      비유: "학생(입력)이 수업(블록)을 듣고 나면, 기존 지식에 새로 배운 걸 더한다"
      → 깊은 네트워크도 안정적으로 학습 가능
    
    ■ Window Attention 모드:
      window_size > 0이면 Window Attention 적용 (지역적)
      window_size == 0이면 Global Attention 적용 (전체적)
    
    Args:
        dim: 입력 차원
        dim_out: 출력 차원 (Stage 전환 시 dim ≠ dim_out)
        num_heads: 어텐션 헤드 수
        mlp_ratio: MLP 은닉 차원 = dim_out * mlp_ratio
        drop_path: DropPath 확률 (Stochastic Depth)
        q_stride: Q-풀링 stride (Stage 전환 시 (2,2))
        window_size: 윈도우 크기 (0이면 Global Attention)
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        # Q-풀링: Stage 간 해상도 축소용
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim, dim_out, num_heads=num_heads, q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out, int(dim_out * mlp_ratio), dim_out,
            num_layers=2, activation=act_layer,
        )

        # 차원이 바뀌면 shortcut에도 선형 변환 적용
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) 입력
        Returns:
            (B, H', W', dim_out) — H'은 Q-풀링이 있으면 H//2
        """
        shortcut = x  # Residual 연결을 위해 입력 보존

        x = self.norm1(x)

        # 차원이 바뀌면 shortcut도 변환 + 풀링
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # ---- Window Partition ----
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # ---- Attention ----
        x = self.attn(x)

        if self.q_stride:
            # Q-풀링으로 크기가 바뀌었으므로 window 크기도 조정
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # ---- Window Unpartition ----
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        # ---- Residual + MLP ----
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Hiera 백본 (Hierarchical Vision Transformer)
    
    ■ 전체 구조 (Tiny 기준):
      입력 (B, 3, 1024, 1024)
        → PatchEmbed: (B, 256, 256, 96)     # stride=4로 4배 다운샘플
        → + Positional Embedding                # 위치 정보 추가
        → Stage 1: Block ×1, 96ch, 256×256     # 세밀한 특징
        → Stage 2: Block ×2, 192ch, 128×128    # 중간 특징 (Q-pool로 축소)
        → Stage 3: Block ×7, 384ch, 64×64      # 핵심 특징 (블록 가장 많음)
        → Stage 4: Block ×2, 768ch, 32×32      # 전체 맥락
      출력: [Stage4_feat, Stage3_feat, Stage2_feat, Stage1_feat]
    
    ■ 위치 임베딩 (Positional Embedding):
      Transformer는 입력 순서를 자체적으로 알 수 없음 → 위치 정보를 명시적으로 추가
      Hiera는 Windowed Positional Embedding 사용:
        - 전역 위치 임베딩 + 윈도우별 지역 위치 임베딩을 결합
        - 물체 위치에 대한 전역/지역 정보 동시 제공
    
    Args:
        embed_dim: 초기 임베딩 차원 (Stage 1의 채널 수, Tiny=96)
        num_heads: 초기 어텐션 헤드 수 (Tiny=1)
        drop_path_rate: Stochastic Depth 비율
        q_pool: Q-풀링이 적용될 Stage 수 (기본 3 = Stage 1,2,3에서 풀링)
        q_stride: Q-풀링 stride (기본 (2,2) = 해상도 절반)
        stages: 각 Stage별 블록 수 (Tiny=[1,2,7,2])
        dim_mul: Stage 전환 시 채널 수 배율 (기본 2.0 = 2배 증가)
        head_mul: Stage 전환 시 헤드 수 배율 (기본 2.0 = 2배 증가)
        window_pos_embed_bkg_spatial_size: 전역 위치 임베딩 크기
        window_spec: Stage별 윈도우 크기
        global_att_blocks: Global Attention을 사용할 블록 인덱스
        return_interm_layers: 중간 Stage 출력을 반환할지 여부
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        return_interm_layers: bool = True,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)  # 총 블록 수
        self.q_stride = q_stride
        # 각 Stage의 마지막 블록 인덱스 계산
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        # Q-풀링이 적용될 블록 인덱스 (각 Stage 시작 블록)
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        # ---- 패치 임베딩 ----
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        
        # Global Attention 적용 블록 인덱스
        self.global_att_blocks = global_att_blocks

        # ---- 위치 임베딩 ----
        # 전역 위치 임베딩 (배경)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        # 윈도우별 지역 위치 임베딩
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        # ---- Stochastic Depth 스케줄 ----
        # 낮은 블록: 드롭 적음, 높은 블록: 드롭 많음 (점진적 증가)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ---- 블록 생성 ----
        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]

            # Global Attention 블록 여부
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            # Stage 전환: 다음 Stage의 첫 블록
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        # 각 Stage의 출력 채널 수 기록 (역순: Stage4 → Stage1)
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """
        입력 크기에 맞게 위치 임베딩을 보간(interpolate)하여 반환.
        
        ■ 동작:
          1. 전역 위치 임베딩을 입력 크기로 보간
          2. 윈도우 지역 임베딩을 타일링(반복)하여 더함
          3. 결과: 전역 + 지역 위치 정보가 합쳐진 임베딩
        """
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 입력 이미지
        Returns:
            outputs: 다중 스케일 특징맵 리스트
                     [Stage4_feat, Stage3_feat, Stage2_feat, Stage1_feat]
                     각각 (B, C, H', W') 형태 (역순 = 저해상도부터)
        """
        # 패치 임베딩: (B, 3, H, W) → (B, H/4, W/4, embed_dim)
        x = self.patch_embed(x)

        # 위치 임베딩 추가
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 각 Stage의 마지막 블록에서 출력 저장
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                # (B, H, W, C) → (B, C, H, W) 형태로 변환
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_num_layers(self) -> int:
        """총 블록 수 반환"""
        return len(self.blocks)
