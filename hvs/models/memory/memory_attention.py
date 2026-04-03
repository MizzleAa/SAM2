"""
메모리 어텐션 (Memory Attention)

■ 역할:
  현재 프레임의 이미지 특징을 과거 메모리(이전 프레임들의 기억)와
  교차 참조하여 시간적 문맥을 반영합니다.

■ 핵심 동작:
  1. Self-Attention: 현재 프레임 내부의 공간적 관계 정제
  2. Cross-Attention: 현재 프레임 ↔ 과거 메모리 간 정보 교환
     ("이 영역이 이전 프레임에서도 결함이었나?")
  3. FFN(MLP): 결합한 정보를 비선형 변환

■ 비유:
  새 문제를 풀 때(현재 프레임) 기존 노트(과거 메모리)를 참조하면서
  현재 정보와 과거 정보를 종합하여 더 나은 판단을 하는 것

■ Object Pointer:
  SAM2의 고유 기능. 과거 프레임에서 추적한 객체의 "포인터"를
  현재 프레임의 어텐션에 추가하여 물체 추적을 강화.
"""

from typing import Optional

import torch
from torch import nn, Tensor

from hvs.models.model_utils import get_activation_fn, get_clones
from hvs.models.head.transformer import Attention, RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """
    메모리 어텐션 레이어

    ■ 3단계 처리:
      Step 1: Self-Attention (현재 프레임 내 공간 관계)
      Step 2: Cross-Attention (현재 ↔ 과거 메모리)
      Step 3: FFN (정보 변환)

    Args:
        d_model: 특징 차원 (보통 256)
        dim_feedforward: FFN 차원 (보통 2048)
        dropout: 드롭아웃 확률
        self_attention: Self-Attention 모듈
        cross_attention: Cross-Attention 모듈
        pos_enc_at_attn: Self-Attention에 위치 인코딩 추가 여부
        pos_enc_at_cross_attn_keys: Cross-Attention Key에 위치 인코딩 추가
        pos_enc_at_cross_attn_queries: Cross-Attention Query에 위치 인코딩 추가
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # FFN (Feed-Forward Network)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """Self-Attention: 현재 프레임 내 관계 정제"""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """Cross-Attention: 현재 프레임 ↔ 과거 메모리"""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, N, C) 현재 프레임 특징
            memory: (B, M, C) 과거 메모리 특징
            pos: memory의 위치 인코딩
            query_pos: tgt의 위치 인코딩
        """
        # Self-Attn → Cross-Attn → FFN
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """
    메모리 어텐션 (Memory Attention)

    ■ 전체 파이프라인:
      현재 프레임 특징 (N, B, C)
        → [선택] + 0.1 * 위치 인코딩
        → MemoryAttentionLayer × num_layers
          (각 레이어: Self-Attn → Cross-Attn → FFN)
        → LayerNorm
      출력: 메모리가 반영된 현재 프레임 특징

    Args:
        d_model: 특징 차원
        pos_enc_at_input: 입력에 위치 인코딩 가산 여부
        layer: MemoryAttentionLayer (복제할 기본 레이어)
        num_layers: 레이어 수 (보통 4)
        batch_first: 배치 차원이 첫 번째인지
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            curr: (N, B, C) 또는 리스트 — 현재 프레임 특징
            memory: (M, B, C) — 과거 메모리 특징
            curr_pos: curr의 위치 인코딩
            memory_pos: memory의 위치 인코딩
            num_obj_ptr_tokens: Object Pointer 토큰 수 (RoPE 제외용)
        Returns:
            (N, B, C) 메모리 반영된 현재 프레임 특징
        """
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )

        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)

        return normed_output
