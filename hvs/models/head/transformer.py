"""
Two-way Transformer + Attention 모듈

■ 역할:
  이미지 특징과 프롬프트(점/박스) 간의 양방향 정보 교환을 수행합니다.
  이것이 SAM의 핵심 — "이 점이 가리키는 물체가 이미지의 어느 영역인가?"를 결정.

■ Two-way (양방향) 의미:
  1. Token → Image: 프롬프트가 이미지의 어느 영역에 주목할지 결정
     (비유: "이 점 근처에서 물체 경계를 찾아라")
  2. Image → Token: 이미지가 프롬프트에게 "여기에 이런 특징이 있다"고 알림
     (비유: "이 영역에 스크래치 같은 경계가 있다")

■ 각 블록의 4단계 처리:
  Step 1: Self-Attention (프롬프트 토큰끼리 정보 교환)
  Step 2: Cross-Attention Token→Image (프롬프트가 이미지를 관찰)
  Step 3: MLP (특징 변환)
  Step 4: Cross-Attention Image→Token (이미지가 프롬프트 정보를 흡수)

■ Attention (어텐션):
  Q(Query, 질문), K(Key, 색인), V(Value, 정보)
  "어떤 질문(Q)과 가장 관련 있는 색인(K)를 찾아 해당 정보(V)를 가져온다"
  비유: 도서관에서 검색어(Q)로 색인 카드(K)를 찾고, 해당 책(V)을 가져오는 것
"""

import math
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from hvs.models.model_utils import MLP


class Attention(nn.Module):
    """
    표준 Multi-Head Attention

    ■ 동작:
      1. 입력을 Q, K, V로 선형 변환
      2. 여러 "헤드"로 분리 (각 헤드가 다른 관점에서 정보 추출)
      3. Scaled Dot-Product Attention 계산
      4. 모든 헤드의 결과를 합침 → 출력 프로젝션

    ■ downsample_rate:
      내부 차원을 줄여서 계산량 절감.
      downsample_rate=2이면 차원이 절반 → 계산량 1/4
      (Cross-Attention에서 사용하여 속도 향상)

    Args:
        embedding_dim: 입출력 차원
        num_heads: 어텐션 헤드 수
        downsample_rate: 내부 차원 축소 비율 (2이면 절반)
        dropout: 드롭아웃 확률
        kv_in_dim: K, V의 입력 차원 (None이면 embedding_dim과 동일)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """(B, N, C) -> (B, num_heads, N, C_per_head)"""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """(B, num_heads, N, C_per_head) -> (B, N, C)"""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 선형 변환: 입력 → Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 멀티 헤드로 분리
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Scaled Dot-Product Attention (PyTorch 최적화 버전)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


# ────────────────────────────────────────────────────────────
# Rotary Positional Encoding (RoPE)
#
# ■ 역할:
#   일반 위치 인코딩과 달리, Q와 K를 회전시켜 위치 정보를 주입합니다.
#   상대적 위치 관계를 자연스럽게 표현할 수 있어
#   MemoryAttention의 Self/Cross Attention에서 사용됩니다.
#
# ■ 핵심 아이디어:
#   - 각 위치에 대해 복소수 회전(polar) 생성
#   - Q, K를 복소수로 변환 후 회전 적용
#   - 내적 시 상대적 회전 차이만 남아 상대 위치 인코딩 효과
#
# 출처: Facebook SAM2 (sam2/modeling/position_encoding.py)
# ────────────────────────────────────────────────────────────

def _init_t_xy(end_x: int, end_y: int):
    """2D 그리드의 x, y 좌표 벡터 생성"""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    2D Axial RoPE 주파수 생성

    Args:
        dim: 헤드 차원 (embedding_dim // num_heads)
        end_x, end_y: 특징맵 크기 (보통 64×64)
        theta: 주파수 베이스 (높을수록 저주파)
    Returns:
        (end_x * end_y, dim//2) 복소수 텐서
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = _init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """freqs_cis를 x에 브로드캐스트 가능하도록 reshape"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    Q, K에 Rotary Position Encoding 적용

    Args:
        xq: (B, H, N_q, D) Query — 분리된 헤드 형태
        xk: (B, H, N_k, D) Key
        freqs_cis: (N, D//2) 복소수 RoPE 주파수
        repeat_freqs_k: K의 시퀀스가 Q보다 긴 경우 freqs를 반복할지
    Returns:
        (xq_out, xk_out): RoPE 적용된 Q, K
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(Attention):
    """
    RoPE (Rotary Position Encoding) Attention

    ■ Attention을 상속하여 RoPE를 추가한 버전
      MemoryAttention에서 Self-Attention과 Cross-Attention에 사용됩니다.

    ■ 핵심 차이:
      - 일반 Attention: Q, K에 위치 인코딩을 더함 (additive)
      - RoPE Attention: Q, K를 회전시킴 (multiplicative)
      → 상대적 위치 관계가 내적에 자연스럽게 반영됨

    ■ num_k_exclude_rope:
      Object Pointer 토큰은 공간적 위치가 없으므로
      RoPE를 적용하지 않아야 합니다. 이 값만큼의 끝부분 K 토큰을 제외합니다.

    Args:
        rope_theta: RoPE 주파수 베이스 (기본 10000.0)
        rope_k_repeat: K가 Q보다 긴 경우 freqs를 반복할지
        feat_sizes: 특징맵 크기 (기본 [64, 64])
    """

    def __init__(
        self,
        *args,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: Tuple[int, int] = (64, 64),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # 선형 변환
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 멀티 헤드 분리
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # RoPE 적용
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        # Object Pointer 토큰은 RoPE 제외
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class TwoWayAttentionBlock(nn.Module):
    """
    양방향 어텐션 블록

    ■ 4단계 처리 순서:
      1. Self-Attention: 프롬프트 토큰끼리 정보 공유
         (예: 전경 점과 배경 점이 서로의 위치를 인식)
      2. Cross-Attention (Token→Image): 프롬프트가 이미지에서 관련 영역 탐색
         ("이 점 근처에 어떤 특징이 있는가?")
      3. MLP: 추출한 정보를 비선형 변환
      4. Cross-Attention (Image→Token): 이미지가 프롬프트 정보를 반영
         ("프롬프트에서 지시한 영역에 집중하겠다")

    Args:
        embedding_dim: 임베딩 차원
        num_heads: 어텐션 헤드 수
        mlp_dim: MLP 은닉 차원
        skip_first_layer_pe: 첫 레이어에서 위치 인코딩 생략 여부
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            queries: (B, N_tokens, C) 프롬프트 토큰
            keys: (B, N_image, C) 이미지 특징
            query_pe: queries의 위치 인코딩
            key_pe: keys의 위치 인코딩
        Returns:
            queries, keys: 업데이트된 프롬프트 토큰과 이미지 특징
        """
        # Step 1: Self-Attention (프롬프트 토큰끼리)
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Step 2: Cross-Attention (Token → Image)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # Step 3: MLP
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Step 4: Cross-Attention (Image → Token)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    양방향 트랜스포머 (Two-Way Transformer)

    ■ 전체 파이프라인:
      입력:
        - image_embedding: (B, C, H, W) 이미지 특징맵
        - image_pe: (B, C, H, W) 이미지 위치 인코딩
        - point_embedding: (B, N, C) 프롬프트 토큰

      처리: depth개의 TwoWayAttentionBlock을 순차 적용
        → 마지막에 Token→Image 최종 어텐션 추가

      출력:
        - queries: (B, N, C) 처리된 프롬프트 토큰
        - keys: (B, H*W, C) 처리된 이미지 특징

    Args:
        depth: 블록 수 (보통 2)
        embedding_dim: 임베딩 차원 (보통 256)
        num_heads: 어텐션 헤드 수 (보통 8)
        mlp_dim: MLP 차원 (보통 2048)
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        # 최종 Token→Image 어텐션 (마지막 정제)
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image_embedding: (B, C, H, W) 이미지 특징맵
            image_pe: (1, C, H, W) 이미지 위치 인코딩
            point_embedding: (B, N, C) 프롬프트 토큰 (iou_token + mask_tokens + sparse)
        Returns:
            queries: (B, N, C) 처리된 프롬프트 토큰
            keys: (B, H*W, C) 처리된 이미지 특징
        """
        # (B, C, H, W) → (B, H*W, C) 시퀀스 형태로 변환
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        # 양방향 어텐션 블록 반복 적용
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 최종 Token→Image 어텐션
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
