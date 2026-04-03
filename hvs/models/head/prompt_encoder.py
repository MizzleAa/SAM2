"""
프롬프트 인코더 (Prompt Encoder)

■ 역할:
  사용자가 입력한 프롬프트(점, 박스, 마스크)를 모델이 이해할 수 있는
  임베딩 벡터로 변환합니다.

■ 프롬프트 유형별 처리:
  1. 점(Point): 좌표 + 레이블(전경/배경) → 위치 인코딩 + 레이블 임베딩
     - label=1: 전경 점 ("여기가 물체입니다")
     - label=0: 배경 점 ("여기는 물체가 아닙니다")
     - label=2: 박스 왼쪽 상단
     - label=3: 박스 오른쪽 하단
     - label=-1: 패딩 (무시)

  2. 박스(Box): 두 꼭짓점(좌상단, 우하단)을 점으로 변환

  3. 마스크(Mask): 이전 예측 마스크를 다운샘플링하여 Dense 임베딩으로 변환
     (이전 마스크를 피드백으로 주어 반복 예측 개선에 사용)

■ 출력:
  - sparse_embeddings: (B, N, C) — 점/박스 프롬프트의 임베딩
  - dense_embeddings: (B, C, H, W) — 마스크 프롬프트의 임베딩 (없으면 no_mask_embed)
"""

from typing import Optional, Tuple, Type

import torch
from torch import nn

from hvs.models.backbone.position_encoding import PositionEmbeddingRandom
from hvs.models.model_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    """
    프롬프트 인코더 — 점/박스/마스크 프롬프트를 임베딩으로 변환

    ■ 내부 구성요소:
      - pe_layer: 점/박스 좌표를 고차원 벡터로 변환 (랜덤 위치 인코딩)
      - point_embeddings: 4개의 학습 가능한 레이블 임베딩
        [0]=배경점, [1]=전경점, [2]=박스좌상단, [3]=박스우하단
      - not_a_point_embed: 패딩 점용 임베딩 (label=-1)
      - mask_downscaling: 마스크를 다운샘플링하여 Dense 임베딩으로 변환
      - no_mask_embed: 마스크가 없을 때 사용하는 기본 임베딩

    Args:
        embed_dim: 임베딩 차원 (보통 256)
        image_embedding_size: 이미지 특징맵 크기 (H, W), 예: (64, 64)
        input_image_size: 원본 이미지 크기, 예: (1024, 1024)
        mask_in_chans: 마스크 다운샘플링의 중간 채널 수 (보통 16)
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        # 랜덤 위치 인코딩 — 점 좌표를 고차원 벡터로 변환
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 4개의 점 레이블 임베딩 (전경/배경/박스 모서리)
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        # 패딩 점 (무시해야 할 점)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # 마스크 다운샘플링 네트워크
        # mask_input_size = 4 * image_embedding_size (예: 256×256)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        # 마스크가 없을 때 사용하는 기본 임베딩
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        이미지 특징맵 크기에 맞는 Dense 위치 인코딩 반환
        Returns: (1, embed_dim, H, W) 형태
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        점 프롬프트를 임베딩으로 변환

        Args:
            points: (B, N, 2) 점 좌표 (x, y)
            labels: (B, N) 점 레이블 (-1/0/1/2/3)
            pad: 패딩 점을 추가할지 (박스가 없을 때)
        Returns:
            (B, N', embed_dim) 점 임베딩
        """
        points = points + 0.5  # 픽셀 중심으로 이동
        if pad:
            # 박스가 없으면 패딩 점 1개 추가 (label=-1)
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        # 좌표를 위치 인코딩으로 변환
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # 레이블에 따라 해당 임베딩을 더함
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        박스 프롬프트를 임베딩으로 변환
        Args:
            boxes: (B, 4) 박스 좌표 (x1, y1, x2, y2)
        Returns:
            (B, 2, embed_dim) 두 모서리의 임베딩
        """
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """마스크를 다운샘플링하여 Dense 임베딩으로 변환"""
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """입력 프롬프트에서 배치 크기 추출"""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        프롬프트를 Sparse/Dense 임베딩으로 변환

        Args:
            points: (좌표 (B,N,2), 레이블 (B,N)) 튜플 또는 None
            boxes: (B, 4) 박스 좌표 또는 None
            masks: (B, 1, H, W) 마스크 또는 None
        Returns:
            sparse_embeddings: (B, N, embed_dim) — 점/박스 임베딩
            dense_embeddings: (B, embed_dim, H, W) — 마스크 임베딩
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            # 마스크가 없으면 no_mask_embed를 이미지 크기로 확장
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
