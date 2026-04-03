"""
마스크 디코더 (Mask Decoder)

■ 역할:
  이미지 특징과 프롬프트 임베딩을 받아 최종 세그멘테이션 마스크를 생성합니다.
  SAM 파이프라인의 마지막 단계: "이미지에서 이 물체의 정확한 윤곽선은?"

■ 핵심 동작:
  1. 출력 토큰 준비 (IoU 토큰 + 마스크 토큰 + 프롬프트 토큰)
  2. Two-Way Transformer로 이미지와 프롬프트 간 양방향 정보 교환
  3. 이미지 특징을 업스케일 (4배 확대)
  4. 마스크 토큰 × 업스케일된 특징 → 마스크 예측
  5. IoU 토큰 → 마스크 품질 점수 예측

■ Multi-mask 출력:
  - 하나의 프롬프트에 대해 최대 4개 마스크 후보 생성
    [0]: 단일 마스크 (기본)
    [1,2,3]: 다중 마스크 (모호한 경우, 예: 전체/부분)
  - multimask_output=True: [1,2,3] 반환 (IoU 가장 높은 것 선택)
  - multimask_output=False: [0]만 반환

■ Hypernetwork MLP:
  마스크 토큰에서 "동적 필터"를 생성하여 특징맵과 내적(dot product)
  → 각 픽셀이 물체에 속하는지 판단하는 마스크 생성
  비유: 토큰이 "이런 패턴을 찾아라"라는 필터를 만들고,
       그 필터로 이미지 전체를 스캔하여 마스크 생성
"""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from hvs.models.model_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    """
    마스크 디코더 — 이미지 특징 + 프롬프트 → 세그멘테이션 마스크

    Args:
        transformer_dim: Transformer 차원 (보통 256)
        transformer: TwoWayTransformer 인스턴스
        num_multimask_outputs: 다중 마스크 후보 수 (기본 3)
        iou_head_depth: IoU 예측 MLP 깊이
        iou_head_hidden_dim: IoU 예측 MLP 은닉 차원
        use_high_res_features: 고해상도 특징맵 사용 여부
        pred_obj_scores: 객체 존재 여부 예측 (비디오 추적용)
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # ---- 출력 토큰 ----
        # IoU 토큰: 마스크 품질 점수 예측에 사용
        self.iou_token = nn.Embedding(1, transformer_dim)
        # 마스크 토큰: 각 마스크 후보의 동적 필터 생성에 사용
        self.num_mask_tokens = num_multimask_outputs + 1  # +1: 단일 마스크용
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 객체 존재 점수 토큰 (비디오 추적: 물체가 사라졌는지 판단)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        # ---- 업스케일링 ----
        # 이미지 특징을 4배 확대하여 마스크 해상도 복원
        # (B, C, H, W) → (B, C/8, 4H, 4W)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        # 고해상도 특징맵 사용 시 추가 컨벌루션
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # ---- Hypernetwork MLP ----
        # 각 마스크 토큰에서 동적 필터(가중치)를 생성
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        # ---- IoU 예측 ----
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        # 객체 존재 점수 예측 헤드
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(
                    transformer_dim, transformer_dim, 1, 3
                )

        # 동적 다중마스크 안정성 폴백
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        마스크 예측

        Args:
            image_embeddings: (B, C, H, W) 이미지 특징
            image_pe: (1, C, H, W) 위치 인코딩
            sparse_prompt_embeddings: (B, N, C) 점/박스 임베딩
            dense_prompt_embeddings: (B, C, H, W) 마스크 임베딩
            multimask_output: True이면 3개 마스크, False이면 1개
            repeat_image: 이미지를 배치 차원에서 반복할지
            high_res_features: 고해상도 특징맵 리스트

        Returns:
            masks: (B, M, 4H, 4W) 마스크 logits
            iou_pred: (B, M) IoU 예측 점수
            sam_tokens_out: (B, M', C) SAM 출력 토큰
            object_score_logits: (B, 1) 객체 존재 점수
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # 다중 마스크 또는 단일 마스크 선택
        if multimask_output:
            masks = masks[:, 1:, :, :]     # 인덱스 1,2,3 (3개)
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]    # 인덱스 0 (1개)
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """내부 마스크 예측 로직"""
        # ---- 출력 토큰 준비 ----
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight,
                 self.mask_tokens.weight],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # [출력토큰 | 프롬프트토큰] 결합
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 이미지 임베딩 배치 처리
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # ---- Two-Way Transformer ----
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

        # ---- 업스케일 + 마스크 생성 ----
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Hypernetwork: 마스크 토큰 → 동적 필터 → 마스크
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        # 동적 필터와 특징맵의 내적 → 마스크
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # ---- IoU 예측 ----
        iou_pred = self.iou_prediction_head(iou_token_out)

        # ---- 객체 존재 점수 ----
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """마스크 안정성 점수 계산"""
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """안정성이 낮으면 다중 마스크에서 최선 선택"""
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds].unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds].unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits, best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores, best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
