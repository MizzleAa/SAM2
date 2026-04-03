"""
SAM2Base — 비디오 추적의 핵심 베이스 클래스

■ 역할:
  SAM2 모델의 모든 핵심 구성 요소를 하나로 통합합니다.
  Facebook SAM2의 SAM2Base를 HVS 아키텍처에 맞게 재구현한 것입니다.

■ 핵심 구성 요소:
  1. Image Encoder (Hiera + FpnNeck)
  2. Memory Attention (Self-Attn + Cross-Attn with RoPE)
  3. Memory Encoder (MaskDownSampler + Fuser)
  4. SAM Heads (PromptEncoder + MaskDecoder)

■ 비디오 추적 고유 파라미터 (체크포인트에서 로드):
  - maskmem_tpos_enc: 시간 위치 인코딩 (메모리 프레임의 시간 거리)
  - no_mem_embed: 메모리 없음 토큰 (첫 프레임용)
  - obj_ptr_proj: 객체 포인터 프로젝션 (SAM 출력 → 포인터)
  - no_obj_ptr: 객체 없음 포인터
  - mask_downsample: 마스크 → 포인터 변환용 다운샘플러

■ Facebook SAM2와의 차이:
  - Facebook은 Hydra YAML 기반 생성 → HVS는 Python 팩토리 패턴
  - Facebook의 SAM2Base.forward()는 NotImplementedError
    → HVS에서는 track_step()이 핵심 추론 진입점
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from hvs.models.model_utils import MLP
from hvs.models.sam2_utils import select_closest_cond_frames, get_1d_sine_pe

# 객체가 없는 프레임의 마스크 점수 자리표시자
NO_OBJ_SCORE = -1024.0


class SAM2Base(nn.Module):
    """
    SAM2 비디오 추적 베이스 모델

    ■ 체크포인트에서 로드되는 학습 파라미터:
      maskmem_tpos_enc:  [num_maskmem, 1, 1, mem_dim]  시간 위치 인코딩
      no_mem_embed:      [1, 1, hidden_dim]            메모리 없음 임베딩
      no_mem_pos_enc:    [1, 1, hidden_dim]            메모리 없음 위치 인코딩
      obj_ptr_proj:      MLP(256→256, 3-layers)        객체 포인터 프로젝션
      no_obj_ptr:        [1, hidden_dim]               객체 없음 포인터
      obj_ptr_tpos_proj: Linear(256→64)                시간 위치 프로젝션
      mask_downsample:   Conv2d(1,1,k=4,s=4)           마스크 다운샘플

    Args:
        image_encoder: ImageEncoder 인스턴스
        memory_attention: MemoryAttention 인스턴스
        memory_encoder: MemoryEncoder 인스턴스
        num_maskmem: 메모리 뱅크 크기 (기본 7 = 1 조건 + 6 히스토리)
        image_size: 입력 이미지 크기 (기본 1024)
        backbone_stride: 백본 출력 stride (기본 16)

        # 메모리 인코딩 설정
        sigmoid_scale_for_mem_enc: 메모리 인코딩 sigmoid 스케일 (기본 20.0)
        sigmoid_bias_for_mem_enc: 메모리 인코딩 sigmoid 바이어스 (기본 -10.0)
        binarize_mask_from_pts_for_mem_enc: 클릭 프레임 마스크 이진화 여부

        # Object Pointer 설정
        use_obj_ptrs_in_encoder: 인코더에서 Object Pointer 사용 (핵심!)
        max_obj_ptrs_in_encoder: 최대 포인터 수
        add_tpos_enc_to_obj_ptrs: 포인터에 시간 PE 추가
        proj_tpos_enc_in_obj_ptrs: 시간 PE 선형 프로젝션 적용
        use_signed_tpos_enc_to_obj_ptrs: 부호 있는 시간 PE 사용

        # SAM 디코더 설정
        use_high_res_features_in_sam: 고해상도 FPN 특징 사용
        multimask_output_in_sam: 멀티마스크 출력 (3개 후보)
        pred_obj_scores: 객체 존재 여부 예측
    """

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        # ── 기본 설정 ──
        num_maskmem=7,
        image_size=1024,
        backbone_stride=16,
        # ── 메모리 인코딩 ──
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        binarize_mask_from_pts_for_mem_enc=False,
        non_overlap_masks_for_mem_enc=False,
        # ── 첫 프레임 처리 ──
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        # ── Object Pointer ──
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        # ── SAM 디코더 ──
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        iou_prediction_use_sigmoid=True,
        # ── 고급 설정 ──
        max_cond_frames_in_attn=-1,
        memory_temporal_stride_for_eval=1,
        # ── 객체 존재 예측 ──
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        soft_no_obj_ptr=False,
        use_mlp_for_obj_ptr_proj=True,
        # ── 공간 no-object 임베딩 ──
        no_obj_embed_spatial=True,
        # ── MaskDecoder 추가 인자 ──
        sam_mask_decoder_extra_args=None,
    ):
        super().__init__()

        # ═══ Part 1: Image Encoder ═══
        self.image_encoder = image_encoder
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1

        # ═══ Part 2: Object Pointer 설정 ═══
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # 마스크 → stride 4로 다운샘플 (포인터 생성용)
            self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # ═══ Part 3: Memory Attention ═══
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model  # 256

        # ═══ Part 4: Memory Encoder ═══
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim  # 기본 256
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # 메모리 채널 압축이 있으면 (보통 256 → 64)
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]

        # ═══ Part 5: 시간 위치 인코딩 (학습 파라미터) ═══
        self.num_maskmem = num_maskmem
        # maskmem_tpos_enc: 체크포인트에서 [7, 1, 1, 64] shape으로 로드됨
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # 첫 프레임용 메모리 없음 토큰 (더미)
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed

        # ═══ Part 6: 메모리 인코딩 설정 ═══
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # ═══ Part 7: 객체 존재 예측 ═══
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)

        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        # ═══ Part 8: SAM Heads (PromptEncoder + MaskDecoder) ═══
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self._build_sam_heads()

        self.max_cond_frames_in_attn = max_cond_frames_in_attn

    # ─────────────────────────────────────────────────────────
    # SAM Heads 생성
    # ─────────────────────────────────────────────────────────

    def _build_sam_heads(self):
        """
        SAM 스타일 PromptEncoder + MaskDecoder 생성.
        체크포인트와 호환되는 키 이름을 사용합니다.

        ■ 키 이름 규칙:
          - sam_prompt_encoder.* → Facebook 체크포인트에서 직접 매핑
          - sam_mask_decoder.*   → Facebook 체크포인트에서 직접 매핑
        """
        from hvs.models.head.prompt_encoder import PromptEncoder
        from hvs.models.head.mask_decoder import MaskDecoder
        from hvs.models.head.transformer import TwoWayTransformer

        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # PromptEncoder (체크포인트 키: sam_prompt_encoder.*)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

        # MaskDecoder (체크포인트 키: sam_mask_decoder.*)
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        # Object Pointer 프로젝션
        if self.use_obj_ptrs_in_encoder:
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
            else:
                self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.obj_ptr_proj = nn.Identity()

        # 시간 PE 프로젝션 (Object Pointer용)
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = nn.Identity()

    # ─────────────────────────────────────────────────────────
    # SAM Heads Forward
    # ─────────────────────────────────────────────────────────

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        SAM PromptEncoder + MaskDecoder 순전파

        ■ 입력:
          backbone_features: (B, C, H, W) — 메모리 융합된 이미지 특징
          point_inputs: {"point_coords": (B,P,2), "point_labels": (B,P)}
          mask_inputs: (B, 1, H*16, W*16) — 이전 마스크 logits
          high_res_features: 고해상도 FPN 특징 리스트

        ■ 출력:
          low_res_multimasks, high_res_multimasks, ious,
          low_res_masks, high_res_masks,
          obj_ptr, object_score_logits
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim

        # a) Point 프롬프트 처리
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            # 포인트 없으면 더미 패딩 (label=-1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Mask 프롬프트 처리
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        # c) PromptEncoder → sparse + dense 임베딩
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )

        # d) MaskDecoder → 마스크 logits + IoU + 출력 토큰
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # e) 객체 존재 여부 처리
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # f) 고해상도 마스크 생성 (업스케일)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # g) Best mask 선택
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # h) Object Pointer 추출
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    # ─────────────────────────────────────────────────────────
    # 마스크를 직접 출력으로 사용
    # ─────────────────────────────────────────────────────────

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        마스크 입력을 SAM 디코더 없이 직접 출력으로 사용합니다.
        use_mask_input_as_output_without_sam=True 일 때 사용됩니다.

        ■ 동작:
          이진 마스크를 ±10 logit으로 변환 → 직접 출력
          Object Pointer는 SAM decoder를 통해 별도 생성
        """
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )

        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    # ─────────────────────────────────────────────────────────
    # Image Encoder Forward
    # ─────────────────────────────────────────────────────────

    def forward_image(self, img_batch):
        """이미지 특징 추출 (backbone + FPN)"""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # 고해상도 FPN level 0, 1을 SAM decoder용으로 프로젝션
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """backbone 특징을 평탄화: (B,C,H,W) → (HW, B, C)"""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # NxCxHxW → HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    # ─────────────────────────────────────────────────────────
    # 메모리 융합 (핵심!)
    # ─────────────────────────────────────────────────────────

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """
        현재 프레임의 시각적 특징을 과거 메모리와 융합합니다.

        ■ 핵심 로직:
          1. 조건 프레임 메모리 수집 (select_closest_cond_frames)
          2. 비조건 프레임 메모리 수집 (최근 num_maskmem-1 프레임)
          3. 시간 위치 인코딩 적용 (maskmem_tpos_enc)
          4. Object Pointer 수집 + 시간 PE
          5. MemoryAttention 실행

        Args:
            frame_idx: 현재 프레임 인덱스
            is_init_cond_frame: 초기 조건 프레임인지
            current_vision_feats: 현재 프레임의 시각 특징
            output_dict: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            num_frames: 전체 프레임 수

        Returns:
            pix_feat_with_mem: (B, C, H, W) 메모리 융합된 특징
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        # 메모리 비활성화 시 (num_maskmem=0): 이미지 전용 SAM
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        # ── Step 1: 메모리 수집 ──
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []

            # 1a. 조건 프레임 메모리
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

            # 1b. 비조건 프레임 메모리 (최근 num_maskmem-1 프레임)
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            # 1c. 메모리 토큰 구성 (공간 + 시간 PE)
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # 시간 위치 인코딩 추가
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # 1d. Object Pointer 토큰 수집
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs

                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_obj_ptrs):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))

                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device).float()
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # 초기 조건 프레임: 메모리 없이 처리
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # ── Step 2: MemoryAttention 실행 ──
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # (HW, B, C) → (B, C, H, W)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    # ─────────────────────────────────────────────────────────
    # 메모리 인코딩
    # ─────────────────────────────────────────────────────────

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        현재 프레임의 예측 마스크를 메모리로 인코딩합니다.

        ■ 동작:
          1. 마스크 logits → sigmoid (또는 이진화)
          2. sigmoid_scale + sigmoid_bias 적용
          3. MemoryEncoder 실행 → maskmem_features
          4. no_obj_embed_spatial 추가 (객체 없는 프레임)
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )

        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    # ─────────────────────────────────────────────────────────
    # Track Step (단일 프레임 추적)
    # ─────────────────────────────────────────────────────────

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        """내부 추적 단계 — 메모리 융합 + SAM 디코딩"""
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        # 고해상도 특징 (SAM 디코더용)
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # 마스크 입력을 직접 출력으로 사용
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # 메모리 융합 → SAM 디코딩
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        """메모리 인코더 실행 → current_out에 maskmem_features 추가"""
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        """
        단일 프레임 추적 (공개 API)

        ■ 전체 파이프라인:
          1. _track_step() — 메모리 융합 + SAM 디코딩
          2. 결과에서 pred_masks, obj_ptr, object_score_logits 추출
          3. _encode_memory_in_output() — 새 메모리 생성
          4. current_out 반환

        Args:
            frame_idx: 현재 프레임
            is_init_cond_frame: 초기 조건 프레임인지
            current_vision_feats: 이미지 특징 리스트
            current_vision_pos_embeds: 위치 인코딩 리스트
            feat_sizes: 특징 크기 리스트
            point_inputs: 포인트 프롬프트
            mask_inputs: 마스크 프롬프트
            output_dict: 지금까지의 출력 딕셔너리
            num_frames: 전체 프레임 수
            run_mem_encoder: 메모리 인코더 실행 여부
            prev_sam_mask_logits: 이전 프레임의 마스크 logits

        Returns:
            current_out: {
                pred_masks, pred_masks_high_res, obj_ptr,
                object_score_logits, maskmem_features, maskmem_pos_enc
            }
        """
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx, is_init_cond_frame,
            current_vision_feats, current_vision_pos_embeds, feat_sizes,
            point_inputs, mask_inputs, output_dict,
            num_frames, track_in_reverse, prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        self._encode_memory_in_output(
            current_vision_feats, feat_sizes, point_inputs,
            run_mem_encoder, high_res_masks, object_score_logits, current_out,
        )

        return current_out

    # ─────────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "SAM2Base.forward()는 직접 호출할 수 없습니다. "
            "SAM2VideoPredictor의 propagate_in_video() 또는 "
            "ImagePredictor를 사용하세요."
        )

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """멀티마스크 출력 사용 여부 판단"""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """다중 객체 마스크의 비겹침 제약 적용"""
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
