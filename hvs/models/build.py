"""
SAM2 모델 빌드 팩토리

■ 역할:
  한 줄로 다양한 크기의 SAM2 모델을 생성합니다.
  모델 크기, 초기화 모드를 선택해서 바로 사용할 수 있습니다.

■ 모델 크기 변형 (Facebook SAM2.1 기준):
  ┌──────────┬───────────┬───────┬─────────────────┬──────────────────────────┐
  │  크기     │ 파라미터  │ 헤드  │ Stages          │ 채널                      │
  ├──────────┼───────────┼───────┼─────────────────┼──────────────────────────┤
  │ Tiny (T) │  ~39M     │  1    │ [1, 2, 7, 2]    │ 96→192→384→768           │
  │ Small (S)│  ~46M     │  1    │ [1, 2, 11, 2]   │ 96→192→384→768           │
  │ Base+ (B)│  ~81M     │  2    │ [2, 3, 16, 3]   │ 112→224→448→896          │
  │ Large (L)│  ~224M    │  2    │ [2, 6, 36, 4]   │ 144→288→576→1152         │
  └──────────┴───────────┴───────┴─────────────────┴──────────────────────────┘

■ 사용법:
  # 방법 1: 이미지 인코더만 빌드 (Phase 0 용)
  encoder = build_image_encoder("tiny")
  
  # 방법 2: 크기별 설정 딕셔너리 직접 접근
  config = MODEL_CONFIGS["tiny"]

■ 초기화 모드 (향후 build_sam2에서 사용):
  - "finetune":     Facebook 체크포인트 로드 → 파인튜닝
  - "scratch":      완전 랜덤 초기화 → 처음부터 학습
  - "backbone_only": ImageNet 백본만 로드 → Neck/Head 스크래치
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from hvs.models.backbone.hiera import Hiera
from hvs.models.backbone.position_encoding import PositionEmbeddingSine
from hvs.models.neck.image_encoder import ImageEncoder, FpnNeck
from hvs.models.head.transformer import Attention


# ═══════════════════════════════════════════════════════════════════
#  모델 크기별 설정 (Facebook SAM2.1 공식 YAML 기반)
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ─── Tiny (T) ─────────────────────────────────────────────
    # 가장 작은 모델. 개발/테스트용, RTX 5060에서 추론 가능.
    # 파라미터 ~39M. 30FPS 목표 달성에 가장 유리.
    "tiny": {
        "backbone": {
            "embed_dim": 96,
            "num_heads": 1,
            "stages": (1, 2, 7, 2),
            "global_att_blocks": (5, 7, 9),
            "window_pos_embed_bkg_spatial_size": (7, 7),
            "window_spec": (8, 4, 14, 7),
        },
        "neck": {
            "d_model": 256,
            "backbone_channel_list": [768, 384, 192, 96],
            "fpn_top_down_levels": [2, 3],
            "fpn_interp_model": "nearest",
        },
        "image_encoder": {
            "scalp": 1,
        },
    },

    # ─── Small (S) ────────────────────────────────────────────
    # Tiny보다 Stage 3 블록 수만 증가 (7→11).
    # 약간의 정확도 향상, 속도 차이 미미.
    "small": {
        "backbone": {
            "embed_dim": 96,
            "num_heads": 1,
            "stages": (1, 2, 11, 2),
            "global_att_blocks": (7, 11, 13),
            "window_pos_embed_bkg_spatial_size": (7, 7),
            "window_spec": (8, 4, 14, 7),
        },
        "neck": {
            "d_model": 256,
            "backbone_channel_list": [768, 384, 192, 96],
            "fpn_top_down_levels": [2, 3],
            "fpn_interp_model": "nearest",
        },
        "image_encoder": {
            "scalp": 1,
        },
    },

    # ─── Base+ (B+) ───────────────────────────────────────────
    # 중간 크기. 4090에서 학습에 적합.
    # embed_dim과 헤드 수 모두 증가 → 더 풍부한 특징 추출.
    "base_plus": {
        "backbone": {
            "embed_dim": 112,
            "num_heads": 2,
            "stages": (2, 3, 16, 3),
            "global_att_blocks": (12, 16, 20),
            "window_pos_embed_bkg_spatial_size": (14, 14),
            "window_spec": (8, 4, 14, 7),
        },
        "neck": {
            "d_model": 256,
            "backbone_channel_list": [896, 448, 224, 112],
            "fpn_top_down_levels": [2, 3],
            "fpn_interp_model": "nearest",
        },
        "image_encoder": {
            "scalp": 1,
        },
    },

    # ─── Large (L) ────────────────────────────────────────────
    # 가장 큰 모델. 최고 정확도, RTX 4090에서도 배치 크기 제한.
    # 실시간 추론(30FPS)에는 부적합할 수 있음.
    "large": {
        "backbone": {
            "embed_dim": 144,
            "num_heads": 2,
            "stages": (2, 6, 36, 4),
            "global_att_blocks": (23, 33, 43),
            "window_pos_embed_bkg_spatial_size": (14, 14),
            "window_spec": (8, 4, 16, 8),
        },
        "neck": {
            "d_model": 256,
            "backbone_channel_list": [1152, 576, 288, 144],
            "fpn_top_down_levels": [2, 3],
            "fpn_interp_model": "nearest",
        },
        "image_encoder": {
            "scalp": 1,
        },
    },
}

# 별칭 매핑 (유연한 입력 지원)
SIZE_ALIASES = {
    "t": "tiny", "tiny": "tiny", "hiera_t": "tiny",
    "s": "small", "small": "small", "hiera_s": "small",
    "b+": "base_plus", "b": "base_plus", "base+": "base_plus",
    "base_plus": "base_plus", "hiera_b+": "base_plus",
    "l": "large", "large": "large", "hiera_l": "large",
}


def _resolve_size(model_size: str) -> str:
    """모델 크기 문자열을 정규화"""
    key = model_size.lower().strip()
    if key not in SIZE_ALIASES:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Available: {available} (aliases: t, s, b+, l)"
        )
    return SIZE_ALIASES[key]


def build_backbone(model_size: str = "tiny") -> Hiera:
    """
    Backbone(Hiera)만 빌드합니다.
    
    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large" 또는 별칭)
    Returns:
        Hiera 인스턴스 (랜덤 초기화)
    """
    size_key = _resolve_size(model_size)
    cfg = MODEL_CONFIGS[size_key]["backbone"]
    return Hiera(**cfg)


def build_image_encoder(model_size: str = "tiny") -> ImageEncoder:
    """
    이미지 인코더 (Backbone + FPN Neck) 를 빌드합니다.
    
    ■ Phase 0 가능성 검증에서 사용:
      encoder = build_image_encoder("tiny")
      output = encoder(torch.randn(1, 3, 1024, 1024))
    
    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large" 또는 별칭)
    Returns:
        ImageEncoder 인스턴스 (랜덤 초기화)
    """
    size_key = _resolve_size(model_size)
    config = MODEL_CONFIGS[size_key]

    # 1. Backbone
    backbone = Hiera(**config["backbone"])

    # 2. Position Encoding
    neck_cfg = config["neck"]
    pos_enc = PositionEmbeddingSine(
        num_pos_feats=neck_cfg["d_model"],
        normalize=True,
        scale=None,
        temperature=10000,
    )

    # 3. FPN Neck
    neck = FpnNeck(
        position_encoding=pos_enc,
        d_model=neck_cfg["d_model"],
        backbone_channel_list=neck_cfg["backbone_channel_list"],
        fpn_top_down_levels=neck_cfg["fpn_top_down_levels"],
        fpn_interp_model=neck_cfg["fpn_interp_model"],
    )

    # 4. Image Encoder
    enc_cfg = config["image_encoder"]
    encoder = ImageEncoder(
        trunk=backbone,
        neck=neck,
        scalp=enc_cfg["scalp"],
    )

    return encoder


def get_model_info(model_size: str = "tiny") -> dict:
    """
    모델 크기별 정보 요약을 반환합니다.
    
    Returns:
        dict: stages, channels, params_estimate 등
    """
    size_key = _resolve_size(model_size)
    cfg = MODEL_CONFIGS[size_key]
    bb = cfg["backbone"]
    
    # 채널 수 계산 (dim_mul=2.0 기준)
    embed_dim = bb["embed_dim"]
    channels = [embed_dim]
    for _ in range(len(bb["stages"]) - 1):
        embed_dim = int(embed_dim * 2)
        channels.append(embed_dim)
    
    return {
        "size": size_key,
        "embed_dim": bb["embed_dim"],
        "num_heads": bb["num_heads"],
        "stages": bb["stages"],
        "total_blocks": sum(bb["stages"]),
        "channels": channels,
        "backbone_channel_list": cfg["neck"]["backbone_channel_list"],
        "d_model": cfg["neck"]["d_model"],
    }


def list_available_models() -> list:
    """사용 가능한 모델 크기 목록 반환"""
    return list(MODEL_CONFIGS.keys())


# ═══════════════════════════════════════════════════════════════════
#  Head (PromptEncoder + MaskDecoder) 빌드
# ═══════════════════════════════════════════════════════════════════

def build_prompt_encoder(
    d_model: int = 256,
    image_size: int = 1024,
    backbone_stride: int = 16,
):
    """
    프롬프트 인코더를 빌드합니다.

    Args:
        d_model: 임베딩 차원 (모든 크기에서 256)
        image_size: 입력 이미지 크기
        backbone_stride: 백본의 최종 stride
    """
    from hvs.models.head.prompt_encoder import PromptEncoder

    image_embedding_size = image_size // backbone_stride
    return PromptEncoder(
        embed_dim=d_model,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )


def build_mask_decoder(
    d_model: int = 256,
    use_high_res_features: bool = True,
    pred_obj_scores: bool = True,
    pred_obj_scores_mlp: bool = True,
    use_multimask_token_for_obj_ptr: bool = True,
    iou_prediction_use_sigmoid: bool = True,
):
    """
    마스크 디코더를 빌드합니다.

    Args:
        d_model: Transformer 차원
        use_high_res_features: 고해상도 특징맵 사용 여부
        pred_obj_scores: 객체 존재 점수 예측 여부
    """
    from hvs.models.head.transformer import TwoWayTransformer
    from hvs.models.head.mask_decoder import MaskDecoder

    return MaskDecoder(
        transformer_dim=d_model,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=d_model,
            mlp_dim=2048,
            num_heads=8,
        ),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=use_high_res_features,
        iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
        pred_obj_scores=pred_obj_scores,
        pred_obj_scores_mlp=pred_obj_scores_mlp,
        use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
    )


def build_sam2_image_model(
    model_size: str = "tiny",
    image_size: int = 1024,
):
    """
    이미지 전용 SAM2 모델을 빌드합니다 (Memory 제외).
    Phase 0 가능성 검증 및 이미지 세그멘테이션에 사용.

    ■ 구조:
      Image Encoder (Backbone + Neck) + Prompt Encoder + Mask Decoder

    ■ 사용법:
      model = build_sam2_image_model("tiny", image_size=512)
      encoder_out = model["image_encoder"](image)
      sparse, dense = model["prompt_encoder"](points, boxes, masks)
      masks, iou, tokens, obj_scores = model["mask_decoder"](...)

    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large")
        image_size: 입력 이미지 크기

    Returns:
        dict: {
            "image_encoder": ImageEncoder,
            "prompt_encoder": PromptEncoder,
            "mask_decoder": MaskDecoder,
            "config": {...},
        }
    """
    size_key = _resolve_size(model_size)
    config = MODEL_CONFIGS[size_key]
    d_model = config["neck"]["d_model"]

    image_encoder = build_image_encoder(model_size)
    prompt_encoder = build_prompt_encoder(
        d_model=d_model,
        image_size=image_size,
    )
    # Tiny 모델은 high_res_features 사용
    mask_decoder = build_mask_decoder(
        d_model=d_model,
        use_high_res_features=True,
    )

    return {
        "image_encoder": image_encoder,
        "prompt_encoder": prompt_encoder,
        "mask_decoder": mask_decoder,
        "config": {
            "model_size": size_key,
            "image_size": image_size,
            "d_model": d_model,
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  Memory 모듈 빌드
# ═══════════════════════════════════════════════════════════════════

def build_memory_encoder(d_model: int = 256, memory_dim: int = 64):
    """
    메모리 인코더 빌드 (마스크+특징 → 메모리 압축)

    Args:
        d_model: 이미지 특징 차원 (보통 256)
        memory_dim: 메모리 출력 차원 (보통 64, 압축)
    """
    from hvs.models.memory.memory_encoder import (
        MaskDownSampler, CXBlock, Fuser, MemoryEncoder,
    )

    mask_downsampler = MaskDownSampler(
        embed_dim=d_model, kernel_size=3, stride=2, padding=1, total_stride=16,
    )
    fuser = Fuser(
        layer=CXBlock(dim=d_model),
        num_layers=2,
    )
    pos_enc = PositionEmbeddingSine(
        num_pos_feats=memory_dim, normalize=True, scale=None, temperature=10000,
    )
    return MemoryEncoder(
        out_dim=memory_dim,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        position_encoding=pos_enc,
        in_dim=d_model,
    )


def build_memory_attention(d_model: int = 256, num_layers: int = 4, memory_dim: int = 64):
    """
    메모리 어텐션 빌드 (현재 프레임 + 과거 메모리 → 시간적 문맥 반영)

    ■ RoPEAttention 사용:
      Facebook SAM2.1과 동일하게 Self-Attention과 Cross-Attention 모두
      Rotary Position Encoding을 사용합니다.

    Args:
        d_model: 특징 차원
        num_layers: 어텐션 레이어 수 (보통 4)
        memory_dim: 메모리 차원 (Cross-Attn의 K/V 입력 차원)
    """
    from hvs.models.memory.memory_attention import (
        MemoryAttentionLayer, MemoryAttention,
    )
    from hvs.models.head.transformer import RoPEAttention

    # Self-Attention: RoPE (feat_sizes=64x64, image_size=1024, stride=16)
    self_attn = RoPEAttention(
        embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1,
        rope_theta=10000.0, feat_sizes=(64, 64),
    )
    # Cross-Attention: RoPE with key repeat (메모리가 더 긴 시퀀스)
    cross_attn = RoPEAttention(
        embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1,
        rope_theta=10000.0, rope_k_repeat=True, feat_sizes=(64, 64),
        kv_in_dim=memory_dim,
    )
    layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=cross_attn,
        d_model=d_model,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=self_attn,
    )
    return MemoryAttention(
        d_model=d_model,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=num_layers,
    )


def build_sam2_full_model(
    model_size: str = "tiny",
    image_size: int = 1024,
    memory_dim: int = 64,
    num_maskmem: int = 7,
):
    """
    전체 SAM2 모델 빌드 (이미지 + 비디오 추적 모드) — dict 기반

    ■ 구성:
      Image Encoder + Prompt Encoder + Mask Decoder
      + Memory Encoder + Memory Attention

    Args:
        model_size: 모델 크기
        image_size: 입력 이미지 크기
        memory_dim: 메모리 차원
        num_maskmem: 메모리 뱅크 프레임 수

    Returns:
        dict: 모든 모듈 + 설정 포함
    """
    size_key = _resolve_size(model_size)
    config = MODEL_CONFIGS[size_key]
    d_model = config["neck"]["d_model"]

    # 이미지 모델 (Encoder + Head)
    image_model = build_sam2_image_model(model_size, image_size)

    # Memory 모듈
    memory_encoder = build_memory_encoder(d_model=d_model, memory_dim=memory_dim)
    memory_attention = build_memory_attention(d_model=d_model, memory_dim=memory_dim)

    return {
        **image_model,
        "memory_encoder": memory_encoder,
        "memory_attention": memory_attention,
        "config": {
            "model_size": size_key,
            "image_size": image_size,
            "d_model": d_model,
            "memory_dim": memory_dim,
            "num_maskmem": num_maskmem,
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  SAM2Base 빌드 (통합 nn.Module)
# ═══════════════════════════════════════════════════════════════════

def build_sam2_base(
    model_size: str = "tiny",
    image_size: int = 1024,
    memory_dim: int = 64,
    num_maskmem: int = 7,
):
    """
    SAM2Base 통합 모델을 빌드합니다 (비디오 추적에 사용).

    ■ dict-based build와의 차이:
      dict: 각 모듈이 독립적 → 기존 load_checkpoint()으로 로드
      SAM2Base: 하나의 nn.Module → load_sam2_base_checkpoint()로 로드

    ■ SAM2Base 장점:
      - Facebook 체크포인트 키와 1:1 매핑 (maskmem_tpos_enc, obj_ptr_proj 등 포함)
      - track_step()으로 비디오 전파 가능
      - Object Pointer, 시간 PE 등 비디오 고유 기능 포함

    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large")
        image_size: 입력 이미지 크기
        memory_dim: 메모리 차원 (보통 64)
        num_maskmem: 메모리 뱅크 프레임 수

    Returns:
        SAM2Base 인스턴스
    """
    from hvs.models.sam2_base import SAM2Base

    size_key = _resolve_size(model_size)
    config = MODEL_CONFIGS[size_key]
    d_model = config["neck"]["d_model"]

    image_encoder = build_image_encoder(model_size)
    memory_encoder = build_memory_encoder(d_model=d_model, memory_dim=memory_dim)
    memory_attention = build_memory_attention(d_model=d_model, memory_dim=memory_dim)

    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=num_maskmem,
        image_size=image_size,
        backbone_stride=16,
        # SAM2.1 Tiny 기본값 적용
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        no_obj_embed_spatial=True,
    )

    return model

