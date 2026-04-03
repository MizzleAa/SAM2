"""
체크포인트 로더 (Checkpoint Loader)

■ 역할:
  기존 Facebook SAM2 사전학습 모델의 가중치를 hvs 프로젝트 모델에 로드합니다.
  3가지 초기화 모드를 지원:
    - "finetune":      Facebook 체크포인트 전체 로드 -> 파인튜닝
    - "scratch":       완전 랜덤 초기화 (체크포인트 미사용)
    - "backbone_only": 백본 가중치만 로드, 나머지 스크래치

■ Facebook 체크포인트 key 구조:
  image_encoder.trunk.*           -> hvs image_encoder.trunk.* (Hiera)
  image_encoder.neck.*            -> hvs image_encoder.neck.* (FpnNeck)
  sam_prompt_encoder.*            -> hvs prompt_encoder.*
  sam_mask_decoder.*              -> hvs mask_decoder.*
  memory_encoder.*                -> hvs memory_encoder.*
  memory_attention.*              -> hvs memory_attention.*

■ HuggingFace 모델 ID:
  "facebook/sam2.1-hiera-tiny"
  "facebook/sam2.1-hiera-small"
  "facebook/sam2.1-hiera-base-plus"
  "facebook/sam2.1-hiera-large"
"""

import logging
import os
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Facebook 체크포인트 직접 다운로드 URL
CHECKPOINT_URLS = {
    "tiny":      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "small":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "large":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

# HuggingFace 모델 ID
HF_MODEL_IDS = {
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}


def _remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Facebook 체크포인트의 key를 hvs 프로젝트 key로 매핑합니다.

    ■ 주요 매핑 (dict-based 로드 시):
      sam_prompt_encoder.* → prompt_encoder.*
      sam_mask_decoder.*   → mask_decoder.*
      그 외는 동일하게 유지

    ■ SAM2Base 로드 시:
      SAM2Base가 sam_prompt_encoder, sam_mask_decoder 등
      Facebook 키 이름을 직접 사용하므로 리매핑 불필요.
      load_sam2_base_checkpoint()를 사용하세요.
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # sam_prompt_encoder -> prompt_encoder
        if key.startswith("sam_prompt_encoder."):
            new_key = key.replace("sam_prompt_encoder.", "prompt_encoder.")

        # sam_mask_decoder -> mask_decoder
        elif key.startswith("sam_mask_decoder."):
            new_key = key.replace("sam_mask_decoder.", "mask_decoder.")

        new_state_dict[new_key] = value

    return new_state_dict


def download_checkpoint(
    model_size: str,
    save_dir: str = None,
    use_hf: bool = False,
) -> str:
    """
    Facebook 체크포인트를 다운로드합니다.

    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large")
        save_dir: 저장 디렉토리 (None이면 프로젝트 checkpoints/)
        use_hf: HuggingFace Hub 사용 여부

    Returns:
        str: 다운로드된 체크포인트 파일 경로
    """
    from hvs.models.build import _resolve_size
    size_key = _resolve_size(model_size)

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    if use_hf:
        try:
            from huggingface_hub import hf_hub_download
            model_id = HF_MODEL_IDS[size_key]
            ckpt_filename = f"sam2.1_hiera_{size_key}.pt"
            ckpt_path = hf_hub_download(
                repo_id=model_id, filename=ckpt_filename, local_dir=save_dir
            )
            return ckpt_path
        except ImportError:
            logger.warning("huggingface_hub not installed, falling back to direct URL")

    # 직접 URL 다운로드
    url = CHECKPOINT_URLS[size_key]
    filename = os.path.basename(url)
    ckpt_path = os.path.join(save_dir, filename)

    if os.path.exists(ckpt_path):
        logger.info(f"Checkpoint already exists: {ckpt_path}")
        return ckpt_path

    logger.info(f"Downloading checkpoint from {url}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, ckpt_path)
        logger.info(f"Saved to {ckpt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint: {e}")

    return ckpt_path


def load_checkpoint(
    model_parts: Dict[str, Any],
    checkpoint_path: str,
    mode: str = "finetune",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    체크포인트를 모델에 로드합니다.

    Args:
        model_parts: build_sam2_full_model()의 반환값 (모듈 dict)
        checkpoint_path: 체크포인트 파일 경로
        mode: 초기화 모드
            - "finetune":      전체 로드
            - "backbone_only": image_encoder.trunk만 로드
            - "scratch":       로드 안 함 (이 함수를 호출할 필요 없음)
        strict: 엄격 모드 (매칭 안 되는 key가 있으면 에러)

    Returns:
        dict: {
            "loaded_keys": 로드된 키 수,
            "missing_keys": 누락된 키 목록,
            "unexpected_keys": 예상하지 못한 키 목록,
            "mode": 사용된 모드,
        }
    """
    if mode == "scratch":
        logger.info("Scratch mode: skipping checkpoint loading")
        return {"loaded_keys": 0, "missing_keys": [], "unexpected_keys": [], "mode": mode}

    # 체크포인트 로드
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw:
        state_dict = raw["model"]
    else:
        state_dict = raw

    # Key 리매핑
    state_dict = _remap_checkpoint_keys(state_dict)

    total_loaded = 0
    all_missing = []
    all_unexpected = []

    # 모듈별 로드
    module_key_prefixes = {
        "image_encoder": "image_encoder.",
        "prompt_encoder": "prompt_encoder.",
        "mask_decoder": "mask_decoder.",
        "memory_encoder": "memory_encoder.",
        "memory_attention": "memory_attention.",
    }

    for module_name, prefix in module_key_prefixes.items():
        if module_name not in model_parts:
            continue

        module = model_parts[module_name]
        if not isinstance(module, nn.Module):
            continue

        # backbone_only 모드: image_encoder만 로드
        if mode == "backbone_only" and module_name != "image_encoder":
            continue

        # 해당 모듈의 가중치만 필터링
        module_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                module_state_dict[local_key] = value

        if not module_state_dict:
            logger.info(f"  {module_name}: No matching keys in checkpoint")
            continue

        # 로드 (non-strict by default)
        missing, unexpected = module.load_state_dict(module_state_dict, strict=strict)
        loaded = len(module_state_dict) - len(unexpected)
        total_loaded += loaded
        all_missing.extend([f"{prefix}{k}" for k in missing])
        all_unexpected.extend([f"{prefix}{k}" for k in unexpected])

        logger.info(
            f"  {module_name}: loaded {loaded} keys, "
            f"missing {len(missing)}, unexpected {len(unexpected)}"
        )

    result = {
        "loaded_keys": total_loaded,
        "missing_keys": all_missing,
        "unexpected_keys": all_unexpected,
        "mode": mode,
    }

    logger.info(
        f"Checkpoint loaded ({mode}): "
        f"{total_loaded} keys loaded, "
        f"{len(all_missing)} missing, "
        f"{len(all_unexpected)} unexpected"
    )

    return result


def load_sam2_base_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    SAM2Base 모델에 체크포인트를 직접 로드합니다.

    ■ SAM2Base가 Facebook 체크포인트와 동일한 키 이름을 사용하므로
      리매핑 없이 직접 load_state_dict() 호출.

    ■ 체크포인트 키 예시:
      image_encoder.trunk.*      → 백본 (Hiera)
      image_encoder.neck.*       → FPN Neck
      sam_prompt_encoder.*       → 프롬프트 인코더
      sam_mask_decoder.*         → 마스크 디코더
      memory_encoder.*           → 메모리 인코더
      memory_attention.*         → 메모리 어텐션 (4-layer RoPE)
      maskmem_tpos_enc           → 시간 PE [7,1,1,64]
      no_mem_embed               → [1,1,256]
      no_obj_ptr                 → [1,256]
      obj_ptr_proj.layers.*      → MLP(256→256, 3-layer)
      obj_ptr_tpos_proj.*        → Linear(256→64)
      mask_downsample.*          → Conv2d(1,1,k=4,s=4)
      no_obj_embed_spatial       → [1,64]

    Args:
        model: SAM2Base 인스턴스
        checkpoint_path: 체크포인트 파일 경로
        strict: 엄격 모드 (True이면 모든 키 매칭 필요)

    Returns:
        dict: loaded_keys, missing_keys, unexpected_keys
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw:
        state_dict = raw["model"]
    else:
        state_dict = raw

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    loaded_keys = len(state_dict) - len(unexpected)

    result = {
        "loaded_keys": loaded_keys,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mode": "sam2_base",
    }

    logger.info(
        f"SAM2Base checkpoint loaded: "
        f"{loaded_keys} keys loaded, "
        f"{len(missing)} missing, "
        f"{len(unexpected)} unexpected"
    )
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    return result


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    체크포인트 파일의 메타 정보를 반환합니다.

    Returns:
        dict: key 수, 모듈별 key 분포, 파일 크기 등
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw:
        state_dict = raw["model"]
    else:
        state_dict = raw

    # 모듈별 key 분포
    module_counts = {}
    for key in state_dict.keys():
        module = key.split(".")[0]
        module_counts[module] = module_counts.get(module, 0) + 1

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    return {
        "total_keys": len(state_dict),
        "module_distribution": module_counts,
        "file_size_mb": round(file_size_mb, 1),
    }
