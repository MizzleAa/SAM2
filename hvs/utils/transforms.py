"""
이미지 전처리 유틸리티

■ 함수 목록:
  1. resize_image()       — 이미지 리사이즈 (비율 유지 옵션)
  2. normalize_image()    — ImageNet 정규화
  3. preprocess_image()   — 통합 전처리 (리사이즈 + 정규화 + 텐서 변환)
  4. denormalize_image()  — 역정규화 (시각화용)
  5. pad_to_square()      — 정사각형 패딩
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Union


# ImageNet 통계 (SAM2 기본값)
PIXEL_MEAN = np.array([123.675, 116.28, 103.53])
PIXEL_STD = np.array([58.395, 57.12, 57.375])


def resize_image(
    image: np.ndarray,
    target_size: int,
    keep_aspect: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    이미지 리사이즈

    Args:
        image: (H, W, 3) uint8
        target_size: 목표 크기 (장변 기준)
        keep_aspect: 비율 유지 여부

    Returns:
        (resized_image, original_hw)
    """
    orig_h, orig_w = image.shape[:2]
    pil_img = Image.fromarray(image)

    if keep_aspect:
        scale = target_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
    else:
        new_h = new_w = target_size

    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(resized), (orig_h, orig_w)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    ImageNet 정규화

    Args:
        image: (H, W, 3) float32 [0, 255]
    Returns:
        (H, W, 3) float32 정규화됨
    """
    return (image.astype(np.float32) - PIXEL_MEAN) / PIXEL_STD


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    역정규화 (시각화용)

    Args:
        image: (H, W, 3) float32 정규화됨
    Returns:
        (H, W, 3) uint8 [0, 255]
    """
    img = image * PIXEL_STD + PIXEL_MEAN
    return np.clip(img, 0, 255).astype(np.uint8)


def pad_to_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    정사각형 패딩 (우측/하단에 패딩)

    Args:
        image: (H, W, 3) uint8
        pad_value: 패딩 값
    Returns:
        (S, S, 3) uint8 (S = max(H, W))
    """
    h, w = image.shape[:2]
    size = max(h, w)
    padded = np.full((size, size, 3), pad_value, dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def preprocess_image(
    image: Union[np.ndarray, str, Image.Image],
    target_size: int = 1024,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    통합 전처리: 로드 → 리사이즈 → 패딩 → 정규화 → 텐서

    Args:
        image: numpy/PIL/경로
        target_size: 목표 이미지 크기
    Returns:
        (tensor (1, 3, S, S), original_hw)
    """
    # 로드
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    orig_hw = image.shape[:2]

    # 리사이즈
    resized, _ = resize_image(image, target_size, keep_aspect=True)

    # 패딩
    padded = pad_to_square(resized, pad_value=0)

    # 정규화
    normalized = normalize_image(padded)

    # (H,W,3) → (1,3,H,W) 텐서
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0)

    # 최종 리사이즈 (정확히 target_size)
    if tensor.shape[-1] != target_size:
        tensor = F.interpolate(
            tensor, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )

    return tensor, orig_hw
