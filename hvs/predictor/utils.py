"""
예측 후처리 유틸리티

■ 함수 목록:
  1. postprocess_masks()       — 마스크 후처리 (이진화 + 작은 영역 제거)
  2. remove_small_regions()    — 작은 연결 영역 제거
  3. fill_holes()              — 마스크 구멍 채우기
  4. mask_to_bbox()            — 마스크 → 바운딩 박스
  5. mask_to_polygon()         — 마스크 → 폴리곤 좌표
  6. calculate_iou()           — IoU 계산
  7. select_best_mask()        — 최고 품질 마스크 선택
"""

import numpy as np
from typing import List, Optional, Tuple


def postprocess_masks(
    masks: np.ndarray,
    threshold: float = 0.0,
    min_area: int = 100,
    fill_holes_area: int = 50,
) -> np.ndarray:
    """
    마스크 후처리 파이프라인

    Args:
        masks: (N, H, W) logits
        threshold: 이진화 임계값
        min_area: 최소 연결 영역 면적
        fill_holes_area: 최소 구멍 면적
    Returns:
        (N, H, W) bool
    """
    result = masks > threshold

    for i in range(len(result)):
        if min_area > 0:
            result[i] = remove_small_regions(result[i], min_area)
        if fill_holes_area > 0:
            result[i] = fill_holes(result[i], fill_holes_area)

    return result


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    작은 연결 영역 제거

    Args:
        mask: (H, W) bool
        min_area: 최소 면적
    Returns:
        (H, W) bool
    """
    try:
        import cv2
        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        result = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            region = labels == label_id
            if region.sum() >= min_area:
                result[region] = True
        return result
    except ImportError:
        return mask


def fill_holes(mask: np.ndarray, min_hole_area: int) -> np.ndarray:
    """
    마스크의 작은 구멍 채우기

    Args:
        mask: (H, W) bool
        min_hole_area: 최소 구멍 면적
    Returns:
        (H, W) bool
    """
    try:
        import cv2
        inverted = (~mask).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(inverted)
        result = mask.copy()
        for label_id in range(1, num_labels):
            hole = labels == label_id
            if hole.sum() < min_hole_area:
                result[hole] = True
        return result
    except ImportError:
        return mask


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    마스크 → 바운딩 박스 [x1, y1, x2, y2]

    Args:
        mask: (H, W) bool
    Returns:
        (4,) float [x1, y1, x2, y2] 또는 [0,0,0,0]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.array([0, 0, 0, 0], dtype=np.float32)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2 + 1, y2 + 1], dtype=np.float32)


def mask_to_polygon(mask: np.ndarray, tolerance: float = 2.0) -> List[np.ndarray]:
    """
    마스크 → 폴리곤 좌표 (컨투어)

    Args:
        mask: (H, W) bool
        tolerance: 단순화 허용치
    Returns:
        list of (N, 2) arrays
    """
    try:
        import cv2
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for c in contours:
            if tolerance > 0:
                c = cv2.approxPolyDP(c, tolerance, True)
            if len(c) >= 3:
                polygons.append(c.squeeze(1).astype(np.float32))
        return polygons
    except ImportError:
        return []


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    두 마스크 간 IoU 계산

    Args:
        mask1, mask2: (H, W) bool
    Returns:
        float [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    최고 품질 마스크 선택

    Args:
        masks: (N, H, W) bool
        scores: (N,) IoU 점수
        min_area: 최소 면적 필터
    Returns:
        (best_mask, best_score)
    """
    best_idx = -1
    best_score = -1.0

    for i in range(len(masks)):
        area = masks[i].sum()
        if area < min_area:
            continue
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    if best_idx < 0:
        return masks[0], float(scores[0])

    return masks[best_idx], float(best_score)
