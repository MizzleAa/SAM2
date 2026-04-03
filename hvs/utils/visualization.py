"""
결과 시각화 유틸리티

■ 함수 목록:
  1. overlay_mask()        — 이미지 위에 마스크 오버레이
  2. draw_points()         — 프롬프트 점 표시
  3. draw_boxes()          — 바운딩 박스 표시
  4. visualize_prediction() — 예측 결과 통합 시각화
  5. save_visualization()   — 파일 저장
  6. create_comparison()    — GT vs Pred 비교 이미지
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple


# ─── 색상 팔레트 ───
COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (0, 255, 128),   # Mint
    (255, 128, 128), # Light Red
]


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    이미지 위에 반투명 마스크 오버레이

    Args:
        image: (H, W, 3) uint8
        mask: (H, W) bool/float
        color: RGB 색상
        alpha: 투명도 (0=투명, 1=불투명)
    Returns:
        (H, W, 3) uint8
    """
    output = image.copy()
    mask_bool = mask.astype(bool)

    overlay = np.zeros_like(image)
    overlay[mask_bool] = color

    output[mask_bool] = (
        (1 - alpha) * output[mask_bool].astype(float) +
        alpha * overlay[mask_bool].astype(float)
    ).astype(np.uint8)

    # 마스크 경계선 표시
    from PIL import Image as PILImage
    pil_mask = PILImage.fromarray(mask_bool.astype(np.uint8) * 255)
    edges = pil_mask.filter(__import__('PIL.ImageFilter', fromlist=['FIND_EDGES']).FIND_EDGES)
    edge_arr = np.array(edges) > 128
    output[edge_arr] = color

    return output


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray = None,
    radius: int = 5,
) -> np.ndarray:
    """
    프롬프트 점 표시

    Args:
        image: (H, W, 3) uint8
        points: (N, 2) [x, y]
        labels: (N,) 1=전경(녹), 0=배경(빨)
        radius: 점 반지름
    Returns:
        (H, W, 3) uint8
    """
    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)

    if labels is None:
        labels = np.ones(len(points), dtype=int)

    for i, (x, y) in enumerate(points):
        color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color, outline=(255, 255, 255), width=1,
        )

    return np.array(pil_img)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
) -> np.ndarray:
    """
    바운딩 박스 표시

    Args:
        image: (H, W, 3)
        boxes: (N, 4) [x1, y1, x2, y2]
    """
    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)

    for box in boxes:
        draw.rectangle(box.tolist(), outline=color, width=width)

    return np.array(pil_img)


def visualize_prediction(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray = None,
    points: np.ndarray = None,
    labels: np.ndarray = None,
    boxes: np.ndarray = None,
) -> np.ndarray:
    """
    예측 결과 통합 시각화

    Args:
        image: (H, W, 3)
        masks: (N, H, W) 마스크
        scores: (N,) IoU 점수
        points: (M, 2) 프롬프트 점
        labels: (M,) 점 레이블
        boxes: (K, 4) 바운딩 박스
    """
    output = image.copy()

    # 마스크 오버레이 (점수 높은 순)
    if scores is not None:
        order = np.argsort(scores)[::-1]
    else:
        order = range(len(masks))

    for idx, i in enumerate(order):
        color = COLORS[idx % len(COLORS)]
        output = overlay_mask(output, masks[i], color=color, alpha=0.4)

    # 프롬프트 점
    if points is not None:
        output = draw_points(output, points, labels)

    # 바운딩 박스
    if boxes is not None:
        output = draw_boxes(output, boxes)

    return output


def create_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
) -> np.ndarray:
    """
    GT vs Prediction 비교 이미지 (나란히)

    Returns:
        (H, W*2+10, 3) uint8
    """
    gt_vis = overlay_mask(image, gt_mask, color=(0, 255, 0), alpha=0.5)
    pred_vis = overlay_mask(image, pred_mask, color=(0, 0, 255), alpha=0.5)

    h, w = image.shape[:2]
    gap = 10
    comparison = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 128
    comparison[:, :w] = gt_vis
    comparison[:, w + gap:] = pred_vis

    return comparison


def save_visualization(
    image: np.ndarray, path: str,
) -> str:
    """시각화 결과 저장"""
    Image.fromarray(image).save(path)
    return path
