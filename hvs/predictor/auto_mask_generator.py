"""
자동 마스크 생성기 (AutoMaskGenerator)

■ 역할:
  프롬프트 없이 이미지 전체에서 모든 객체의 마스크를 자동 생성합니다.
  이미지 위에 균등 분포 그리드 점을 배치하고, 각 점에 대해 마스크를 예측합니다.

■ 사용법:
  generator = AutoMaskGenerator(model_size="tiny", checkpoint_path="...")
  masks = generator.generate(image)
  # masks: [{"segmentation": np.array, "area": int, "bbox": [...], "predicted_iou": float}, ...]

■ 산업 결함 검출 시나리오:
  - 부품 표면을 스캔하여 모든 결함 자동 검출
  - 점검 이미지에서 이상 영역 자동 추출
  - 레이블링 보조 도구 (초기 마스크 → 인간 검수/수정)

■ 핵심 파라미터:
  - points_per_side: 그리드 밀도 (32 → 32×32=1024개 점)
  - pred_iou_thresh: 품질 필터 (0.8 이상만 유지)
  - stability_score_thresh: 안정성 필터 (0.95 이상)
  - box_nms_thresh: 중복 제거 IoU 임계값 (0.7)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hvs.predictor.image_predictor import ImagePredictor

logger = logging.getLogger(__name__)


def _build_point_grid(n_per_side: int) -> np.ndarray:
    """
    n×n 균등 분포 그리드 점 생성

    Args:
        n_per_side: 한 변 당 점 수
    Returns:
        (n*n, 2) 정규화 좌표 [0, 1]
    """
    offset = 1.0 / (2 * n_per_side)
    points_x = np.linspace(offset, 1 - offset, n_per_side)
    points_y = np.linspace(offset, 1 - offset, n_per_side)
    xx, yy = np.meshgrid(points_x, points_y)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return points.astype(np.float32)


def _calculate_stability_score(
    masks: torch.Tensor,
    mask_threshold: float = 0.0,
    threshold_offset: float = 1.0,
) -> torch.Tensor:
    """
    마스크 안정성 점수 계산

    threshold를 약간 바꿔도 마스크가 크게 변하지 않으면 안정적.
    """
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / (unions + 1e-6)


def _mask_to_box(mask: np.ndarray) -> np.ndarray:
    """이진 마스크 → bbox [x1, y1, x2, y2]"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.array([0, 0, 0, 0], dtype=np.float32)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2 + 1, y2 + 1], dtype=np.float32)


class AutoMaskGenerator:
    """
    이미지 전체 자동 마스크 생성기

    Args:
        model_size: 모델 크기
        image_size: 입력 이미지 크기
        device: 추론 디바이스
        checkpoint_path: 체크포인트 경로
        init_mode: 초기화 모드
        points_per_side: 그리드 밀도
        points_per_batch: 배치당 점 수
        pred_iou_thresh: IoU 품질 필터
        stability_score_thresh: 안정성 필터
        box_nms_thresh: NMS IoU 임계값
        min_mask_area: 최소 마스크 면적
    """

    def __init__(
        self,
        model_size: str = "tiny",
        image_size: int = 1024,
        device: str = None,
        checkpoint_path: str = None,
        init_mode: str = "finetune",
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.92,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        mask_threshold: float = 0.0,
        min_mask_area: int = 0,
        multimask_output: bool = True,
    ):
        self.predictor = ImagePredictor(
            model_size=model_size,
            image_size=image_size,
            device=device,
            checkpoint_path=checkpoint_path,
            init_mode=init_mode,
            mask_threshold=mask_threshold,
        )

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.multimask_output = multimask_output

        # 그리드 생성
        self.point_grid = _build_point_grid(points_per_side)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 모든 마스크 자동 생성

        Args:
            image: (H, W, 3) uint8 RGB 이미지

        Returns:
            list of dict: 각 마스크 정보
              - segmentation: (H, W) bool
              - area: int
              - bbox: [x, y, w, h]
              - predicted_iou: float
              - stability_score: float
              - point_coords: [[x, y]]
        """
        orig_h, orig_w = image.shape[:2]

        # 이미지 인코딩 (1회)
        self.predictor.set_image(image)

        # 그리드 점을 원본 이미지 좌표로 변환
        points = self.point_grid.copy()
        points[:, 0] *= orig_w
        points[:, 1] *= orig_h

        # 배치별 예측
        all_masks = []
        all_scores = []
        all_logits = []
        all_points = []

        for i in range(0, len(points), self.points_per_batch):
            batch_points = points[i:i + self.points_per_batch]
            batch_labels = np.ones(len(batch_points), dtype=np.int32)

            # 각 점을 개별로 예측 (멀티마스크)
            for j in range(len(batch_points)):
                pt = batch_points[j:j+1]
                lb = batch_labels[j:j+1]

                masks, scores, logits = self.predictor.predict(
                    point_coords=pt,
                    point_labels=lb,
                    multimask_output=self.multimask_output,
                    return_logits=True,
                )

                all_masks.append(masks)
                all_scores.append(scores)
                all_logits.append(logits)
                all_points.append(pt)

        if not all_masks:
            self.predictor.reset()
            return []

        # 모든 마스크 결합
        masks_cat = np.concatenate(all_masks, axis=0)       # (N*C, H, W) logits
        scores_cat = np.concatenate(all_scores, axis=0)     # (N*C,)
        points_rep = []
        for pt, m in zip(all_points, all_masks):
            points_rep.extend([pt[0]] * m.shape[0])
        points_arr = np.array(points_rep)

        # 필터링
        results = []
        for idx in range(len(masks_cat)):
            mask_logit = masks_cat[idx]
            score = float(scores_cat[idx])
            point = points_arr[idx]

            # IoU 필터
            if score < self.pred_iou_thresh:
                continue

            # 안정성 필터
            mask_t = torch.from_numpy(mask_logit).float().unsqueeze(0)
            stability = _calculate_stability_score(
                mask_t, self.mask_threshold, self.stability_score_offset
            ).item()
            if stability < self.stability_score_thresh:
                continue

            # 이진화
            binary_mask = mask_logit > self.mask_threshold
            area = int(binary_mask.sum())

            # 최소 면적 필터
            if area < self.min_mask_area:
                continue

            # bbox
            bbox = _mask_to_box(binary_mask)
            bbox_xywh = [
                float(bbox[0]), float(bbox[1]),
                float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
            ]

            results.append({
                "segmentation": binary_mask,
                "area": area,
                "bbox": bbox_xywh,
                "predicted_iou": score,
                "stability_score": stability,
                "point_coords": [[float(point[0]), float(point[1])]],
            })

        # NMS (중복 마스크 제거)
        if self.box_nms_thresh < 1.0 and len(results) > 0:
            results = self._nms(results)

        self.predictor.reset()
        return results

    def _nms(self, results: List[Dict]) -> List[Dict]:
        """간단한 박스 NMS"""
        if len(results) <= 1:
            return results

        boxes = np.array([
            [r["bbox"][0], r["bbox"][1],
             r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
            for r in results
        ])
        scores = np.array([r["predicted_iou"] for r in results])

        # IoU 계산 + Greedy NMS
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            ious = self._box_iou(boxes[i], boxes[order[1:]])
            remaining = np.where(ious < self.box_nms_thresh)[0]
            order = order[remaining + 1]

        return [results[i] for i in keep]

    @staticmethod
    def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """단일 박스와 여러 박스 간 IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)
