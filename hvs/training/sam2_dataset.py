"""
SAM2 데이터셋 어댑터

■ 역할:
  COCO 형식의 세그멘테이션 데이터셋을 SAM2 학습에 맞는 형태로 변환합니다.

■ 지원 형식:
  images/
    img001.jpg
    img002.jpg
    ...
  annotations/
    instances.json  (COCO format)

■ SAM2 학습에 필요한 출력:
  - image: (3, H, W) 정규화된 이미지
  - mask: (1, H, W) 이진 마스크 (0 또는 1)
  - point_coords: (N, 2) 프롬프트 점 좌표
  - point_labels: (N,) 레이블 (1=전경, 0=배경)

■ 프롬프트 자동 생성:
  학습 시 정답 마스크에서 무작위로 전경/배경 점을 샘플링합니다.
  이것이 SAM의 학습 방식 — "이 점이 주어졌을 때 마스크를 예측하라"
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import pycocotools.mask as mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


class SAM2Dataset(Dataset):
    """
    COCO 형식 세그멘테이션 데이터를 SAM2 학습용으로 변환

    Args:
        image_dir: 이미지 디렉토리 경로
        annotation_file: COCO JSON 파일 경로 (None이면 합성 데이터 모드)
        image_size: 리사이즈 크기 (가로=세로 정사각형)
        num_points: 자동 생성할 프롬프트 점 수 (전경+배경)
        transform: 이미지 전처리 함수 (None이면 기본 정규화)
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        annotation_file: Optional[str] = None,
        image_size: int = 1024,
        num_points: int = 1,
        transform=None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.num_points = num_points
        self.transform = transform

        # 이미지 정규화 (ImageNet 통계)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        self.samples = []
        if annotation_file and os.path.exists(annotation_file):
            self._load_coco_annotations(annotation_file)

    def _load_coco_annotations(self, annotation_file: str):
        """COCO JSON 로드 및 파싱"""
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # 이미지 ID → 파일명 매핑
        images = {img["id"]: img for img in coco_data["images"]}

        # 이미지별 어노테이션 그룹화
        img_to_anns = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # 각 이미지-어노테이션 쌍을 샘플로 구성
        for img_id, anns in img_to_anns.items():
            img_info = images.get(img_id)
            if img_info is None:
                continue
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                self.samples.append({
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "annotation": ann,
                })

    def _decode_mask(self, ann: dict, height: int, width: int) -> np.ndarray:
        """어노테이션에서 이진 마스크 디코딩"""
        if "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                # 폴리곤 형식 -> 마스크
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    import cv2
                    cv2.fillPoly(mask, [pts], 1)
                return mask
            elif isinstance(seg, dict) and HAS_PYCOCOTOOLS:
                # RLE 형식
                return mask_util.decode(seg)
        elif "bbox" in ann:
            # bbox → 마스크 (fallback)
            mask = np.zeros((height, width), dtype=np.uint8)
            x, y, w, h = [int(v) for v in ann["bbox"]]
            mask[y:y+h, x:x+w] = 1
            return mask
        return np.zeros((height, width), dtype=np.uint8)

    def _sample_points(
        self, mask: np.ndarray, num_points: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """마스크에서 전경/배경 프롬프트 점 샘플링"""
        fg_coords = np.argwhere(mask > 0)  # (N, 2) [y, x]
        bg_coords = np.argwhere(mask == 0)

        coords_list = []
        labels_list = []

        # 전경 점
        num_fg = max(1, num_points // 2) if num_points > 1 else num_points
        if len(fg_coords) > 0:
            indices = np.random.choice(len(fg_coords), size=min(num_fg, len(fg_coords)), replace=True)
            for idx in indices:
                y, x = fg_coords[idx]
                coords_list.append([float(x), float(y)])  # [x, y] 형식
                labels_list.append(1)
        else:
            # 전경이 없으면 중앙 점
            coords_list.append([mask.shape[1] / 2, mask.shape[0] / 2])
            labels_list.append(1)

        # 배경 점 (선택적)
        num_bg = num_points - len(coords_list)
        if num_bg > 0 and len(bg_coords) > 0:
            indices = np.random.choice(len(bg_coords), size=min(num_bg, len(bg_coords)), replace=True)
            for idx in indices:
                y, x = bg_coords[idx]
                coords_list.append([float(x), float(y)])
                labels_list.append(0)

        return np.array(coords_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """이미지 전처리: 리사이즈 + 정규화"""
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def _preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """마스크 전처리: 리사이즈"""
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        return torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float()

    def __len__(self):
        return max(len(self.samples), 1)  # 최소 1 (합성 데이터 모드)

    def __getitem__(self, idx):
        if not self.samples:
            # 합성 데이터 fallback (어노테이션 없을 때)
            return self._generate_synthetic(idx)

        sample = self.samples[idx % len(self.samples)]
        image_path = os.path.join(self.image_dir, sample["file_name"])
        image = Image.open(image_path)

        mask = self._decode_mask(
            sample["annotation"],
            sample["height"],
            sample["width"],
        )

        # 리사이즈 비율 계산 (점 좌표 보정용)
        scale_x = self.image_size / sample["width"]
        scale_y = self.image_size / sample["height"]

        # 점 샘플링 (원본 해상도에서)
        coords, labels = self._sample_points(mask, self.num_points)

        # 점 좌표를 리사이즈된 이미지 스케일로 변환
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y

        # 전처리
        image_tensor = self._preprocess_image(image)
        mask_tensor = self._preprocess_mask(mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "point_coords": torch.from_numpy(coords),
            "point_labels": torch.from_numpy(labels),
        }

    def _generate_synthetic(self, idx):
        """합성 데이터 생성 (테스트/데모용)"""
        image = torch.randn(3, self.image_size, self.image_size)
        mask = torch.zeros(1, self.image_size, self.image_size)

        cx, cy = self.image_size // 2, self.image_size // 2
        radius = self.image_size // 6
        y, x = torch.meshgrid(
            torch.arange(self.image_size),
            torch.arange(self.image_size),
            indexing="ij",
        )
        circle = ((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2
        mask[0] = circle.float()

        coords = torch.tensor([[float(cx), float(cy)]])
        labels = torch.tensor([1], dtype=torch.int32)

        return {
            "image": image,
            "mask": mask,
            "point_coords": coords,
            "point_labels": labels,
        }
