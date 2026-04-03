"""
이미지 예측기 (Image Predictor)

■ 역할:
  이미지 1장에 대해 프롬프트(점/박스)를 주면 세그멘테이션 마스크를 예측합니다.
  산업 현장에서 가장 자주 사용할 추론 인터페이스.

■ 사용법:
  predictor = ImagePredictor.from_pretrained("tiny", checkpoint_path="...")
  predictor.set_image(image)
  masks, scores, logits = predictor.predict(
      point_coords=np.array([[100, 200]]),
      point_labels=np.array([1]),
  )

■ 최적화 고려 (30FPS 목표):
  - set_image()는 한 번만 호출 (이미지 인코딩은 무거움)
  - predict()는 여러 번 호출 가능 (프롬프트 변경 시 빠르게 재예측)
  - 이미지 인코딩 결과를 캐시하여 재사용

■ Windows 배포 고려:
  - PIL Image, numpy 배열, 파일 경로 모두 지원
  - GPU/CPU 자동 감지
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hvs.models.build import build_sam2_image_model
from hvs.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


class ImagePredictor:
    """
    이미지 세그멘테이션 예측기

    ■ 2단계 예측 패턴:
      Step 1: set_image() → 이미지 인코딩 (1회)
      Step 2: predict() → 프롬프트로 마스크 예측 (반복 가능)

    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large")
        image_size: 입력 이미지 크기 (기본 1024)
        device: 추론 디바이스
        checkpoint_path: 체크포인트 경로
        init_mode: 초기화 모드 ("finetune", "scratch")
        mask_threshold: 마스크 이진화 임계값 (기본 0.0)
    """

    def __init__(
        self,
        model_size: str = "tiny",
        image_size: int = 1024,
        device: str = None,
        checkpoint_path: str = None,
        init_mode: str = "finetune",
        mask_threshold: float = 0.0,
    ):
        # 디바이스 자동 감지
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.mask_threshold = mask_threshold

        # 모델 빌드
        self.model_parts = build_sam2_image_model(model_size, image_size)
        self.ie = self.model_parts["image_encoder"].to(self.device).eval()
        self.pe = self.model_parts["prompt_encoder"].to(self.device).eval()
        self.md = self.model_parts["mask_decoder"].to(self.device).eval()

        # 체크포인트 로드
        if checkpoint_path and init_mode != "scratch":
            result = load_checkpoint(
                self.model_parts, checkpoint_path,
                mode=init_mode, strict=False,
            )
            logger.info(f"Checkpoint loaded: {result['loaded_keys']} keys")

        # 이미지 정규화 파라미터 (ImageNet)
        self.pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53], device=self.device
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            [58.395, 57.12, 57.375], device=self.device
        ).view(1, 3, 1, 1)

        # 캐시 상태
        self._is_image_set = False
        self._features = None
        self._orig_hw = None

    @classmethod
    def from_pretrained(
        cls,
        model_size: str = "tiny",
        checkpoint_path: str = None,
        **kwargs,
    ) -> "ImagePredictor":
        """사전학습된 모델로 예측기 생성"""
        if checkpoint_path is None:
            from hvs.utils.checkpoint import download_checkpoint
            checkpoint_path = download_checkpoint(model_size)
        return cls(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            init_mode="finetune",
            **kwargs,
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리: RGB → 텐서 → 리사이즈 → 정규화

        Args:
            image: (H, W, 3) uint8 RGB 이미지
        Returns:
            (1, 3, image_size, image_size) 정규화된 텐서
        """
        # numpy → tensor
        img = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
        img = img.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 리사이즈
        img = F.interpolate(
            img, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        # 정규화
        img = (img - self.pixel_mean) / self.pixel_std
        return img

    @torch.no_grad()
    def set_image(self, image: Union[np.ndarray, Image.Image, str]) -> None:
        """
        이미지 인코딩 (1회 호출)

        이미지를 인코딩하여 특징을 캐시합니다.
        이후 predict()에서 프롬프트만 변경하며 빠르게 재예측 가능.

        Args:
            image: numpy array (H,W,3), PIL Image, 또는 파일 경로
        """
        self.reset()

        # 입력 형식 변환
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        assert isinstance(image, np.ndarray) and image.ndim == 3
        self._orig_hw = image.shape[:2]

        # 이미지 인코딩
        input_image = self._preprocess_image(image)
        enc_out = self.ie(input_image)
        fpn = enc_out["backbone_fpn"]

        self._features = {
            "image_embed": fpn[-1],  # 최저해상도 (backbone features)
            "high_res_feats": [
                self.md.conv_s0(fpn[0]),
                self.md.conv_s1(fpn[1]),
            ],
        }
        self._is_image_set = True

    def _transform_coords(
        self, coords: np.ndarray, orig_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """원본 좌표를 모델 입력 크기로 변환"""
        coords = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        # (x, y)를 원본 해상도에서 모델 해상도로 스케일링
        orig_h, orig_w = orig_hw
        coords[..., 0] = coords[..., 0] * self.image_size / orig_w
        coords[..., 1] = coords[..., 1] * self.image_size / orig_h
        return coords

    @torch.no_grad()
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        프롬프트로 마스크 예측

        Args:
            point_coords: (N, 2) 점 좌표 [x, y] (원본 이미지 기준)
            point_labels: (N,) 레이블 (1=전경, 0=배경)
            box: (4,) 박스 [x1, y1, x2, y2] (원본 이미지 기준)
            mask_input: (1, H, W) 이전 마스크 로짓 (반복 예측용)
            multimask_output: True면 3개 마스크, False면 1개
            return_logits: True면 로짓 반환 (이진화 안 함)

        Returns:
            masks: (C, H, W) 이진 마스크 (원본 크기)
            iou_predictions: (C,) IoU 점수
            low_res_masks: (C, h, w) 저해상도 마스크 로짓
        """
        if not self._is_image_set:
            raise RuntimeError(
                "set_image()를 먼저 호출해야 합니다."
            )

        # 프롬프트 준비
        concat_points = None
        if point_coords is not None:
            assert point_labels is not None, "point_labels is required"
            coords = self._transform_coords(point_coords, self._orig_hw)
            labels = torch.as_tensor(
                point_labels, dtype=torch.int32, device=self.device
            )
            if coords.ndim == 2:
                coords = coords.unsqueeze(0)  # (1, N, 2)
                labels = labels.unsqueeze(0)  # (1, N)
            concat_points = (coords, labels)

        # 박스 → 점으로 변환 (2개 모서리점)
        if box is not None:
            box_coords = self._transform_coords(
                box.reshape(-1, 2, 2), self._orig_hw
            )
            box_labels = torch.tensor(
                [[2, 3]], dtype=torch.int32, device=self.device
            ).expand(box_coords.shape[0], -1)
            if concat_points is not None:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )
            else:
                concat_points = (box_coords, box_labels)

        # 마스크 입력
        mask_input_tensor = None
        if mask_input is not None:
            mask_input_tensor = torch.as_tensor(
                mask_input, dtype=torch.float32, device=self.device
            )
            if mask_input_tensor.ndim == 3:
                mask_input_tensor = mask_input_tensor.unsqueeze(0)

        # Prompt Encoding
        sparse, dense = self.pe(
            points=concat_points,
            boxes=None,
            masks=mask_input_tensor,
        )
        image_pe = self.pe.get_dense_pe()

        # Mask Decoding
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )
        low_res_masks, iou_pred, _, _ = self.md(
            image_embeddings=self._features["image_embed"],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=self._features["high_res_feats"],
        )

        # 원본 해상도로 업스케일
        masks = F.interpolate(
            low_res_masks, size=self._orig_hw,
            mode="bilinear", align_corners=False,
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        if not return_logits:
            masks = masks > self.mask_threshold

        # numpy로 변환
        masks_np = masks.squeeze(0).float().cpu().numpy()
        iou_np = iou_pred.squeeze(0).float().cpu().numpy()
        low_res_np = low_res_masks.squeeze(0).float().cpu().numpy()

        return masks_np, iou_np, low_res_np

    def reset(self) -> None:
        """예측기 상태 초기화"""
        self._is_image_set = False
        self._features = None
        self._orig_hw = None

    @property
    def model_info(self) -> dict:
        """모델 정보 반환"""
        total = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.ie, self.pe, self.md]
        )
        return {
            "image_size": self.image_size,
            "device": str(self.device),
            "total_params": total,
        }
