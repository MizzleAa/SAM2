"""
비디오 예측기 (Video Predictor) — SAM2Base 기반

■ 역할:
  프레임 시퀀스에 대해 객체 추적 + 세그멘테이션을 수행합니다.
  첫 프레임에 프롬프트(점/박스)를 주면, 이후 프레임에서 자동 추적.

■ Facebook SAM2VideoPredictor와의 동일점 (핵심):
  - SAM2Base.track_step()을 사용한 단일 프레임 추론
  - Object Pointer + 시간 PE 기반 메모리 전파
  - cond_frame_outputs / non_cond_frame_outputs 분리
  - select_closest_cond_frames 기반 메모리 선택

■ 사용법:
  predictor = VideoPredictor(model_size="tiny", checkpoint_path="...")
  state = predictor.init_state(video_frames)
  predictor.add_points(state, frame_idx=0, obj_id=1, points=[[100,200]], labels=[1])
  for idx, obj_ids, mask in predictor.propagate(state):
      save_mask(mask)

■ 산업 결함 검출에서의 활용:
  - 컨베이어 벨트 위 부품이 이동하는 영상에서 결함 추적
  - 첫 프레임에 결함 위치를 지정하면 이후 프레임에서 자동 추적
"""

import logging
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hvs.models.build import build_sam2_base
from hvs.models.sam2_base import NO_OBJ_SCORE
from hvs.utils.checkpoint import load_sam2_base_checkpoint

logger = logging.getLogger(__name__)


def concat_points(old_point_inputs, new_points, new_labels):
    """포인트 입력 결합 유틸리티"""
    if old_point_inputs is None:
        return {"point_coords": new_points, "point_labels": new_labels}
    return {
        "point_coords": torch.cat([old_point_inputs["point_coords"], new_points], dim=1),
        "point_labels": torch.cat([old_point_inputs["point_labels"], new_labels], dim=1),
    }


class VideoPredictor:
    """
    비디오 세그멘테이션 예측기 (SAM2Base 기반)

    ■ 3단계 사용 패턴:
      Step 1: init_state()   → 비디오 프레임 로드
      Step 2: add_points()   → 프레임에 프롬프트 추가
      Step 3: propagate()    → 전체 프레임으로 마스크 전파

    ■ Facebook SAM2VideoPredictor와의 차이:
      - Facebook: SAM2Base 상속 → Hydra YAML 생성
      - HVS: 래퍼 클래스 → build_sam2_base() 팩토리 + load_sam2_base_checkpoint()
      - API 동일: init_state, add_points, propagate

    Args:
        model_size: 모델 크기
        image_size: 입력 이미지 크기
        device: 추론 디바이스
        checkpoint_path: 체크포인트 경로
        num_maskmem: 메모리 뱅크 프레임 수
        memory_dim: 메모리 차원
    """

    def __init__(
        self,
        model_size: str = "tiny",
        image_size: int = 1024,
        device: str = None,
        checkpoint_path: str = None,
        init_mode: str = "finetune",
        num_maskmem: int = 7,
        memory_dim: int = 64,
        fill_hole_area: int = 0,
        non_overlap_masks: bool = False,
        use_fp16: bool = True,
        compile_model: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks

        # ── 성능 최적화 설정 ──
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.autocast_dtype = torch.float16 if self.use_fp16 else torch.float32

        # SAM2Base 통합 모델 빌드
        self.model = build_sam2_base(
            model_size=model_size,
            image_size=image_size,
            memory_dim=memory_dim,
            num_maskmem=num_maskmem,
        ).to(self.device).eval()

        # 체크포인트 로드
        if checkpoint_path and init_mode != "scratch":
            load_sam2_base_checkpoint(self.model, checkpoint_path, strict=False)

        # torch.compile 적용 (이미지 인코더 가속)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model.image_encoder = torch.compile(
                    self.model.image_encoder,
                    mode="max-autotune",
                    fullgraph=True,
                )
                logger.info("torch.compile 적용 완료 (image_encoder)")
            except Exception as e:
                logger.warning(f"torch.compile 실패, 기본 모드로 실행: {e}")

        # 이미지 정규화
        self.pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53], device=self.device
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            [58.395, 57.12, 57.375], device=self.device
        ).view(1, 3, 1, 1)

    # ─────────────────────────────────────────────────────────
    # 전처리
    # ─────────────────────────────────────────────────────────

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """단일 프레임 전처리: resize + normalize"""
        img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img = F.interpolate(
            img, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        img = (img - self.pixel_mean) / self.pixel_std
        return img

    # ─────────────────────────────────────────────────────────
    # State 관리
    # ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def init_state(
        self,
        video_frames: Union[List[np.ndarray], str],
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> dict:
        """
        비디오 프레임 로드 및 추적 상태 초기화

        ■ 성능 최적화:
          - 전체 프레임을 배치 단위로 사전 인코딩하여 캐시
          - propagate() 시 이미지 인코더 재실행 불필요 (캐시 히트 100%)
          - 논문 대비 속도 향상의 핵심

        Args:
            video_frames: 다음 중 하나:
              - 프레임 리스트: [(H,W,3) numpy]
              - 디렉토리 경로: 프레임 이미지 파일이 있는 폴더
              - 비디오 파일 경로: .avi, .mp4, .mkv 등
            offload_state_to_cpu: 상태를 CPU에 저장 (GPU 메모리 절약)
            async_loading_frames: True면 사전 인코딩 건너뜀 (lazy 모드)

        Returns:
            inference_state: 추적 상태 딕셔너리
        """
        if isinstance(video_frames, str):
            path = video_frames
            video_exts = {".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv"}
            ext = os.path.splitext(path)[1].lower()

            if os.path.isfile(path) and ext in video_exts:
                # 비디오 파일 로드
                import cv2
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"비디오 파일을 열 수 없습니다: {path}")
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                video_frames = frames
            elif os.path.isdir(path):
                # 이미지 디렉토리 로드
                img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                files = sorted([
                    f for f in os.listdir(path)
                    if os.path.splitext(f)[1].lower() in img_exts
                ])
                frames = []
                for f in files:
                    img = np.array(Image.open(os.path.join(path, f)).convert("RGB"))
                    frames.append(img)
                video_frames = frames
            else:
                raise FileNotFoundError(f"파일 또는 디렉토리를 찾을 수 없습니다: {path}")

        assert len(video_frames) > 0, "비디오 프레임이 비어있습니다"

        # 전처리된 텐서로 변환
        images = []
        for frame in video_frames:
            img = self._preprocess_frame(frame)
            images.append(img.squeeze(0))  # (3, H, W)

        orig_h, orig_w = video_frames[0].shape[:2]

        inference_state = {
            "images": images,
            "num_frames": len(images),
            "video_height": orig_h,
            "video_width": orig_w,
            "device": self.device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else self.device,
            # 객체 관리
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # 프레임별 입력
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            # 특징 캐시 (전체 프레임)
            "cached_features": {},
            "constants": {},
            # 객체별 출력 (Facebook 구조와 동일)
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }

        # ── 인코딩 모드 선택 ──
        # GPU 8GB에서는 전체 사전 인코딩이 CPU→GPU 전송 오버헤드로 오히려 느림
        # → 기본: lazy 모드 (propagate 중 프레임별 인코딩)
        if not async_loading_frames:
            self._precompute_all_features(inference_state)
        else:
            # lazy 모드: 첫 프레임만 웜업
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    @torch.inference_mode()
    def _precompute_all_features(self, inference_state):
        """
        전체 프레임의 이미지 특징을 사전 인코딩합니다.
        ■ GPU 메모리 절약: 특징을 CPU에 저장, 추적 시 GPU로 이동
        ■ 효과: propagate()에서 이미지 인코더 재실행 완전 제거
        """
        num_frames = inference_state["num_frames"]
        device = inference_state["device"]

        for frame_idx in range(num_frames):
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_fp16):
                backbone_out = self.model.forward_image(image)
            # CPU로 오프로드하여 GPU 메모리 절약
            cpu_backbone_out = {
                "backbone_fpn": [x.cpu() for x in backbone_out["backbone_fpn"]],
                "vision_pos_enc": [x.cpu() for x in backbone_out["vision_pos_enc"]],
            }
            inference_state["cached_features"][frame_idx] = (image.cpu(), cpu_backbone_out)

        # GPU 메모리 정리
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  Pre-encoded: {num_frames} frames (cached on CPU)")

    # ─────────────────────────────────────────────────────────
    # 객체 관리
    # ─────────────────────────────────────────────────────────

    def _obj_id_to_idx(self, inference_state, obj_id):
        """객체 ID → 인덱스 매핑 (새 객체 자동 등록)"""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # 새 객체 등록
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])

        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["frames_tracked_per_obj"][obj_idx] = {}
        return obj_idx

    def _get_obj_num(self, inference_state):
        return len(inference_state["obj_idx_to_id"])

    # ─────────────────────────────────────────────────────────
    # Image Feature 추출
    # ─────────────────────────────────────────────────────────

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        프레임의 이미지 특징 추출 (CPU 캐시 → GPU 전송 또는 즉시 인코딩)
        """
        device = inference_state["device"]
        cached = inference_state["cached_features"].get(frame_idx, None)

        if cached is not None:
            image, backbone_out = cached
            # CPU 캐시 → GPU 전송 (사전 인코딩된 경우)
            if image.device != device:
                image = image.to(device, non_blocking=True)
            if backbone_out["backbone_fpn"][0].device != device:
                backbone_out = {
                    "backbone_fpn": [x.to(device, non_blocking=True) for x in backbone_out["backbone_fpn"]],
                    "vision_pos_enc": [x.to(device, non_blocking=True) for x in backbone_out["vision_pos_enc"]],
                }
        else:
            # 캐시 미스: 즉시 인코딩
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_fp16):
                backbone_out = self.model.forward_image(image)

        # 배치 확장 (멀티 객체용)
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

        features = self.model._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    # ─────────────────────────────────────────────────────────
    # 포인트 / 마스크 추가
    # ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def add_points(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
    ) -> Tuple[int, list, np.ndarray]:
        """
        프레임에 프롬프트 점 추가 + 즉시 마스크 예측

        Args:
            state: init_state()의 반환값
            frame_idx: 프롬프트를 추가할 프레임 인덱스
            obj_id: 객체 ID
            points: (N, 2) [x, y] 좌표
            labels: (N,) 레이블 (1=전경, 0=배경)
            clear_old_points: 이전 포인트 입력 삭제
            normalize_coords: 좌표 정규화 (0~1 범위 → 이미지 크기)

        Returns:
            (frame_idx, obj_ids, video_res_masks)
        """
        obj_idx = self._obj_id_to_idx(state, obj_id)
        point_inputs_per_frame = state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if normalize_coords:
            video_H = state["video_height"]
            video_W = state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        points = points * self.image_size
        points = points.to(state["device"])
        labels = labels.to(state["device"])

        if not clear_old_points:
            old_point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            old_point_inputs = None
        point_inputs = concat_points(old_point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        # 초기 조건 프레임 여부 판단
        obj_frames_tracked = state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx].get("reverse", False)

        obj_output_dict = state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = state["temp_output_dict_per_obj"][obj_idx]
        storage_key = "cond_frame_outputs"

        # 이전 마스크 logits 가져오기
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and prev_out.get("pred_masks") is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].to(state["device"], non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        # 단일 프레임 추론
        current_out, pred_masks_gpu = self._run_single_frame_inference(
            inference_state=state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # 원본 해상도 마스크
        _, video_res_masks = self._get_orig_video_res_output(state, pred_masks_gpu)
        masks_np = video_res_masks.squeeze(0).cpu().numpy()

        return frame_idx, state["obj_ids"], masks_np

    @torch.inference_mode()
    def add_mask(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> Tuple[int, list, np.ndarray]:
        """프레임에 마스크 프롬프트 추가"""
        obj_idx = self._obj_id_to_idx(state, obj_id)
        mask_inputs_per_frame = state["mask_inputs_per_obj"][obj_idx]
        point_inputs_per_frame = state["point_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_inputs = mask[None, None].float().to(state["device"])

        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask_inputs = F.interpolate(
                mask_inputs, size=(self.image_size, self.image_size),
                align_corners=False, mode="bilinear", antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)

        obj_frames_tracked = state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx].get("reverse", False)

        obj_output_dict = state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = state["temp_output_dict_per_obj"][obj_idx]
        storage_key = "cond_frame_outputs"

        current_out, pred_masks_gpu = self._run_single_frame_inference(
            inference_state=state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        _, video_res_masks = self._get_orig_video_res_output(state, pred_masks_gpu)
        masks_np = video_res_masks.squeeze(0).cpu().numpy()
        return frame_idx, state["obj_ids"], masks_np

    # ─────────────────────────────────────────────────────────
    # 전파 (Propagate)
    # ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """전파 전 준비: temp_output을 output_dict로 통합 + 메모리 인코딩"""
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError("입력 프롬프트가 없습니다. add_points() 또는 add_mask()를 먼저 호출하세요.")

        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    if out.get("maskmem_features") is None:
                        # 메모리 인코더 실행
                        high_res_masks = F.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state, frame_idx, batch_size=1,
                            high_res_masks=high_res_masks,
                            object_score_logits=out.get("object_score_logits"),
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out
                obj_temp_output_dict[storage_key].clear()

            # 입력이 있는지 확인
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = inference_state["obj_idx_to_id"][obj_idx]
                raise RuntimeError(f"객체 {obj_id}에 입력이 없습니다.")

            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    @torch.inference_mode()
    def propagate(
        self,
        state: dict,
        start_frame: int = None,
        max_frame_num_to_track: int = None,
        reverse: bool = False,
    ):
        """
        마스크를 전체 프레임으로 전파 (Generator)

        ■ Facebook SAM2VideoPredictor.propagate_in_video()와 동일한 로직:
          1. preflight() — temp_output 통합 + 메모리 인코딩
          2. 프레임 순회 → _run_single_frame_inference()
          3. 조건 프레임은 skip, 비조건 프레임은 추론 + 메모리 생성

        Args:
            state: init_state()의 반환값
            start_frame: 시작 프레임 (None이면 자동)
            max_frame_num_to_track: 최대 추적 프레임 수
            reverse: 역방향 전파

        Yields:
            (frame_idx, obj_ids, masks_np)
        """
        self.propagate_in_video_preflight(state)

        obj_ids = state["obj_ids"]
        num_frames = state["num_frames"]
        batch_size = self._get_obj_num(state)

        if start_frame is None:
            start_frame = min(
                t
                for obj_out in state["output_dict_per_obj"].values()
                for t in obj_out["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        if reverse:
            end_frame = max(start_frame - max_frame_num_to_track, 0)
            processing_order = range(start_frame, end_frame - 1, -1) if start_frame > 0 else []
        else:
            end_frame = min(start_frame + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame, end_frame + 1)

        for frame_idx in processing_order:
            pred_masks_per_obj = [None] * batch_size

            for obj_idx in range(batch_size):
                obj_output_dict = state["output_dict_per_obj"][obj_idx]

                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    # 조건 프레임: 이미 예측된 결과 사용
                    current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        state["device"], non_blocking=True
                    )
                else:
                    # 비조건 프레임: 메모리 기반 추론
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out

                state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks

            # 모든 객체 마스크 합치기
            all_pred_masks = (
                torch.cat(pred_masks_per_obj, dim=0) if len(pred_masks_per_obj) > 1
                else pred_masks_per_obj[0]
            )
            _, video_res_masks = self._get_orig_video_res_output(state, all_pred_masks)
            masks_np = video_res_masks.cpu().numpy()  # (num_obj, 1, H, W)

            yield frame_idx, obj_ids, masks_np

    # ─────────────────────────────────────────────────────────
    # 내부 추론 메서드
    # ─────────────────────────────────────────────────────────

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """단일 프레임 추론 — SAM2Base.track_step() 호출 (fp16 autocast)"""
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_fp16):
            current_out = self.model.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=inference_state["num_frames"],
                track_in_reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=prev_sam_mask_logits,
            )

        # 저장 최적화
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out.get("maskmem_features")
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)

        pred_masks_gpu = current_out["pred_masks"].float()  # fp16→fp32 for output
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)

        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out.get("obj_ptr")
        object_score_logits = current_out.get("object_score_logits")

        compact_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_out, pred_masks_gpu

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size,
        high_res_masks, object_score_logits, is_mask_from_pts,
    ):
        """메모리 인코더 실행"""
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """maskmem_pos_enc 캐싱 (프레임 간 동일)"""
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            return [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return None

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """마스크를 원본 비디오 해상도로 리사이즈"""
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self.model._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    # ─────────────────────────────────────────────────────────
    # 상태 관리
    # ─────────────────────────────────────────────────────────

    def reset_state(self, state: dict):
        """추적 상태 완전 초기화"""
        state["obj_id_to_idx"].clear()
        state["obj_idx_to_id"].clear()
        state["obj_ids"].clear()
        state["point_inputs_per_obj"].clear()
        state["mask_inputs_per_obj"].clear()
        state["output_dict_per_obj"].clear()
        state["temp_output_dict_per_obj"].clear()
        state["frames_tracked_per_obj"].clear()
        state["cached_features"].clear()
        state["constants"].clear()
