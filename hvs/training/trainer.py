"""
SAM2 학습 루프 (Training Loop)

■ 핵심 함수 구조 (명확히 분리):

  [초기화]
    Trainer.__init__()       — 모델 빌드 + 옵티마이저 + AMP 설정
    Trainer._build_model()   — 모델 구성요소 빌드
    Trainer._build_optimizer() — 옵티마이저 + 스케줄러 빌드

  [학습]
    Trainer.forward_step()   — 단일 배치 forward + loss (AMP 지원)
    Trainer.train_epoch()    — 1 에폭 학습 루프
    Trainer.validate()       — 검증 (IoU 계산)

  [저장/복원]
    Trainer.save_checkpoint()       — 모델+옵티마이저+스케줄러 저장
    Trainer.load_training_checkpoint() — 학습 상태 복원

  [유틸]
    Trainer.count_parameters()  — 모듈별 파라미터 수
    Trainer.get_all_params()    — 전체 파라미터 리스트

■ AMP (Automatic Mixed Precision):
  use_amp=True 시 float16으로 forward/backward 수행하여
  메모리 50% 절약 + 속도 2~3배 향상 (GPU에서).
  GradScaler로 loss scaling 자동 관리.

■ 초기화 모드:
  - finetune: Facebook 체크포인트 전체 로드
  - scratch: 완전 랜덤 초기화
  - backbone_only: 백본만 로드
"""

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hvs.models.build import build_sam2_image_model
from hvs.training.loss_fns import SAM2Loss
from hvs.training.optimizer import build_optimizer, build_scheduler, get_optimizer_summary
from hvs.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    SAM2 이미지 세그멘테이션 학습기

    Args:
        model_size: 모델 크기 ("tiny", "small", "base_plus", "large")
        image_size: 입력 이미지 크기
        lr: 기본 학습률
        weight_decay: 가중치 감쇠
        backbone_lr_factor: 백본 LR 배율 (기본 0.1)
        layer_lr_decay: 레이어별 LR 감쇠율 (1.0=비활성)
        device: 학습 디바이스
        init_mode: 초기화 모드
        checkpoint_path: 체크포인트 경로
        save_dir: 체크포인트 저장 디렉토리
        grad_clip: 그래디언트 클리핑 (0이면 비활성)
        use_amp: AMP (Mixed Precision) 사용 여부
        warmup_steps: 학습률 워밍업 스텝 수
        total_steps: 총 학습 스텝 수 (스케줄러용, 0이면 스케줄러 미사용)
    """

    def __init__(
        self,
        model_size: str = "tiny",
        image_size: int = 1024,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        backbone_lr_factor: float = 0.1,
        layer_lr_decay: float = 1.0,
        device: str = "cuda",
        init_mode: str = "scratch",
        checkpoint_path: Optional[str] = None,
        save_dir: str = "./checkpoints",
        grad_clip: float = 1.0,
        use_amp: bool = False,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        self.use_amp = use_amp and self.device.type == "cuda"
        os.makedirs(save_dir, exist_ok=True)

        # ── 모델 빌드 ──
        self._build_model(model_size, image_size)

        # ── 체크포인트 로드 ──
        if init_mode != "scratch" and checkpoint_path:
            result = load_checkpoint(
                self.model_parts, checkpoint_path,
                mode=init_mode, strict=False,
            )
            logger.info(f"Checkpoint loaded: {result}")

        # ── 옵티마이저 + 스케줄러 ──
        self._build_optimizer(lr, weight_decay, backbone_lr_factor, layer_lr_decay)
        self._build_scheduler(total_steps, warmup_steps)

        # ── AMP ──
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── 손실 함수 ──
        self.criterion = SAM2Loss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)

        # ── 학습 상태 ──
        self.global_step = 0
        self.best_iou = 0.0

    # ─────────────────────────────────────────
    # 초기화 함수 (분리)
    # ─────────────────────────────────────────

    def _build_model(self, model_size: str, image_size: int):
        """모델 구성요소 빌드"""
        self.model_parts = build_sam2_image_model(model_size, image_size)
        self.ie = self.model_parts["image_encoder"].to(self.device)
        self.pe = self.model_parts["prompt_encoder"].to(self.device)
        self.md = self.model_parts["mask_decoder"].to(self.device)

    def _build_optimizer(
        self, lr: float, weight_decay: float,
        backbone_lr_factor: float, layer_lr_decay: float,
    ):
        """옵티마이저 빌드 (모듈별 LR 분리)"""
        self.optimizer = build_optimizer(
            model_parts=self.model_parts,
            lr=lr,
            weight_decay=weight_decay,
            backbone_lr_factor=backbone_lr_factor,
            layer_lr_decay=layer_lr_decay,
        )

    def _build_scheduler(self, total_steps: int, warmup_steps: int):
        """스케줄러 빌드"""
        if total_steps > 0:
            self.scheduler = build_scheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=0.01,
            )
        else:
            self.scheduler = None

    # ─────────────────────────────────────────
    # 학습 함수 (분리)
    # ─────────────────────────────────────────

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        단일 배치 forward + loss 계산 (AMP 지원)

        Args:
            batch: {image, mask, point_coords, point_labels}
        Returns:
            dict: {total, focal, dice, iou, masks_pred}
        """
        image = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device)
        point_coords = batch["point_coords"].to(self.device)
        point_labels = batch["point_labels"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # 1) Image Encoding
            enc_out = self.ie(image)
            fpn = enc_out["backbone_fpn"]
            backbone_features = fpn[-1]
            high_res_features = [
                self.md.conv_s0(fpn[0]),
                self.md.conv_s1(fpn[1]),
            ]

            # 2) Prompt Encoding
            sparse, dense = self.pe(
                points=(point_coords, point_labels),
                boxes=None, masks=None,
            )
            image_pe = self.pe.get_dense_pe()

            # 3) Mask Decoding
            masks_pred, iou_pred, _, _ = self.md(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # 4) 마스크 크기 맞춤
            if masks_pred.shape[-2:] != mask.shape[-2:]:
                mask_resized = F.interpolate(
                    mask, size=masks_pred.shape[-2:], mode="nearest"
                )
            else:
                mask_resized = mask

            # 5) Loss 계산
            losses = self.criterion(masks_pred, mask_resized, iou_pred)

        losses["masks_pred"] = masks_pred
        return losses

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """1 에폭 학습 (AMP + 스케줄러 통합)"""
        self.ie.train()
        self.pe.train()
        self.md.train()

        epoch_losses = {"total": 0.0, "focal": 0.0, "dice": 0.0, "count": 0}

        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad(set_to_none=True)

            losses = self.forward_step(batch)
            total_loss = losses["total"]

            # AMP backward
            self.scaler.scale(total_loss).backward()

            # 그래디언트 클리핑
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.get_all_params(), self.grad_clip,
                )

            # AMP step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 스케줄러
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            epoch_losses["total"] += total_loss.item()
            epoch_losses["focal"] += losses["focal"].item()
            epoch_losses["dice"] += losses["dice"].item()
            epoch_losses["count"] += 1

        n = epoch_losses["count"]
        return {
            "total": epoch_losses["total"] / max(n, 1),
            "focal": epoch_losses["focal"] / max(n, 1),
            "dice": epoch_losses["dice"] / max(n, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """검증: 평균 IoU 계산"""
        self.ie.eval()
        self.pe.eval()
        self.md.eval()

        total_iou = 0.0
        count = 0

        for batch in dataloader:
            losses = self.forward_step(batch)
            masks_pred = losses["masks_pred"]

            mask_gt = batch["mask"].to(self.device)
            if masks_pred.shape[-2:] != mask_gt.shape[-2:]:
                mask_gt = F.interpolate(
                    mask_gt, size=masks_pred.shape[-2:], mode="nearest"
                )

            pred_binary = (masks_pred > 0).float()
            intersection = (pred_binary * mask_gt).sum(dim=(-2, -1))
            union = (pred_binary + mask_gt).clamp(0, 1).sum(dim=(-2, -1))
            iou = (intersection / (union + 1e-6)).mean().item()

            total_iou += iou
            count += 1

        return {"iou": total_iou / max(count, 1)}

    # ─────────────────────────────────────────
    # 저장/복원 함수 (분리)
    # ─────────────────────────────────────────

    def save_checkpoint(self, path: str = None, epoch: int = 0, extra: dict = None):
        """체크포인트 저장 (모델 + 옵티마이저 + 스케줄러 + AMP)"""
        if path is None:
            path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch:04d}.pt")

        state = {
            "model": {
                "image_encoder": self.ie.state_dict(),
                "prompt_encoder": self.pe.state_dict(),
                "mask_decoder": self.md.state_dict(),
            },
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if extra:
            state.update(extra)

        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_training_checkpoint(self, path: str):
        """학습 체크포인트에서 학습 상태 복원"""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.ie.load_state_dict(state["model"]["image_encoder"])
        self.pe.load_state_dict(state["model"]["prompt_encoder"])
        self.md.load_state_dict(state["model"]["mask_decoder"])

        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if "global_step" in state:
            self.global_step = state["global_step"]

        logger.info(f"Training state restored from: {path}")
        return state.get("epoch", 0)

    # ─────────────────────────────────────────
    # 유틸 함수 (분리)
    # ─────────────────────────────────────────

    def get_all_params(self):
        """전체 학습 가능 파라미터 리스트"""
        return (
            list(self.ie.parameters()) +
            list(self.pe.parameters()) +
            list(self.md.parameters())
        )

    def count_parameters(self) -> dict:
        """모듈별 파라미터 수 요약"""
        ie_p = sum(p.numel() for p in self.ie.parameters())
        pe_p = sum(p.numel() for p in self.pe.parameters())
        md_p = sum(p.numel() for p in self.md.parameters())
        total = ie_p + pe_p + md_p
        trainable = sum(p.numel() for p in self.get_all_params() if p.requires_grad)
        return {
            "image_encoder": ie_p,
            "prompt_encoder": pe_p,
            "mask_decoder": md_p,
            "total": total,
            "trainable": trainable,
        }

    def get_lr_summary(self) -> list:
        """옵티마이저 LR 그룹 요약"""
        return get_optimizer_summary(self.optimizer)
