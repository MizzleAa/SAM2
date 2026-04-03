"""
HVS SAM2로 COCO mini 이미지 Fine-tuning

■ 파이프라인:
  1. HVS 모델 빌드 (build_sam2_image_model) + Facebook 체크포인트 로드
  2. COCO mini 데이터셋으로 몇 에폭 fine-tune
  3. 학습된 체크포인트 저장
  4. predict_image_hvs.py로 추론 검증 가능

■ HVS Trainer 사용:
  - AMP (float16)
  - GradClip
  - 모듈별 LR 분리 (backbone 0.1x)
"""
import sys
import os
import time

import torch
from torch.utils.data import DataLoader

# HVS 경로
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hvs.training.trainer import Trainer
from hvs.training.sam2_dataset import SAM2Dataset

# ─── 설정 ──────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "annotations", "instances.json")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "hvs_finetuned")

EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-5
IMAGE_SIZE = 1024


def train():
    print("=" * 60)
    print("  HVS SAM2 Fine-tuning (COCO mini)")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Trainer (Facebook 체크포인트로 초기화)
    print("\n  [1] Building HVS Trainer...")
    t0 = time.time()
    trainer = Trainer(
        model_size="tiny",
        image_size=IMAGE_SIZE,
        lr=LR,
        weight_decay=0.01,
        backbone_lr_factor=0.1,
        device="cuda",
        init_mode="finetune",
        checkpoint_path=CHECKPOINT,
        save_dir=SAVE_DIR,
        grad_clip=1.0,
        use_amp=True,
    )
    print(f"      Built in {time.time() - t0:.2f}s")

    # 파라미터 정보
    params = trainer.count_parameters()
    print(f"      Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    print(f"        image_encoder: {params['image_encoder']:,}")
    print(f"        prompt_encoder: {params['prompt_encoder']:,}")
    print(f"        mask_decoder: {params['mask_decoder']:,}")

    # 데이터셋
    print("\n  [2] Loading dataset...")
    dataset = SAM2Dataset(
        image_dir=IMAGES_DIR,
        annotation_file=ANNOTATION_FILE,
        image_size=IMAGE_SIZE,
        num_points=1,
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )
    print(f"      Samples: {len(dataset)}, Batches: {len(dataloader)}")

    # 학습
    print(f"\n  [3] Training: {EPOCHS} epochs")
    print("-" * 60)

    total_start = time.time()

    for epoch in range(EPOCHS):
        # Train
        train_result = trainer.train_epoch(dataloader, epoch)

        # Validate
        val_result = trainer.validate(dataloader)

        print(
            f"  Epoch {epoch+1}/{EPOCHS}: "
            f"loss={train_result['total']:.4f} "
            f"(focal={train_result['focal']:.4f}, dice={train_result['dice']:.4f}), "
            f"IoU={val_result['iou']:.4f}"
        )

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"  Training done in {total_time:.1f}s")

    # 체크포인트 저장
    save_path = trainer.save_checkpoint(
        path=os.path.join(SAVE_DIR, "sam2_tiny_finetuned.pt"),
        epoch=EPOCHS,
    )
    print(f"  Checkpoint saved: {save_path}")

    print("\n" + "=" * 60)
    print(f"  HVS fine-tune 완료!")
    print(f"  체크포인트: {save_path}")
    print(f"  검증: python hvs/sample/predict_image_hvs.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
