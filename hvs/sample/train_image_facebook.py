"""
Facebook 공식 SAM2로 COCO mini 이미지 Fine-tuning

■ 파이프라인:
  1. Facebook SAM2 모델(build_sam2) + 체크포인트 로드
  2. COCO mini 데이터셋으로 몇 에폭 fine-tune
  3. 학습된 체크포인트 저장
  4. predict_image_facebook.py로 추론 검증 가능

■ 최적화:
  - AMP (bfloat16)
  - TF32
  - GradScaler
"""
import sys
import os
import time
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Facebook SAM2 경로
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── 설정 ──────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "annotations", "instances.json")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "facebook_finetuned")

EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-5
IMAGE_SIZE = 1024


# ─── COCO 데이터셋 ──────────────────────────────────────

class COCOMiniDataset(Dataset):
    """COCO mini 데이터셋을 SAM2 학습용으로 변환"""

    def __init__(self, image_dir, annotation_file, image_size=1024):
        self.image_dir = image_dir
        self.image_size = image_size

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        images = {img["id"]: img for img in coco_data["images"]}

        self.samples = []
        for ann in coco_data.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            img_info = images.get(ann["image_id"])
            if img_info is None:
                continue
            self.samples.append({
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "annotation": ann,
            })

        print(f"  Dataset: {len(self.samples)} samples from {len(images)} images")

    def _decode_mask(self, ann, height, width):
        seg = ann.get("segmentation", [])
        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        return mask

    def _sample_point(self, mask):
        fg = np.argwhere(mask > 0)
        if len(fg) > 0:
            idx = np.random.randint(len(fg))
            y, x = fg[idx]
            return np.array([[float(x), float(y)]], dtype=np.float32), np.array([1], dtype=np.int32)
        return np.array([[mask.shape[1] / 2, mask.shape[0] / 2]], dtype=np.float32), np.array([1], dtype=np.int32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self._decode_mask(sample["annotation"], sample["height"], sample["width"])

        # 리사이즈
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 포인트 샘플링 (리사이즈 후 마스크에서)
        coords, labels = self._sample_point(mask)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "image_rgb": image,
            "mask": mask_tensor,
            "point_coords": torch.from_numpy(coords),
            "point_labels": torch.from_numpy(labels),
        }


# ─── 학습 루프 ──────────────────────────────────────

def train():
    print("=" * 60)
    print("  Facebook SAM2 Fine-tuning (COCO mini)")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 모델 빌드
    print("\n  [1] Building model...")
    t0 = time.time()
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=torch.device("cuda"))
    print(f"      Built in {time.time() - t0:.2f}s")

    # 학습 모드로 전환 (image_encoder, sam_mask_decoder, sam_prompt_encoder)
    sam2_model.train()

    # 옵티마이저 (image_encoder LR 낮게)
    param_groups = [
        {"params": [p for n, p in sam2_model.named_parameters() if "image_encoder" in n and p.requires_grad],
         "lr": LR * 0.1, "name": "image_encoder"},
        {"params": [p for n, p in sam2_model.named_parameters() if "image_encoder" not in n and p.requires_grad],
         "lr": LR, "name": "head"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # 데이터셋
    print("\n  [2] Loading dataset...")
    dataset = COCOMiniDataset(IMAGES_DIR, ANNOTATION_FILE, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # 학습
    print(f"\n  [3] Training: {EPOCHS} epochs, {len(dataloader)} steps/epoch")
    print("-" * 60)

    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_iou = 0.0
        count = 0
        sam2_model.train()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 이미지 인코딩 (SAM2Base의 forward_image 사용)
                image = batch["image"].to("cuda")  # (B, 3, H, W)

                # Facebook SAM2 내부의 이미지 인코더 직접 사용
                backbone_out = sam2_model.forward_image(image)
                # backbone_out에서 features 추출
                _, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
                # FPN 특징맵을 사용하여 이미지 임베딩 생성
                if sam2_model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + sam2_model.no_mem_embed

                bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]

                # 프롬프트 인코딩
                point_coords = batch["point_coords"].to("cuda")
                point_labels = batch["point_labels"].to("cuda")

                sparse, dense = sam2_model.sam_prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                image_pe = sam2_model.sam_prompt_encoder.get_dense_pe()

                # high-res features (conv_s0/s1은 mask_decoder 내부에서 처리)
                high_res_features = feats[:-1]  # [feats[0], feats[1]]

                # 마스크 디코딩
                masks_pred, iou_pred, _, _ = sam2_model.sam_mask_decoder(
                    image_embeddings=feats[-1],
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

                # Loss 계산
                mask_gt = batch["mask"].to("cuda")
                if masks_pred.shape[-2:] != mask_gt.shape[-2:]:
                    mask_gt = F.interpolate(mask_gt, size=masks_pred.shape[-2:], mode="nearest")

                # Focal + Dice Loss (Facebook 학습과 동일)
                prob = torch.sigmoid(masks_pred)
                ce = F.binary_cross_entropy_with_logits(masks_pred, mask_gt.expand_as(masks_pred), reduction="none")
                p_t = prob * mask_gt.expand_as(prob) + (1 - prob) * (1 - mask_gt.expand_as(prob))
                focal = ((1 - p_t) ** 2) * ce
                focal_loss = focal.mean() * 20.0

                pred_flat = prob.flatten(1)
                gt_flat = mask_gt.expand_as(masks_pred).flatten(1)
                inter = (pred_flat * gt_flat).sum(1)
                dice = 1 - (2 * inter + 1) / (pred_flat.sum(1) + gt_flat.sum(1) + 1)
                dice_loss = dice.mean()

                total_loss = focal_loss + dice_loss

            # Backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sam2_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # IoU 계산
            with torch.no_grad():
                pred_bin = (masks_pred > 0).float()
                inter = (pred_bin * mask_gt.expand_as(pred_bin)).sum()
                union = (pred_bin + mask_gt.expand_as(pred_bin)).clamp(0, 1).sum()
                iou = (inter / (union + 1e-6)).item()

            epoch_loss += total_loss.item()
            epoch_iou += iou
            count += 1

        avg_loss = epoch_loss / max(count, 1)
        avg_iou = epoch_iou / max(count, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, IoU={avg_iou:.4f}")

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"  Training done in {total_time:.1f}s")

    # 체크포인트 저장 (Facebook 형식)
    save_path = os.path.join(SAVE_DIR, "sam2_tiny_finetuned.pt")
    torch.save({"model": sam2_model.state_dict()}, save_path)
    print(f"  Checkpoint saved: {save_path}")

    # ── 추론 검증 ──
    print(f"\n  [4] Verifying prediction...")
    sam2_model.eval()
    predictor = SAM2ImagePredictor(sam2_model)

    test_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])[:3]
    for img_name in test_images:
        img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_rgb)
            masks, scores, logits = predictor.predict(
                point_coords=np.array([[w // 2, h // 2]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        print(f"    {img_name}: score={scores[best_idx]:.4f}, area={masks[best_idx].sum():.0f}px")

    print("\n" + "=" * 60)
    print(f"  Facebook fine-tune 완료!")
    print(f"  체크포인트: {save_path}")
    print(f"  검증: python hvs/sample/predict_image_facebook.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
