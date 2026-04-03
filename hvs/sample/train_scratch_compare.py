"""
HVS SAM2 Scratch Training (no pretrained checkpoint)
Compare: finetune vs scratch on COCO mini 20 images
"""
import sys
import os
import time
import numpy as np
import cv2
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hvs.training.trainer import Trainer
from hvs.training.sam2_dataset import SAM2Dataset

# Settings
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "annotations", "instances.json")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "hvs_scratch")

EPOCHS = 5
IMAGE_SIZE = 1024


def train_scratch():
    print("=" * 60)
    print("  HVS SAM2 SCRATCH Training (no checkpoint)")
    print("  vs Fine-tune (Facebook checkpoint)")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Dataset
    dataset = SAM2Dataset(
        image_dir=IMAGES_DIR,
        annotation_file=ANNOTATION_FILE,
        image_size=IMAGE_SIZE,
        num_points=1,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # ── Scratch model ──
    print("\n  [1] Building SCRATCH model (random init)...")
    t0 = time.time()
    scratch_trainer = Trainer(
        model_size="tiny",
        image_size=IMAGE_SIZE,
        lr=1e-4,  # Higher LR for scratch
        weight_decay=0.01,
        backbone_lr_factor=1.0,  # Same LR for backbone (no pretrained)
        device="cuda",
        init_mode="scratch",     # <<< NO CHECKPOINT
        checkpoint_path=None,
        save_dir=SAVE_DIR,
        grad_clip=1.0,
        use_amp=True,
    )
    params = scratch_trainer.count_parameters()
    print(f"      Built in {time.time() - t0:.2f}s")
    print(f"      Parameters: {params['total']:,}")

    # Train
    print(f"\n  [2] Training SCRATCH: {EPOCHS} epochs, {len(dataloader)} steps/epoch")
    print("-" * 60)

    for epoch in range(EPOCHS):
        train_result = scratch_trainer.train_epoch(dataloader, epoch)
        val_result = scratch_trainer.validate(dataloader)
        print(
            f"  Epoch {epoch+1}/{EPOCHS}: "
            f"loss={train_result['total']:.4f} "
            f"(focal={train_result['focal']:.4f}, dice={train_result['dice']:.4f}), "
            f"IoU={val_result['iou']:.4f}"
        )

    print("-" * 60)

    # Save
    save_path = scratch_trainer.save_checkpoint(
        path=os.path.join(SAVE_DIR, "sam2_tiny_scratch.pt"),
        epoch=EPOCHS,
    )
    print(f"  Checkpoint saved: {save_path}")

    # ── Predict comparison ──
    print(f"\n  [3] Comparing predictions: Scratch vs Finetune vs Pretrained")

    from hvs.predictor.image_predictor import ImagePredictor

    # Scratch predictor
    scratch_pred = ImagePredictor(
        model_size="tiny", image_size=1024,
        checkpoint_path=None, init_mode="scratch",
    )
    raw = torch.load(save_path, map_location="cpu", weights_only=True)
    state = raw["model"]
    scratch_pred.ie.load_state_dict(state["image_encoder"])
    scratch_pred.pe.load_state_dict(state["prompt_encoder"])
    scratch_pred.md.load_state_dict(state["mask_decoder"])

    # Pretrained predictor
    pretrained_ckpt = os.path.join(
        os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
    )
    pretrained_pred = ImagePredictor(
        model_size="tiny", image_size=1024,
        checkpoint_path=pretrained_ckpt, init_mode="finetune",
    )

    # Finetune predictor
    ft_ckpt = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "hvs_finetuned", "sam2_tiny_finetuned.pt"
    )
    ft_pred = ImagePredictor(
        model_size="tiny", image_size=1024,
        checkpoint_path=None, init_mode="scratch",
    )
    ft_raw = torch.load(ft_ckpt, map_location="cpu", weights_only=True)
    ft_state = ft_raw["model"]
    ft_pred.ie.load_state_dict(ft_state["image_encoder"])
    ft_pred.pe.load_state_dict(ft_state["prompt_encoder"])
    ft_pred.md.load_state_dict(ft_state["mask_decoder"])

    # Load GT
    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)
    img_map = {img["id"]: img for img in coco["images"]}
    ann_by_file = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        img_info = img_map.get(ann["image_id"])
        if img_info:
            fname = img_info["file_name"]
            if fname not in ann_by_file:
                ann_by_file[fname] = {"info": img_info, "anns": []}
            ann_by_file[fname]["anns"].append(ann)

    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])[:5]
    result_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "scratch_compare")
    os.makedirs(result_dir, exist_ok=True)

    print("-" * 70)
    print(f"  {'Image':<25} {'Scratch':>10} {'Pretrained':>10} {'Finetune':>10} {'GT(Scr)':>10} {'GT(Pre)':>10} {'GT(FT)':>10}")
    print("-" * 70)

    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # GT point
        if img_name in ann_by_file:
            ann = ann_by_file[img_name]["anns"][0]
            seg = ann.get("segmentation", [])
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(gt_mask, [pts], 1)
            fg = np.argwhere(gt_mask > 0)
            if len(fg) > 0:
                cy, cx = fg.mean(axis=0).astype(int)
                px, py = int(cx), int(cy)
            else:
                px, py = w // 2, h // 2
        else:
            px, py = w // 2, h // 2
            gt_mask = None

        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        scores = {}
        masks = {}
        gt_ious = {}
        for name, pred in [("scratch", scratch_pred), ("pretrained", pretrained_pred), ("finetune", ft_pred)]:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred.set_image(image_rgb)
                m, s, _ = pred.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
            best = int(np.argmax(s))
            scores[name] = float(s[best])
            masks[name] = m[best].astype(bool)
            if gt_mask is not None:
                gt_bool = gt_mask.astype(bool)
                inter = np.logical_and(masks[name], gt_bool).sum()
                union = np.logical_or(masks[name], gt_bool).sum()
                gt_ious[name] = inter / (union + 1e-6)
            else:
                gt_ious[name] = -1

        short = img_name[:22] + "..." if len(img_name) > 22 else img_name
        print(f"  {short:<25} "
              f"{scores['scratch']:>10.4f} {scores['pretrained']:>10.4f} {scores['finetune']:>10.4f} "
              f"{gt_ious['scratch']:>10.4f} {gt_ious['pretrained']:>10.4f} {gt_ious['finetune']:>10.4f}")

        # Save comparison (4 cols: Input+GT | Scratch | Pretrained | Finetune)
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
        orig = image_rgb.copy()
        cv2.circle(orig, (px, py), 8, (255, 0, 0), -1)
        if gt_mask is not None:
            orig[gt_mask > 0] = (orig[gt_mask > 0] * 0.7 + np.array([255, 255, 0]) * 0.3).astype(np.uint8)
        canvas[:, :w] = orig

        colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
        for i, (name, color) in enumerate(zip(["scratch", "pretrained", "finetune"], colors)):
            ov = image_rgb.copy()
            ov[masks[name]] = (ov[masks[name]] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            canvas[:, w*(i+1):w*(i+2)] = ov

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, label in enumerate([
            "Input + GT",
            f"Scratch ({scores['scratch']:.3f})",
            f"Pretrained ({scores['pretrained']:.3f})",
            f"Finetune ({scores['finetune']:.3f})",
        ]):
            cv2.putText(canvas, label, (w*i + 5, 25), font, 0.5, (255, 255, 255), 2)

        cv2.imwrite(
            os.path.join(result_dir, f"scratch_{os.path.splitext(img_name)[0]}.jpg"),
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        )

    print("-" * 70)
    print(f"\n  Comparison images: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train_scratch()
