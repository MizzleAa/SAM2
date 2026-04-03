"""
Facebook vs HVS Prediction Comparison
(Same image, same prompt, same checkpoint -- no training needed)

Compares pixel-level mask agreement between Facebook SAM2ImagePredictor
and HVS ImagePredictor using the same pretrained weights.
"""
import sys
import os
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hvs.predictor.image_predictor import ImagePredictor

# ─── Settings ──────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "compare")


def compute_mask_iou(mask_a, mask_b):
    """Calculate IoU of two masks"""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / (union + 1e-6)


def compare():
    print("=" * 70)
    print("  Facebook vs HVS Prediction Comparison (same checkpoint, no training)")
    print("=" * 70)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 1. Build models ──
    print("\n  [1] Building Facebook model...", end="", flush=True)
    fb_model = build_sam2(MODEL_CFG, CHECKPOINT, device=torch.device("cuda"))
    fb_predictor = SAM2ImagePredictor(fb_model)
    print(" done")

    print("  [2] Building HVS model...", end="", flush=True)
    hvs_predictor = ImagePredictor(
        model_size="tiny",
        image_size=1024,
        checkpoint_path=CHECKPOINT,
        init_mode="finetune",
    )
    print(" done")

    # ── 2. Image list ──
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])
    print(f"\n  [3] Comparing {len(images)} images...")
    print("-" * 70)
    print(f"  {'Image':<30} {'FB Score':>10} {'HVS Score':>10} {'IoU':>8} {'Match%':>8}")
    print("-" * 70)

    all_ious = []
    all_match_pcts = []

    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Center point prompt
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        # ── Facebook prediction ──
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            fb_predictor.set_image(image_rgb)
            fb_masks, fb_scores, _ = fb_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # ── HVS prediction ──
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hvs_predictor.set_image(image_rgb)
            hvs_masks, hvs_scores, _ = hvs_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # Compare masks
        fb_best = int(np.argmax(fb_scores))
        hvs_best = int(np.argmax(hvs_scores))

        fb_mask = fb_masks[fb_best].astype(bool)
        hvs_mask = hvs_masks[hvs_best].astype(bool)

        iou = compute_mask_iou(fb_mask, hvs_mask)
        match_pct = (fb_mask == hvs_mask).mean() * 100

        all_ious.append(iou)
        all_match_pcts.append(match_pct)

        print(f"  {img_name:<30} {fb_scores[fb_best]:>10.4f} {hvs_scores[hvs_best]:>10.4f} {iou:>8.4f} {match_pct:>7.1f}%")

        # Save visual comparison for first 3 images
        if len(all_ious) <= 3:
            # Generate comparison image (Original | FB mask | HVS mask | Diff)
            canvas_h, canvas_w = h, w * 4
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Original
            canvas[:, :w] = image_rgb

            # FB mask overlay
            fb_overlay = image_rgb.copy()
            fb_overlay[fb_mask] = (fb_overlay[fb_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            canvas[:, w:w*2] = fb_overlay

            # HVS mask overlay
            hvs_overlay = image_rgb.copy()
            hvs_overlay[hvs_mask] = (hvs_overlay[hvs_mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            canvas[:, w*2:w*3] = hvs_overlay

            # Diff (Red=FB only, Blue=HVS only, Green=Both)
            diff = np.zeros_like(image_rgb)
            both = fb_mask & hvs_mask
            fb_only = fb_mask & ~hvs_mask
            hvs_only = ~fb_mask & hvs_mask
            diff[both] = [0, 255, 0]    # Green: Both
            diff[fb_only] = [255, 0, 0]  # Red: FB only
            diff[hvs_only] = [0, 0, 255] # Blue: HVS only
            canvas[:, w*3:w*4] = diff

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, label in enumerate(["Original", "Facebook", "HVS", f"Diff (IoU={iou:.3f})"]):
                cv2.putText(canvas, label, (w * i + 10, 30), font, 0.7, (255, 255, 255), 2)

            save_name = f"compare_{os.path.splitext(img_name)[0]}.jpg"
            cv2.imwrite(
                os.path.join(RESULT_DIR, save_name),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
            )

    # Stats
    print("-" * 70)
    avg_iou = np.mean(all_ious)
    avg_match = np.mean(all_match_pcts)
    min_iou = np.min(all_ious)
    max_iou = np.max(all_ious)

    print(f"\n  Summary ({len(images)} images):")
    print(f"    Average Mask IoU:       {avg_iou:.4f}")
    print(f"    Min / Max IoU:          {min_iou:.4f} / {max_iou:.4f}")
    print(f"    Average Pixel Match:    {avg_match:.1f}%")

    # Verdict
    if avg_iou >= 0.99:
        verdict = "[PASS] Perfect match -- HVS and Facebook produce identical results."
    elif avg_iou >= 0.95:
        verdict = "[PASS] Near-identical -- Only minor floating point differences (preprocessing path)."
    elif avg_iou >= 0.85:
        verdict = "[WARN] Similar -- Slight differences due to preprocessing differences."
    else:
        verdict = "[FAIL] Mismatch -- Architecture or checkpoint loading issue."

    print(f"\n  Verdict: {verdict}")
    print(f"\n  Compare images: {RESULT_DIR}")
    print("=" * 70)

    # Save results to file
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Facebook vs HVS Prediction Comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Image':<30} {'FB Score':>10} {'HVS Score':>10} {'IoU':>8} {'Match%':>8}\n")
        f.write("-" * 70 + "\n")
        for i, img_name in enumerate(images):
            img_path = os.path.join(IMAGES_DIR, img_name)
            f.write(f"  IoU={all_ious[i]:.4f}  Match={all_match_pcts[i]:.1f}%  {img_name}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Mask IoU:    {avg_iou:.4f}\n")
        f.write(f"Min / Max IoU:       {min_iou:.4f} / {max_iou:.4f}\n")
        f.write(f"Average Pixel Match: {avg_match:.1f}%\n")
        f.write(f"Verdict: {verdict}\n")

    return {"avg_iou": avg_iou, "avg_match": avg_match, "all_ious": all_ious}


if __name__ == "__main__":
    compare()
