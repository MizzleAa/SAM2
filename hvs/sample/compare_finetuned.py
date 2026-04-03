"""
Fine-tuned model prediction comparison.

Flow:
  1. Facebook train -> Facebook predict
  2. HVS train -> HVS predict
  3. Compare results side by side

Uses the already-trained checkpoints:
  - checkpoints/facebook_finetuned/sam2_tiny_finetuned.pt
  - checkpoints/hvs_finetuned/sam2_tiny_finetuned.pt
"""
import sys
import os
import json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── Settings ──────────────────────────────────────
# Pretrained (original) checkpoint
PRETRAINED_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
# Fine-tuned checkpoints
FB_FINETUNED_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "facebook_finetuned", "sam2_tiny_finetuned.pt"
)
HVS_FINETUNED_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "hvs_finetuned", "sam2_tiny_finetuned.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "annotations", "instances.json")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "finetuned_compare")


def compute_mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / (union + 1e-6)


def load_coco_annotations(annotation_file, images_dir):
    """Load COCO annotations and get GT point prompts per image."""
    with open(annotation_file, "r") as f:
        coco = json.load(f)
    
    img_map = {img["id"]: img for img in coco["images"]}
    
    # Group annotations by image
    samples = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in img_map:
            continue
        img_info = img_map[img_id]
        fname = img_info["file_name"]
        
        if fname not in samples:
            samples[fname] = {
                "file_name": fname,
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": [],
            }
        samples[fname]["annotations"].append(ann)
    
    return samples


def get_point_from_mask(ann, h, w):
    """Get a point prompt from annotation mask centroid."""
    seg = ann.get("segmentation", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
    
    fg = np.argwhere(mask > 0)
    if len(fg) > 0:
        cy, cx = fg.mean(axis=0).astype(int)
        return (int(cx), int(cy)), mask
    return (w // 2, h // 2), mask


def build_fb_predictor(checkpoint_path):
    """Build Facebook predictor with checkpoint."""
    model = build_sam2(MODEL_CFG, checkpoint_path, device=torch.device("cuda"))
    model.eval()
    return SAM2ImagePredictor(model)


def build_hvs_predictor(checkpoint_path):
    """Build HVS predictor with fine-tuned checkpoint."""
    from hvs.predictor.image_predictor import ImagePredictor
    
    # HVS Trainer saves as {"model": {"image_encoder": ..., "prompt_encoder": ..., "mask_decoder": ...}}
    # We need to flatten this for load_checkpoint
    predictor = ImagePredictor(
        model_size="tiny",
        image_size=1024,
        checkpoint_path=None,  # Don't load yet
        init_mode="scratch",
    )
    
    # Load fine-tuned weights
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw:
        state = raw["model"]
        # HVS format: {"image_encoder": {...}, "prompt_encoder": {...}, "mask_decoder": {...}}
        if "image_encoder" in state and isinstance(state["image_encoder"], dict):
            predictor.ie.load_state_dict(state["image_encoder"])
            predictor.pe.load_state_dict(state["prompt_encoder"])
            predictor.md.load_state_dict(state["mask_decoder"])
            print(f"    Loaded HVS fine-tuned checkpoint (nested dict format)")
        else:
            # Flat format with prefixed keys  
            from hvs.utils.checkpoint import load_checkpoint
            load_checkpoint(predictor.model_parts, checkpoint_path, mode="finetune", strict=False)
            print(f"    Loaded flat checkpoint format")
    
    return predictor


def predict_and_compare():
    print("=" * 70)
    print("  Fine-tuned Model Prediction Comparison")
    print("  (Facebook trained vs HVS trained vs Pretrained baseline)")
    print("=" * 70)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load COCO annotations for GT-based point prompts
    coco_samples = load_coco_annotations(ANNOTATION_FILE, IMAGES_DIR)

    # Build predictors
    print("\n  [1] Building predictors...")
    print("    - Pretrained (baseline)...", end="", flush=True)
    pretrained_predictor = build_fb_predictor(PRETRAINED_CKPT)
    print(" done")
    
    print("    - Facebook fine-tuned...", end="", flush=True)
    fb_ft_predictor = build_fb_predictor(FB_FINETUNED_CKPT)
    print(" done")

    print("    - HVS fine-tuned...", end="", flush=True)
    hvs_ft_predictor = build_hvs_predictor(HVS_FINETUNED_CKPT)
    print(" done")

    # Compare
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])[:5]
    
    print(f"\n  [2] Predicting on {len(images)} images (with GT point prompts)...")
    print("-" * 70)
    print(f"  {'Image':<25} {'Pretrained':>10} {'FB-FT':>10} {'HVS-FT':>10} {'GT IoU(Pre)':>11} {'GT IoU(FB)':>11} {'GT IoU(HVS)':>11}")
    print("-" * 70)

    results = []

    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Get GT point from COCO annotation
        sample = coco_samples.get(img_name)
        if sample and sample["annotations"]:
            ann = sample["annotations"][0]
            (px, py), gt_mask = get_point_from_mask(ann, h, w)
        else:
            px, py = w // 2, h // 2
            gt_mask = None

        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        # Predict with all 3 models
        all_masks = {}
        all_scores = {}
        for name, predictor in [
            ("pretrained", pretrained_predictor),
            ("fb_ft", fb_ft_predictor),
            ("hvs_ft", hvs_ft_predictor),
        ]:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            best_idx = int(np.argmax(scores))
            all_masks[name] = masks[best_idx].astype(bool)
            all_scores[name] = float(scores[best_idx])

        # IoU vs GT
        gt_ious = {}
        if gt_mask is not None:
            gt_bool = gt_mask.astype(bool)
            for name in ["pretrained", "fb_ft", "hvs_ft"]:
                gt_ious[name] = compute_mask_iou(all_masks[name], gt_bool)
        else:
            for name in ["pretrained", "fb_ft", "hvs_ft"]:
                gt_ious[name] = -1.0

        short_name = img_name[:20] + "..." if len(img_name) > 20 else img_name
        print(f"  {short_name:<25} "
              f"{all_scores['pretrained']:>10.4f} "
              f"{all_scores['fb_ft']:>10.4f} "
              f"{all_scores['hvs_ft']:>10.4f} "
              f"{gt_ious['pretrained']:>11.4f} "
              f"{gt_ious['fb_ft']:>11.4f} "
              f"{gt_ious['hvs_ft']:>11.4f}")

        results.append({
            "image": img_name,
            "scores": all_scores,
            "gt_ious": gt_ious,
        })

        # Save comparison image (5 columns: Original | GT | Pretrained | FB-FT | HVS-FT)
        canvas_w = w * 5
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)

        # Original with point
        orig = image_rgb.copy()
        cv2.circle(orig, (px, py), 8, (255, 0, 0), -1)
        cv2.circle(orig, (px, py), 10, (255, 255, 255), 2)
        canvas[:, :w] = orig

        # GT mask
        if gt_mask is not None:
            gt_overlay = image_rgb.copy()
            gt_overlay[gt_mask > 0] = (gt_overlay[gt_mask > 0] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
            canvas[:, w:w*2] = gt_overlay
        else:
            canvas[:, w:w*2] = image_rgb

        # Pretrained / FB-FT / HVS-FT
        colors = [(0, 255, 0), (0, 128, 255), (255, 0, 128)]
        for idx, (name, color) in enumerate(zip(["pretrained", "fb_ft", "hvs_ft"], colors)):
            overlay = image_rgb.copy()
            m = all_masks[name]
            overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            canvas[:, w*(idx+2):w*(idx+3)] = overlay

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            f"Input (pt={px},{py})",
            "GT Mask",
            f"Pretrained ({all_scores['pretrained']:.3f})",
            f"FB-FT ({all_scores['fb_ft']:.3f})",
            f"HVS-FT ({all_scores['hvs_ft']:.3f})",
        ]
        for i, label in enumerate(labels):
            cv2.putText(canvas, label, (w * i + 5, 25), font, 0.5, (255, 255, 255), 2)
            if gt_ious.get("pretrained", -1) >= 0 and i >= 2:
                key = ["pretrained", "fb_ft", "hvs_ft"][i-2]
                cv2.putText(canvas, f"GT IoU={gt_ious[key]:.3f}", (w * i + 5, 50), font, 0.5, (255, 255, 0), 2)

        save_name = f"ft_compare_{os.path.splitext(img_name)[0]}.jpg"
        cv2.imwrite(os.path.join(RESULT_DIR, save_name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Summary
    print("-" * 70)
    avg_gt = {k: np.mean([r["gt_ious"][k] for r in results if r["gt_ious"][k] >= 0]) for k in ["pretrained", "fb_ft", "hvs_ft"]}
    avg_sc = {k: np.mean([r["scores"][k] for r in results]) for k in ["pretrained", "fb_ft", "hvs_ft"]}

    print(f"\n  Summary ({len(results)} images):")
    print(f"    {'Model':<20} {'Avg Score':>10} {'Avg GT IoU':>12}")
    print(f"    {'Pretrained':<20} {avg_sc['pretrained']:>10.4f} {avg_gt['pretrained']:>12.4f}")
    print(f"    {'Facebook FT':<20} {avg_sc['fb_ft']:>10.4f} {avg_gt['fb_ft']:>12.4f}")
    print(f"    {'HVS FT':<20} {avg_sc['hvs_ft']:>10.4f} {avg_gt['hvs_ft']:>12.4f}")
    print(f"\n  Comparison images saved to: {RESULT_DIR}")
    print("=" * 70)

    # Save summary
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w") as f:
        f.write("Fine-tuned Model Prediction Comparison\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"{r['image']}: scores={r['scores']}, gt_ious={r['gt_ious']}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Avg Score: pretrained={avg_sc['pretrained']:.4f}, fb_ft={avg_sc['fb_ft']:.4f}, hvs_ft={avg_sc['hvs_ft']:.4f}\n")
        f.write(f"Avg GT IoU: pretrained={avg_gt['pretrained']:.4f}, fb_ft={avg_gt['fb_ft']:.4f}, hvs_ft={avg_gt['hvs_ft']:.4f}\n")


if __name__ == "__main__":
    predict_and_compare()
