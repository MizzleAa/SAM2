"""
HVS SAM2 ImagePredictor로 단일 이미지 예측 (속도 벤치마크)

■ predict_image_facebook.py와 동일한 구조:
  - 동일 이미지, 동일 프롬프트
  - 동일 시간 측정 (Cold start vs Warmed-up)
  - 결과를 results/image_hvs/ 에 저장하여 Facebook 결과와 비교 가능

■ 최적화:
  - AMP (bfloat16)
  - TF32
  - torch.inference_mode()
"""
import sys
import os
import time
import argparse
import numpy as np
import cv2
import torch

# HVS 경로
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hvs.predictor.image_predictor import ImagePredictor

# ─── 기본 설정 ──────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "image_hvs")


# ─── 유틸리티 ──────────────────────────────────────

def format_time(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def create_overlay(image_rgb, mask, color=(0, 255, 0), alpha=0.4):
    overlay = image_rgb.copy()
    overlay[mask > 0] = [int(c * alpha + o * (1 - alpha)) for c, o in zip(color, overlay[mask > 0].mean(axis=0))]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# ─── 메인 ──────────────────────────────────────

def predict_single(
    image_path=None,
    point=None,
    box=None,
    label=1,
    multimask=True,
    checkpoint_path=None,
):
    timings = {}
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 이미지 결정
    if image_path is None:
        images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])
        image_path = os.path.join(IMAGES_DIR, images[0])

    # ── 0. 이미지 로드 ──
    t0 = time.perf_counter()
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    timings["image_load"] = (time.perf_counter() - t0) * 1000

    # 프롬프트 기본값
    if point is None and box is None:
        point = (w // 2, h // 2)
    if box is not None:
        box = np.array(list(box), dtype=np.float32)

    img_name = os.path.basename(image_path)
    print(f"\n  Image: {img_name} ({h}x{w})")
    if point is not None:
        print(f"  Prompt: point ({point[0]}, {point[1]}), label={label}")
    elif box is not None:
        print(f"  Prompt: box ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    # ── 최적화 설정 ──
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 1. 모델 빌드 ──
    ckpt = checkpoint_path or CHECKPOINT
    print(f"\n  [1/4] Building HVS model...", end="", flush=True)
    t0 = time.perf_counter()
    predictor = ImagePredictor(
        model_size="tiny",
        image_size=1024,
        checkpoint_path=ckpt,
        init_mode="finetune",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["model_build"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['model_build'])} (TF32=true)")

    # ── 2. 이미지 인코딩 (AMP) ──
    print(f"  [2/4] Encoding image (AMP bfloat16)...", end="", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["image_encoding"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['image_encoding'])}")

    # ── 3. 마스크 예측 (AMP) ──
    print(f"  [3/4] Predicting masks (AMP bfloat16)...", end="", flush=True)
    t0 = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if point is not None:
                masks, scores, logits = predictor.predict(
                    point_coords=np.array([list(point)], dtype=np.float32),
                    point_labels=np.array([label], dtype=np.int32),
                    multimask_output=multimask,
                )
            elif box is not None:
                masks, scores, logits = predictor.predict(
                    box=box,
                    multimask_output=multimask,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["mask_prediction"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['mask_prediction'])}")

    # 결과 정리
    timings["total_inference"] = timings["image_encoding"] + timings["mask_prediction"]
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8)
    mask_area = best_mask.sum()
    mask_ratio = mask_area / (h * w) * 100

    print(f"\n  Results: {len(masks)} masks")
    for i, s in enumerate(scores):
        marker = " ← best" if i == best_idx else ""
        print(f"    mask[{i}]: score={s:.4f}{marker}")
    print(f"  Best mask area: {int(mask_area):,} pixels ({mask_ratio:.1f}%)")

    # ── 워밍업 후 실제 속도 측정 ──
    print(f"\n  [*] Measuring warmed-up speed (10 runs, AMP)...", end="", flush=True)
    pt_arr = np.array([list(point if point else (w//2, h//2))], dtype=np.float32)
    lb_arr = np.array([label], dtype=np.int32)

    warm_enc, warm_pred = [], []
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                predictor.set_image(image_rgb)
                torch.cuda.synchronize()
                warm_enc.append((time.perf_counter() - t0) * 1000)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                predictor.predict(point_coords=pt_arr, point_labels=lb_arr, multimask_output=multimask)
                torch.cuda.synchronize()
                warm_pred.append((time.perf_counter() - t0) * 1000)

    timings["warm_encoding"] = float(np.mean(warm_enc))
    timings["warm_prediction"] = float(np.mean(warm_pred))
    timings["warm_total"] = timings["warm_encoding"] + timings["warm_prediction"]
    timings["warm_fps"] = 1000.0 / timings["warm_total"] if timings["warm_total"] > 0 else 0
    print(f" {format_time(timings['warm_total'])} ({timings['warm_fps']:.1f} FPS)")
    print(f"      enc={timings['warm_encoding']:.1f}ms + pred={timings['warm_prediction']:.1f}ms")

    # ── 4. 저장 ──
    print(f"\n  [4/4] Saving results...", end="", flush=True)
    t0 = time.perf_counter()

    overlay = create_overlay(image_rgb, best_mask)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(RESULT_DIR, "overlay.jpg"), overlay_bgr)
    cv2.imwrite(os.path.join(RESULT_DIR, "mask.png"), best_mask * 255)

    # 결과 텍스트
    info_lines = [
        "HVS SAM2 Image Prediction Result",
        "=" * 40,
        f"Image: {img_name} ({h}x{w})",
        f"Checkpoint: {ckpt}",
        f"Multimask: {multimask}",
        "",
        "Mask Results:",
        f"  Best mask: [{best_idx}] score={scores[best_idx]:.4f}",
        f"  Mask area: {int(mask_area):,} pixels ({mask_ratio:.1f}%)",
        f"  All scores: {[f'{s:.4f}' for s in scores]}",
        "",
        "Timing (ms):",
    ]
    for key, val in timings.items():
        info_lines.append(f"  {key}: {val:.2f}")

    with open(os.path.join(RESULT_DIR, "result_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    timings["save_results"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['save_results'])}")

    # 총 소요 시간
    timings["total_pipeline"] = sum(v for k, v in timings.items() if k.startswith(("image_load", "model_build", "total_inference", "save_results")))

    # 최종 요약
    print(f"\n{'=' * 60}")
    print(f"  HVS SAM2 Prediction Summary")
    print(f"{'=' * 60}")
    print(f"  Cold start:")
    print(f"    Model build:    {format_time(timings['model_build'])}")
    print(f"    Encoding:       {format_time(timings['image_encoding'])}")
    print(f"    Prediction:     {format_time(timings['mask_prediction'])}")
    print(f"    Total:          {format_time(timings['total_inference'])}")
    print(f"  Warmed up (AMP bfloat16):")
    print(f"    Encoding:       {timings['warm_encoding']:.1f}ms")
    print(f"    Prediction:     {timings['warm_prediction']:.1f}ms")
    print(f"    Total:          {timings['warm_total']:.1f}ms ({timings['warm_fps']:.1f} FPS)")
    print(f"{'=' * 60}")
    print(f"  * Cold start = CUDA init + first run (1회성)")
    print(f"  * Warmed up  = 실제 반복 추론 속도")
    print(f"{'=' * 60}")

    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--point", type=str, default=None, help="x,y")
    parser.add_argument("--box", type=str, default=None, help="x1,y1,x2,y2")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None, help="HVS 체크포인트 경로")
    args = parser.parse_args()

    point = tuple(map(int, args.point.split(","))) if args.point else None
    box = tuple(map(int, args.box.split(","))) if args.box else None

    predict_single(
        image_path=args.image,
        point=point,
        box=box,
        label=args.label,
        checkpoint_path=args.checkpoint,
    )
