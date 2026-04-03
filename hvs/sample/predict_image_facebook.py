"""
Facebook 공식 SAM2 ImagePredictor로 단일 이미지 예측 (속도 벤치마크)

■ 최적화:
  - bfloat16 AMP (Automatic Mixed Precision)
  - TF32 (Ampere+ GPU)
  - torch.inference_mode()
  - (torch.compile은 Windows에서 Triton 미지원으로 제외)

■ 측정 항목:
  1. 모델 빌드 시간
  2. 이미지 인코딩 시간 (set_image + AMP)
  3. 마스크 디코딩 시간 (predict + AMP)
  4. 워밍업 후 실제 속도 (10회 반복)
  5. 후처리/저장 시간

사용법:
    python hvs/sample/predict_image_facebook.py
    python hvs/sample/predict_image_facebook.py --image path/to/image.jpg --point 200,300
    python hvs/sample/predict_image_facebook.py --image path/to/image.jpg --box 100,100,400,300
"""
import sys
import os
import time
import argparse
import numpy as np
import cv2
import torch

# Facebook SAM2 경로
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── 기본 설정 ──────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "image_fb")


# ─── 유틸리티 ──────────────────────────────────────

def create_point_overlay(image_rgb, point, label=1, radius=8):
    """포인트 프롬프트를 이미지에 표시"""
    overlay = image_rgb.copy()
    color = (0, 255, 0) if label == 1 else (255, 0, 0)
    x, y = int(point[0]), int(point[1])
    cv2.circle(overlay, (x, y), radius, color, -1)
    cv2.circle(overlay, (x, y), radius + 2, (255, 255, 255), 2)
    return overlay


def create_box_overlay(image_rgb, box, thickness=3):
    """박스 프롬프트를 이미지에 표시"""
    overlay = image_rgb.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), thickness)
    return overlay


def create_mask_overlay(image_rgb, masks, scores, alpha=0.5):
    """마스크 결과 오버레이 (최고 점수 마스크 사용)"""
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    overlay = image_rgb.copy()
    color = np.array([30, 220, 80], dtype=np.uint8)

    mask_bool = best_mask.astype(bool)
    colored = np.zeros_like(overlay)
    colored[:] = color
    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0
    )

    contours, _ = cv2.findContours(
        best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return overlay, best_idx, best_score


def format_time(ms):
    """시간 포맷팅"""
    if ms < 1:
        return f"{ms * 1000:.1f}us"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms / 1000:.2f}s"


# ─── 메인 예측 ──────────────────────────────────────

def predict_single_image(
    image_path,
    point=None,
    box=None,
    label=1,
    output_dir=None,
    multimask=True,
):
    """Facebook 공식 SAM2 ImagePredictor로 단일 이미지 예측"""
    if output_dir is None:
        output_dir = RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    timings = {}

    # ── 이미지 로드 ──
    t0 = time.perf_counter()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    timings["image_load"] = (time.perf_counter() - t0) * 1000

    image_name = os.path.basename(image_path)
    print(f"\n{'=' * 60}")
    print(f"  Facebook SAM2 Single Image Prediction")
    print(f"{'=' * 60}")
    print(f"  Image: {image_name} ({w}x{h})")

    # 기본 프롬프트 (없으면 이미지 중앙)
    if point is None and box is None:
        point = (w // 2, h // 2)
        print(f"  Prompt: center point ({point[0]}, {point[1]})")
    elif point is not None:
        print(f"  Prompt: point ({point[0]}, {point[1]}), label={label}")
    elif box is not None:
        print(f"  Prompt: box ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    # ── 최적화 설정 (Facebook benchmark.py 동일) ──
    # TF32 활성화 (Ampere 이상 GPU)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 1. 모델 빌드 ──
    print(f"\n  [1/4] Building model...", end="", flush=True)
    t0 = time.perf_counter()
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=torch.device("cuda"))
    predictor = SAM2ImagePredictor(sam2_model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["model_build"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['model_build'])} (TF32=true)")

    # ── 2. 이미지 인코딩 (set_image + AMP) ──
    print(f"  [2/4] Encoding image (AMP bfloat16)...", end="", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["image_encoding"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['image_encoding'])}")

    # ── 3. 마스크 예측 (predict + AMP) ──
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
                    box=np.array(list(box), dtype=np.float32),
                    multimask_output=multimask,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["mask_prediction"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['mask_prediction'])}")

    # 마스크 결과 정보
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_mask = masks[best_idx]
    mask_area = best_mask.sum()
    mask_ratio = mask_area / (h * w) * 100

    print(f"\n  Masks: {len(masks)}")
    for i, s in enumerate(scores):
        marker = " <-- best" if i == best_idx else ""
        print(f"    mask[{i}]: score={s:.4f}{marker}")
    print(f"  Best mask area: {int(mask_area):,} pixels ({mask_ratio:.1f}%)")

    # ── 워밍업 후 실제 속도 측정 (AMP + compile 포함) ──
    print(f"\n  [*] Measuring warmed-up speed (10 runs, AMP+compile)...", end="", flush=True)
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

    # ── 4. 후처리 + 저장 ──
    print(f"\n  [4/4] Saving results...", end="", flush=True)
    t0 = time.perf_counter()

    cv2.imwrite(os.path.join(output_dir, "input.jpg"), image_bgr)

    if point is not None:
        prompt_img = create_point_overlay(image_rgb, point, label)
    else:
        prompt_img = create_box_overlay(image_rgb, box)
    cv2.imwrite(
        os.path.join(output_dir, "prompt.jpg"),
        cv2.cvtColor(prompt_img, cv2.COLOR_RGB2BGR),
    )

    overlay, _, _ = create_mask_overlay(image_rgb, masks, scores)
    cv2.imwrite(
        os.path.join(output_dir, "mask_overlay.jpg"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        os.path.join(output_dir, "mask_binary.png"),
        (best_mask * 255).astype(np.uint8),
    )

    if multimask and len(masks) > 1:
        for i, (m, s) in enumerate(zip(masks, scores)):
            cv2.imwrite(
                os.path.join(output_dir, f"mask_{i}_score{s:.4f}.png"),
                (m * 255).astype(np.uint8),
            )

    timings["save_results"] = (time.perf_counter() - t0) * 1000
    print(f" {format_time(timings['save_results'])}")

    # ── 결과 요약 ──
    timings["total_inference"] = timings["image_encoding"] + timings["mask_prediction"]
    timings["total_pipeline"] = sum(timings[k] for k in [
        "image_load", "model_build", "image_encoding", "mask_prediction", "save_results"
    ])

    print(f"\n{'-' * 60}")
    print(f"  Timing Summary")
    print(f"{'-' * 60}")
    print(f"  [Cold Start]")
    print(f"    Image Load:      {format_time(timings['image_load']):>10s}")
    print(f"    Model Build:     {format_time(timings['model_build']):>10s}")
    print(f"    Image Encoding:  {format_time(timings['image_encoding']):>10s}")
    print(f"    Mask Prediction: {format_time(timings['mask_prediction']):>10s}")
    print(f"    Save Results:    {format_time(timings['save_results']):>10s}")
    print(f"    Inference:       {format_time(timings['total_inference']):>10s}")
    print(f"  [Warmed Up - Actual Speed]")
    print(f"    Image Encoding:  {format_time(timings['warm_encoding']):>10s}")
    print(f"    Mask Prediction: {format_time(timings['warm_prediction']):>10s}")
    print(f"    Total:           {format_time(timings['warm_total']):>10s}  ({timings['warm_fps']:.1f} FPS)")
    print(f"{'-' * 60}")
    print(f"  * Cold start = CUDA init + first run (1회성)")
    print(f"  * Warmed up  = 실제 반복 추론 속도")
    print(f"{'=' * 60}")

    # 결과 파일 저장
    with open(os.path.join(output_dir, "result_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Facebook SAM2 Image Prediction Result\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Image: {image_name} ({w}x{h})\n")
        if point is not None:
            f.write(f"Prompt: point ({point[0]}, {point[1]}), label={label}\n")
        elif box is not None:
            f.write(f"Prompt: box ({box[0]}, {box[1]}, {box[2]}, {box[3]})\n")
        f.write(f"Checkpoint: {os.path.abspath(CHECKPOINT)}\n")
        f.write(f"Multimask: {multimask}\n")
        f.write(f"\nMask Results:\n")
        f.write(f"  Best mask: [{best_idx}] score={best_score:.4f}\n")
        f.write(f"  Mask area: {int(mask_area):,} pixels ({mask_ratio:.1f}%)\n")
        f.write(f"  All scores: {[f'{s:.4f}' for s in scores]}\n")
        f.write(f"\nTiming (ms):\n")
        for key, val in timings.items():
            f.write(f"  {key}: {val:.2f}\n")

    return timings, masks, scores


# ─── 배치 예측 ──────────────────────────────────────

def predict_batch_images(image_dir=None, output_base=None):
    """디렉토리 내 모든 이미지에 대해 순차적으로 예측 (모델 1회 빌드)"""
    if image_dir is None:
        image_dir = IMAGES_DIR
    if output_base is None:
        output_base = os.path.join(RESULT_DIR, "batch")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])

    if not images:
        print(f"  No images found in {image_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Facebook SAM2 Batch Image Prediction")
    print(f"{'=' * 60}")
    print(f"  Images: {len(images)} files in {image_dir}")

    # 모델 빌드 (1회)
    print(f"\n  Building model...", end="", flush=True)
    t0 = time.perf_counter()
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=torch.device("cuda"))
    predictor = SAM2ImagePredictor(sam2_model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_build = (time.perf_counter() - t0) * 1000
    print(f" {format_time(t_build)}")

    all_results = []

    for i, fname in enumerate(images):
        image_path = os.path.join(image_dir, fname)
        output_dir = os.path.join(output_base, os.path.splitext(fname)[0])
        os.makedirs(output_dir, exist_ok=True)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        pt = (w // 2, h // 2)

        # 인코딩
        t0 = time.perf_counter()
        predictor.set_image(image_rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_enc = (time.perf_counter() - t0) * 1000

        # 예측
        t0 = time.perf_counter()
        masks, scores, logits = predictor.predict(
            point_coords=np.array([list(pt)], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pred = (time.perf_counter() - t0) * 1000

        t_total = t_enc + t_pred
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        overlay, _, _ = create_mask_overlay(image_rgb, masks, scores)
        cv2.imwrite(
            os.path.join(output_dir, "overlay.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, "mask.png"),
            (masks[best_idx] * 255).astype(np.uint8),
        )

        all_results.append({
            "name": fname, "size": f"{w}x{h}",
            "encoding_ms": t_enc, "prediction_ms": t_pred,
            "total_ms": t_total, "best_score": best_score,
        })

        print(f"  [{i+1:2d}/{len(images)}] {fname:30s}  "
              f"enc={format_time(t_enc):>8s}  "
              f"pred={format_time(t_pred):>8s}  "
              f"total={format_time(t_total):>8s}  "
              f"score={best_score:.4f}")

    # 요약
    if all_results:
        enc_times = [r["encoding_ms"] for r in all_results]
        pred_times = [r["prediction_ms"] for r in all_results]
        total_times = [r["total_ms"] for r in all_results]

        print(f"\n{'-' * 60}")
        print(f"  Batch Summary ({len(all_results)} images)")
        print(f"{'-' * 60}")
        print(f"  Model Build:       {format_time(t_build):>10s}")
        print(f"  Avg Encoding:      {format_time(np.mean(enc_times)):>10s}")
        print(f"  Avg Prediction:    {format_time(np.mean(pred_times)):>10s}")
        print(f"  Avg Total/Image:   {format_time(np.mean(total_times)):>10s}")
        print(f"  Min Total/Image:   {format_time(np.min(total_times)):>10s}")
        print(f"  Max Total/Image:   {format_time(np.max(total_times)):>10s}")
        print(f"  Avg FPS:           {1000.0 / np.mean(total_times):>8.1f} FPS")
        print(f"{'=' * 60}")

        with open(os.path.join(output_base, "batch_results.txt"), "w", encoding="utf-8") as f:
            f.write(f"Facebook SAM2 Batch Image Prediction\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Model Build: {t_build:.2f}ms\n")
            f.write(f"Images: {len(all_results)}\n\n")
            f.write(f"{'Image':30s}  {'Size':>8s}  {'Enc(ms)':>8s}  {'Pred(ms)':>8s}  {'Total(ms)':>9s}  {'Score':>6s}\n")
            f.write(f"{'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}\n")
            for r in all_results:
                f.write(f"{r['name']:30s}  {r['size']:>8s}  "
                        f"{r['encoding_ms']:>8.2f}  {r['prediction_ms']:>8.2f}  "
                        f"{r['total_ms']:>9.2f}  {r['best_score']:>6.4f}\n")
            f.write(f"\nAvg Encoding:    {np.mean(enc_times):.2f}ms\n")
            f.write(f"Avg Prediction:  {np.mean(pred_times):.2f}ms\n")
            f.write(f"Avg Total:       {np.mean(total_times):.2f}ms\n")
            f.write(f"Avg FPS:         {1000.0 / np.mean(total_times):.1f}\n")

    return all_results


# ─── CLI ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Facebook SAM2 Single Image Prediction")
    parser.add_argument("--image", type=str, default=None, help="입력 이미지 경로")
    parser.add_argument("--point", type=str, default=None, help="포인트 프롬프트 (x,y)")
    parser.add_argument("--box", type=str, default=None, help="박스 프롬프트 (x1,y1,x2,y2)")
    parser.add_argument("--label", type=int, default=1, help="포인트 레이블 (1=전경, 0=배경)")
    parser.add_argument("--output", type=str, default=None, help="결과 저장 디렉토리")
    parser.add_argument("--batch", action="store_true", help="배치 모드")
    parser.add_argument("--no-multimask", action="store_true", help="단일 마스크만 출력")
    args = parser.parse_args()

    if args.batch:
        predict_batch_images(output_base=args.output)
    else:
        image_path = args.image
        if image_path is None:
            exts = (".jpg", ".jpeg", ".png")
            images = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(exts)])
            if images:
                image_path = os.path.join(IMAGES_DIR, images[0])
            else:
                print("Error: No images found. Specify --image path.")
                sys.exit(1)

        point = None
        box = None
        if args.point:
            parts = args.point.split(",")
            point = (float(parts[0]), float(parts[1]))
        if args.box:
            parts = args.box.split(",")
            box = tuple(float(p) for p in parts)

        predict_single_image(
            image_path=image_path,
            point=point,
            box=box,
            label=args.label,
            output_dir=args.output,
            multimask=not args.no_multimask,
        )


if __name__ == "__main__":
    main()
