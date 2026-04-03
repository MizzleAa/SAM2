"""
Facebook SAM2 단일 이미지 추론 속도 최적화 벤치마크
- 워밍업 후 실제 속도 측정
- FP32 vs float16 비교
"""
import sys
import os
import time
import json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "image_fb", "bench_result.json")


def bench_predict(predictor, image_rgb, point, num_warmup=3, num_runs=10):
    """워밍업 후 인코딩/예측 시간 측정"""
    pt = np.array([list(point)], dtype=np.float32)
    lb = np.array([1], dtype=np.int32)

    # Warmup
    for _ in range(num_warmup):
        predictor.set_image(image_rgb)
        predictor.predict(point_coords=pt, point_labels=lb, multimask_output=True)
    torch.cuda.synchronize()

    # Measure encoding
    enc_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.set_image(image_rgb)
        torch.cuda.synchronize()
        enc_times.append((time.perf_counter() - t0) * 1000)

    # Measure prediction
    predictor.set_image(image_rgb)
    torch.cuda.synchronize()
    pred_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict(point_coords=pt, point_labels=lb, multimask_output=True)
        torch.cuda.synchronize()
        pred_times.append((time.perf_counter() - t0) * 1000)

    return {
        "enc_mean": float(np.mean(enc_times)),
        "enc_std": float(np.std(enc_times)),
        "pred_mean": float(np.mean(pred_times)),
        "pred_std": float(np.std(pred_times)),
        "total_mean": float(np.mean(enc_times) + np.mean(pred_times)),
        "fps": float(1000.0 / (np.mean(enc_times) + np.mean(pred_times))),
    }


def main():
    results = {}

    # GPU 정보
    gpu_name = "unknown"
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    results["gpu"] = gpu_name
    results["gpu_mem_gb"] = round(gpu_mem, 1)

    # 이미지 로드
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
    image_path = os.path.join(IMAGES_DIR, images[0])
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    point = (w // 2, h // 2)
    results["image"] = images[0]
    results["image_size"] = f"{w}x{h}"

    NUM_WARMUP = 5
    NUM_RUNS = 15

    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Image: {images[0]} ({w}x{h})")
    print(f"Warmup: {NUM_WARMUP}, Runs: {NUM_RUNS}")
    print()

    # ── Test 1: FP32 ──
    print("[Test 1] FP32 Default ...")
    t0 = time.perf_counter()
    model = build_sam2(MODEL_CFG, CHECKPOINT, device="cuda")
    predictor = SAM2ImagePredictor(model)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"  Build: {build_time:.0f}ms")

    r1 = bench_predict(predictor, image_rgb, point, NUM_WARMUP, NUM_RUNS)
    print(f"  Encoding:   {r1['enc_mean']:.1f} +/- {r1['enc_std']:.1f} ms")
    print(f"  Prediction: {r1['pred_mean']:.1f} +/- {r1['pred_std']:.1f} ms")
    print(f"  Total:      {r1['total_mean']:.1f} ms ({r1['fps']:.1f} FPS)")
    results["fp32"] = r1
    results["fp32"]["build_ms"] = build_time

    del predictor, model
    torch.cuda.empty_cache()

    # ── Test 2: float16 ──
    print("\n[Test 2] float16 ...")
    try:
        model = build_sam2(MODEL_CFG, CHECKPOINT, device="cuda")
        # float16으로 변환 (autocast 사용)
        predictor = SAM2ImagePredictor(model)

        # 워밍업 + 측정 (autocast로)
        pt_arr = np.array([list(point)], dtype=np.float32)
        lb_arr = np.array([1], dtype=np.int32)

        # Warmup with autocast
        for _ in range(NUM_WARMUP):
            with torch.autocast("cuda", dtype=torch.float16):
                predictor.set_image(image_rgb)
                predictor.predict(point_coords=pt_arr, point_labels=lb_arr, multimask_output=True)
        torch.cuda.synchronize()

        # Measure encoding with autocast
        enc_times = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                predictor.set_image(image_rgb)
            torch.cuda.synchronize()
            enc_times.append((time.perf_counter() - t0) * 1000)

        # Measure prediction with autocast
        with torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(image_rgb)
        torch.cuda.synchronize()
        pred_times = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                predictor.predict(point_coords=pt_arr, point_labels=lb_arr, multimask_output=True)
            torch.cuda.synchronize()
            pred_times.append((time.perf_counter() - t0) * 1000)

        r2 = {
            "enc_mean": float(np.mean(enc_times)),
            "enc_std": float(np.std(enc_times)),
            "pred_mean": float(np.mean(pred_times)),
            "pred_std": float(np.std(pred_times)),
            "total_mean": float(np.mean(enc_times) + np.mean(pred_times)),
            "fps": float(1000.0 / (np.mean(enc_times) + np.mean(pred_times))),
        }
        print(f"  Encoding:   {r2['enc_mean']:.1f} +/- {r2['enc_std']:.1f} ms")
        print(f"  Prediction: {r2['pred_mean']:.1f} +/- {r2['pred_std']:.1f} ms")
        print(f"  Total:      {r2['total_mean']:.1f} ms ({r2['fps']:.1f} FPS)")
        print(f"  vs FP32:    {r1['total_mean']/r2['total_mean']:.2f}x")
        results["fp16_autocast"] = r2
        del predictor, model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["fp16_autocast"] = {"error": str(e)}

    # ── Test 3: torch.compile ──
    print("\n[Test 3] FP32 + torch.compile ...")
    try:
        model = build_sam2(MODEL_CFG, CHECKPOINT, device="cuda")
        model.image_encoder = torch.compile(model.image_encoder, mode="reduce-overhead")
        predictor = SAM2ImagePredictor(model)
        print("  Compiling (first runs will be slow)...")

        r3 = bench_predict(predictor, image_rgb, point, num_warmup=5, num_runs=NUM_RUNS)
        print(f"  Encoding:   {r3['enc_mean']:.1f} +/- {r3['enc_std']:.1f} ms")
        print(f"  Prediction: {r3['pred_mean']:.1f} +/- {r3['pred_std']:.1f} ms")
        print(f"  Total:      {r3['total_mean']:.1f} ms ({r3['fps']:.1f} FPS)")
        print(f"  vs FP32:    {r1['total_mean']/r3['total_mean']:.2f}x")
        results["fp32_compile"] = r3
        del predictor, model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["fp32_compile"] = {"error": str(e)}

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY (warmup={NUM_WARMUP}, runs={NUM_RUNS})")
    print(f"{'='*60}")
    print(f"  {'Config':<22s} {'Enc':>7s} {'Pred':>7s} {'Total':>7s} {'FPS':>6s}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    
    tests = [("FP32", "fp32"), ("FP16 autocast", "fp16_autocast"), ("FP32+compile", "fp32_compile")]
    for name, key in tests:
        if key in results and "error" not in results[key]:
            r = results[key]
            print(f"  {name:<22s} {r['enc_mean']:>6.1f} {r['pred_mean']:>6.1f} {r['total_mean']:>6.1f} {r['fps']:>5.1f}")

    print(f"\n  Target: 30 FPS = 33.3 ms/frame")
    print(f"{'='*60}")

    # JSON 저장
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
