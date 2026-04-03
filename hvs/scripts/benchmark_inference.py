"""
추론 속도 벤치마크 (Inference Benchmark)

■ 측정 항목 (함수별 분리):
  1. benchmark_image_encoding()  — 이미지 인코딩 속도
  2. benchmark_mask_decoding()   — 마스크 디코딩 속도 (프롬프트→마스크)
  3. benchmark_end_to_end()      — 전체 파이프라인 (set_image + predict)
  4. benchmark_torch_compile()   — torch.compile 적용 효과

■ 모델 크기별:
  tiny, small, base_plus, large

■ 최적화 기법 (함수별 분리):
  5. apply_torch_compile()   — torch.compile 적용
  6. apply_half_precision()  — FP16 변환
  7. create_benchmark_report() — 결과 보고서 생성

■ 목표: RTX 5060에서 30FPS = 33ms/frame

사용법:
    python hvs/scripts/benchmark_inference.py
    python hvs/scripts/benchmark_inference.py --model_size tiny --device cuda
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hvs.predictor.image_predictor import ImagePredictor
from hvs.models.build import build_sam2_image_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# 1. 이미지 인코딩 벤치마크
# ─────────────────────────────────────────────────

def benchmark_image_encoding(
    predictor: ImagePredictor,
    image_size: Tuple[int, int] = (640, 480),
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """
    이미지 인코딩 속도 측정

    Returns:
        {mean_ms, std_ms, fps}
    """
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(num_warmup):
        predictor.set_image(image)

    # 측정
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictor.set_image(image)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}


# ─────────────────────────────────────────────────
# 2. 마스크 디코딩 벤치마크
# ─────────────────────────────────────────────────

def benchmark_mask_decoding(
    predictor: ImagePredictor,
    image_size: Tuple[int, int] = (640, 480),
    num_warmup: int = 5,
    num_runs: int = 50,
) -> dict:
    """
    마스크 디코딩 속도 측정 (이미지 인코딩 제외)

    Returns:
        {mean_ms, std_ms, fps}
    """
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    predictor.set_image(image)

    point = np.array([[image_size[1] // 2, image_size[0] // 2]])
    label = np.array([1])

    # Warmup
    for _ in range(num_warmup):
        predictor.predict(point_coords=point, point_labels=label)

    # 측정
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictor.predict(point_coords=point, point_labels=label)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}


# ─────────────────────────────────────────────────
# 3. End-to-End 벤치마크
# ─────────────────────────────────────────────────

def benchmark_end_to_end(
    predictor: ImagePredictor,
    image_size: Tuple[int, int] = (640, 480),
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """
    전체 파이프라인 속도 측정 (set_image + predict)

    Returns:
        {mean_ms, std_ms, fps, encoding_ms, decoding_ms}
    """
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    point = np.array([[image_size[1] // 2, image_size[0] // 2]])
    label = np.array([1])

    # Warmup
    for _ in range(num_warmup):
        predictor.set_image(image)
        predictor.predict(point_coords=point, point_labels=label)

    # 측정
    total_times = []
    enc_times = []
    dec_times = []

    for _ in range(num_runs):
        # Encoding
        start = time.perf_counter()
        predictor.set_image(image)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        enc_elapsed = (time.perf_counter() - start) * 1000

        # Decoding
        start = time.perf_counter()
        predictor.predict(point_coords=point, point_labels=label)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        dec_elapsed = (time.perf_counter() - start) * 1000

        total_times.append(enc_elapsed + dec_elapsed)
        enc_times.append(enc_elapsed)
        dec_times.append(dec_elapsed)

    mean_ms = np.mean(total_times)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    return {
        "mean_ms": mean_ms,
        "std_ms": np.std(total_times),
        "fps": fps,
        "encoding_ms": np.mean(enc_times),
        "decoding_ms": np.mean(dec_times),
    }


# ─────────────────────────────────────────────────
# 4. torch.compile 벤치마크
# ─────────────────────────────────────────────────

def apply_torch_compile(predictor: ImagePredictor, mode: str = "reduce-overhead") -> bool:
    """
    torch.compile 적용

    Args:
        predictor: ImagePredictor
        mode: "default", "reduce-overhead", "max-autotune"

    Returns:
        True if successful
    """
    try:
        predictor.ie = torch.compile(predictor.ie, mode=mode)
        predictor.pe = torch.compile(predictor.pe, mode=mode)
        predictor.md = torch.compile(predictor.md, mode=mode)
        return True
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
        return False


# ─────────────────────────────────────────────────
# 5. FP16 적용
# ─────────────────────────────────────────────────

def apply_half_precision(predictor: ImagePredictor) -> None:
    """모델을 FP16으로 변환 (GPU 전용)"""
    if predictor.device.type != "cuda":
        logger.warning("FP16 is only effective on CUDA devices")
        return
    predictor.ie = predictor.ie.half()
    predictor.pe = predictor.pe.half()
    predictor.md = predictor.md.half()
    predictor.pixel_mean = predictor.pixel_mean.half()
    predictor.pixel_std = predictor.pixel_std.half()


# ─────────────────────────────────────────────────
# 6. 보고서 생성
# ─────────────────────────────────────────────────

def create_benchmark_report(
    results: Dict[str, dict],
    model_size: str,
    device: str,
    image_size: int,
) -> str:
    """벤치마크 결과 보고서 생성"""
    lines = [
        f"═══ SAM2 Inference Benchmark Report ═══",
        f"  Model: {model_size}",
        f"  Device: {device}",
        f"  Input size: {image_size}",
        "",
    ]

    for name, r in results.items():
        fps = r.get("fps", 0)
        mean = r.get("mean_ms", 0)
        std = r.get("std_ms", 0)
        target = "✅ 30FPS" if fps >= 30 else f"❌ {fps:.1f}FPS"

        lines.append(f"  [{name}]")
        lines.append(f"    Time: {mean:.1f} ± {std:.1f} ms")
        lines.append(f"    FPS:  {fps:.1f} ({target})")

        if "encoding_ms" in r:
            lines.append(f"    ├ Encoding:  {r['encoding_ms']:.1f} ms")
            lines.append(f"    └ Decoding:  {r['decoding_ms']:.1f} ms")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────

def run_benchmark(
    model_size: str = "tiny",
    image_size: int = 1024,
    device: str = "cpu",
    checkpoint_path: str = None,
    num_runs: int = 10,
    use_compile: bool = False,
    use_fp16: bool = False,
) -> Dict[str, dict]:
    """
    전체 벤치마크 실행

    Returns:
        {encoding: {...}, decoding: {...}, end_to_end: {...}}
    """
    predictor = ImagePredictor(
        model_size=model_size,
        image_size=image_size,
        device=device,
        checkpoint_path=checkpoint_path,
        init_mode="finetune" if checkpoint_path else "scratch",
    )

    if use_fp16:
        apply_half_precision(predictor)

    if use_compile:
        apply_torch_compile(predictor)

    results = {}

    # 1. 이미지 인코딩
    results["encoding"] = benchmark_image_encoding(
        predictor, image_size=(480, 640), num_runs=num_runs,
    )

    # 2. 마스크 디코딩
    results["decoding"] = benchmark_mask_decoding(
        predictor, image_size=(480, 640), num_runs=num_runs * 5,
    )

    # 3. End-to-End
    results["end_to_end"] = benchmark_end_to_end(
        predictor, image_size=(480, 640), num_runs=num_runs,
    )

    # 보고서
    report = create_benchmark_report(results, model_size, device, image_size)
    print(report)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Inference Benchmark")
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        model_size=args.model_size,
        image_size=args.image_size,
        device=args.device,
        checkpoint_path=args.checkpoint,
        num_runs=args.num_runs,
        use_compile=args.compile,
        use_fp16=args.fp16,
    )
