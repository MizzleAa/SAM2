"""전체 비디오 샘플 배치 예측 (fp16 최적화)"""
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

SAMPLES_DIR = os.path.join(_ROOT, ".dataset", "video_samples")
RESULT_DIR = os.path.join(_ROOT, ".dataset", "results", "video")

def find_video_file(sample_dir):
    """디렉토리에서 비디오 파일 찾기"""
    for f in sorted(os.listdir(sample_dir)):
        if f.lower().endswith(('.mp4', '.avi')):
            return os.path.join(sample_dir, f)
    return None

def run_all():
    from hvs.scripts.predict_video import predict_video

    samples = sorted(os.listdir(SAMPLES_DIR))
    samples = [s for s in samples if os.path.isdir(os.path.join(SAMPLES_DIR, s)) and not s.startswith('_')]

    print(f"\n{'='*60}")
    print(f"  배치 예측: {len(samples)}개 비디오 (fp16 최적화)")
    print(f"{'='*60}\n")

    results = []
    total_t0 = time.time()

    for name in samples:
        sample_dir = os.path.join(SAMPLES_DIR, name)
        video_path = find_video_file(sample_dir)
        if not video_path:
            print(f"  [SKIP] {name}: 비디오 파일 없음")
            continue

        # meta.json 로드
        meta_path = os.path.join(sample_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

        # point 결정 (CLI --point 형식)
        point_str = None
        if "objects" not in meta and "prompt" in meta:
            pt = meta["prompt"]["point"]
            point_str = f"{pt[0]},{pt[1]}"

        output_dir = os.path.join(RESULT_DIR, name)

        print(f"\n--- {name} ---")
        t0 = time.time()
        predict_video(
            video_path=video_path,
            output_dir=output_dir,
            point=point_str,
        )
        elapsed = time.time() - t0
        results.append((name, elapsed))
        print(f"  Total time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  전체 완료: {total_elapsed:.1f}s")
    print(f"{'='*60}")
    for name, t in results:
        print(f"  {name}: {t:.1f}s")
    print()

if __name__ == "__main__":
    run_all()
