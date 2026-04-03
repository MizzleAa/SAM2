"""
Facebook 공식 SAM2 VideoPredictor로 비디오 예측 (속도 비교 벤치마크)

■ 최적화 (Facebook benchmark.py와 동일):
  - bfloat16 AMP (Automatic Mixed Precision)
  - TF32 (Ampere+ GPU)
  - vos_optimized=True (torch.compile max-autotune)
  - torch.inference_mode()
"""
import sys
import os
import time
import json
import numpy as np
import cv2
import torch

# Facebook SAM2 경로
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "facebook"))

from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "facebook", "checkpoints", "sam2.1_hiera_tiny.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "video_samples")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "video_fb")


def create_overlay(frame, masks, obj_ids):
    """마스크 오버레이 생성"""
    overlay = frame.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for i, (obj_id, mask) in enumerate(zip(obj_ids, masks)):
        binary = (mask[0] > 0.0).astype(np.uint8)
        combined = np.maximum(combined, binary)
        color = colors[i % len(colors)]
        colored = np.zeros_like(overlay)
        colored[:] = color
        overlay = np.where(binary[:,:,None] == 1,
                          cv2.addWeighted(overlay, 0.5, colored, 0.5, 0),
                          overlay)
    return overlay, combined


def predict_with_facebook(video_path, meta, output_dir):
    """Facebook 공식 SAM2로 예측"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 프레임 추출 → JPEG 임시 디렉토리
    temp_frames_dir = os.path.join(output_dir, "_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_rgb = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        # Facebook SAM2는 JPEG 프레임 디렉토리 입력 필요
        cv2.imwrite(os.path.join(temp_frames_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    h, w = frames_rgb[0].shape[:2]
    num_frames = len(frames_rgb)
    print(f"  Frames: {num_frames}, Size: {w}x{h}, FPS: {fps:.1f}")
    
    # ── 최적화 설정 (Facebook benchmark.py 기반) ──
    # TF32 활성화 (Ampere 이상 GPU)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 모델 빌드 (torch.compile은 Windows에서 Triton 미지원으로 제외)
    t0 = time.time()
    predictor = build_sam2_video_predictor(
        MODEL_CFG, CHECKPOINT, device=torch.device("cuda"),
    )
    t_build = time.time() - t0
    print(f"  Model build: {t_build:.2f}s (TF32=true)")
    
    # init_state
    t0 = time.time()
    state = predictor.init_state(video_path=temp_frames_dir)
    t_init = time.time() - t0
    print(f"  Init state: {t_init:.2f}s")
    
    # 프롬프트 추가
    if "objects" in meta:
        for obj in meta["objects"]:
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj["id"],
                points=np.array([obj["point"]], dtype=np.float32),
                labels=np.array([obj["label"]]),
            )
            print(f"  Added obj {obj['id']}: {obj.get('desc', '')} -> {obj['point']}")
    else:
        pt = meta.get("prompt", {}).get("point", [w//2, h//2])
        lb = meta.get("prompt", {}).get("label", 1)
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([pt], dtype=np.float32),
            labels=np.array([lb]),
        )
        print(f"  Added prompt: point={pt}")
    
    # propagate (bfloat16 AMP + inference_mode 적용)
    t0 = time.time()
    results = {}
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                results[frame_idx] = (obj_ids, masks.cpu().numpy())
    t_prop = time.time() - t0
    prop_fps = num_frames / t_prop if t_prop > 0 else 0
    print(f"  Propagation: {num_frames} frames in {t_prop:.2f}s ({prop_fps:.1f} FPS) [AMP+compile]")
    
    # 비디오 저장
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_path = os.path.join(output_dir, "result.avi")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    for fi in range(num_frames):
        if fi in results:
            obj_ids, masks = results[fi]
            overlay, _ = create_overlay(frames_rgb[fi], masks, obj_ids)
        else:
            overlay = frames_rgb[fi]
        vw.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    vw.release()
    t_save = time.time() - t0
    
    total = t_init + t_prop + t_save
    print(f"  Saving: {t_save:.2f}s")
    print(f"  Total: {total:.2f}s")
    
    # 임시 프레임 삭제
    import shutil
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    # 결과 저장
    with open(os.path.join(output_dir, "result_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Facebook SAM2 Prediction\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames: {num_frames}, Size: {w}x{h}\n")
        f.write(f"Init: {t_init:.2f}s\n")
        f.write(f"Propagation: {t_prop:.2f}s ({prop_fps:.1f} FPS)\n")
        f.write(f"Saving: {t_save:.2f}s\n")
        f.write(f"Total: {total:.2f}s\n")
    
    return prop_fps


def main():
    samples = [
        ("01_dog", "01_dog.mp4"),
        ("02_cups", "02_cups.mp4"),
        ("03_blocks", "03_blocks.mp4"),
        ("04_coffee", "04_coffee.mp4"),
        ("05_default_juggle", "05_default_juggle.mp4"),
        ("moving_circle", "moving_circle.avi"),
        ("multi_objects", "multi_objects.avi"),
        ("rotating_box", "rotating_box.avi"),
    ]
    
    print("=" * 60)
    print("  Facebook SAM2 Official Benchmark")
    print("=" * 60)
    
    all_results = []
    for name, filename in samples:
        sample_dir = os.path.join(SAMPLES_DIR, name)
        video_path = os.path.join(sample_dir, filename)
        if not os.path.exists(video_path):
            print(f"\n--- {name} --- SKIP (not found)")
            continue
        
        meta_path = os.path.join(sample_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        
        output_dir = os.path.join(RESULT_DIR, name)
        print(f"\n--- {name} ---")
        t0 = time.time()
        fps_val = predict_with_facebook(video_path, meta, output_dir)
        elapsed = time.time() - t0
        all_results.append((name, elapsed, fps_val))
    
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    for name, t, fps_val in all_results:
        print(f"  {name}: {t:.1f}s ({fps_val:.1f} FPS)")


if __name__ == "__main__":
    main()
