"""
비디오 예측 샘플 스크립트 (Video Prediction Sample)

■ 사용법:
  python hvs/sample/predict_video_sample.py
  python hvs/sample/predict_video_sample.py --video facebook/demo/data/gallery/01_dog.mp4
  python hvs/sample/predict_video_sample.py --video input.avi --point 320,240
  python hvs/sample/predict_video_sample.py --all

■ 기능:
  1. .avi/.mp4 비디오 파일 입력
  2. 첫 프레임에서 HSV 색 감지 또는 지정 좌표로 프롬프트
  3. 전체 프레임 전파 + 결과 비디오(AVI) 저장
  4. 프레임별 오버레이/마스크 이미지 저장

■ 출력:
  .dataset/results/video/{name}/
  ├── result.avi          — 마스크 오버레이 비디오
  ├── result_mask.avi     — 이진 마스크 비디오
  ├── frame_0000.jpg      — 샘플 프레임
  └── result_info.txt     — 추적 요약
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

CKPT_PATH = os.path.join(_ROOT, "facebook", "checkpoints", "sam2.1_hiera_tiny.pt")
SAMPLES_DIR = os.path.join(_ROOT, ".dataset", "video_samples")
RESULT_DIR = os.path.join(_ROOT, ".dataset", "results", "video")


def detect_prominent_object(frame_rgb, method="center"):
    """
    첫 프레임에서 프롬프트 좌표 자동 감지

    Args:
        frame_rgb: (H, W, 3) RGB numpy
        method: "center" | "red" | "bright"

    Returns:
        (x, y) 좌표
    """
    h, w = frame_rgb.shape[:2]

    if method == "red":
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = (cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) |
                cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    return w // 2, h // 2


def predict_single_video(
    video_path,
    output_dir=None,
    point=None,
    detect_method="center",
    max_frames=None,
    checkpoint_path=CKPT_PATH,
    save_every_frame=True,
):
    """단일 비디오 예측"""
    import torch
    from hvs.predictor.video_predictor import VideoPredictor
    from hvs.scripts.predict_video import create_overlay_frame

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(RESULT_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # 비디오 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    limit = max_frames or total
    while len(frames) < limit:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"  Video: {video_name}")
    print(f"  Size: {w}x{h}, {len(frames)}/{total} frames, {fps:.0f} FPS")

    # 프롬프트 좌표
    if point:
        px, py = point
    else:
        px, py = detect_prominent_object(frames[0], detect_method)
    print(f"  Prompt: ({px}, {py}) [{detect_method}]")

    # Predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = VideoPredictor(
        model_size="tiny", image_size=1024, device=device,
        checkpoint_path=checkpoint_path, init_mode="finetune",
    )
    print(f"  Device: {device}")

    t0 = time.time()
    state = predictor.init_state(frames)
    predictor.add_points(
        state, frame_idx=0, obj_id=1,
        points=np.array([[px, py]], dtype=np.float32),
        labels=np.array([1]),
    )
    t_init = time.time() - t0

    t0 = time.time()
    results = list(predictor.propagate(state))
    t_prop = time.time() - t0
    prop_fps = len(results) / t_prop if t_prop > 0 else 0
    print(f"  Propagation: {len(results)} frames, {t_prop:.1f}s ({prop_fps:.2f} FPS)")

    # 결과 비디오 저장
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(os.path.join(output_dir, 'result.avi'), fourcc, fps, (w, h))
    vw_mask = cv2.VideoWriter(os.path.join(output_dir, 'result_mask.avi'), fourcc, fps, (w, h), isColor=False)

    info_lines = [
        f"SAM2 Video Prediction: {video_name}",
        "=" * 50,
        f"Input: {video_path}",
        f"Frames: {len(frames)}/{total}, Size: {w}x{h}, FPS: {fps:.0f}",
        f"Prompt: ({px}, {py}) [{detect_method}]",
        f"Device: {device}",
        f"Init: {t_init:.2f}s, Propagation: {t_prop:.1f}s ({prop_fps:.2f} FPS)",
        "",
    ]

    for frame_idx, obj_ids, mask_data in results:
        overlay, binary = create_overlay_frame(frames[frame_idx], mask_data, frame_idx, obj_ids)
        vw.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        vw_mask.write((binary * 255).astype(np.uint8))
        area = int(binary.sum())
        info_lines.append(f"Frame {frame_idx}: area={area}")

        if save_every_frame or frame_idx % 30 == 0 or frame_idx == len(results) - 1:
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_idx:04d}.jpg'),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if frame_idx % 30 == 0 or frame_idx == len(results) - 1:
            print(f"    Frame {frame_idx:3d}: area={area}")

    vw.release()
    vw_mask.release()

    with open(os.path.join(output_dir, 'result_info.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(info_lines))

    sz = os.path.getsize(os.path.join(output_dir, 'result.avi')) // 1024
    print(f"  Result: {os.path.join(output_dir, 'result.avi')} ({sz}KB)")
    return output_dir


def predict_all_samples(max_frames=None):
    """
    .dataset/video_samples/ 내의 모든 비디오 예측

    Facebook 데모 비디오 + 합성 비디오 전부 수행
    """
    if not os.path.exists(SAMPLES_DIR):
        print(f"Samples not found: {SAMPLES_DIR}")
        print("Run: python hvs/sample/prepare_video_samples.py")
        return

    # 비디오 파일 + 디렉토리 수집
    entries = sorted(os.listdir(SAMPLES_DIR))
    videos = []

    for entry in entries:
        full = os.path.join(SAMPLES_DIR, entry)
        if os.path.isfile(full) and entry.lower().endswith(('.mp4', '.avi')):
            videos.append(full)
        elif os.path.isdir(full):
            # 디렉토리 안의 비디오 파일 탐색 (mp4 > avi 우선)
            vid_files = sorted([
                f for f in os.listdir(full)
                if f.lower().endswith(('.mp4', '.avi'))
            ])
            if vid_files:
                videos.append(os.path.join(full, vid_files[0]))
            else:
                # 프레임 이미지 디렉토리로 간주
                videos.append(full)

    print(f"\n{'='*60}")
    print(f"  Predicting {len(videos)} videos")
    print(f"{'='*60}\n")

    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        # 상위 디렉토리명을 결과 디렉토리명으로 사용
        parent_name = os.path.basename(os.path.dirname(v))
        if parent_name != "video_samples":
            out_name = parent_name
        else:
            out_name = name

        # meta.json에서 프롬프트/감지 방법 로드
        meta_path = os.path.join(os.path.dirname(v), 'meta.json')
        method = "center"
        pt = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as mf:
                meta = json.load(mf)
            method = meta.get("detect_method", "center")
            if "prompt" in meta:
                pt = tuple(meta["prompt"]["point"])

        out_dir = os.path.join(RESULT_DIR, out_name)
        print(f"\n--- {out_name} ---")
        predict_single_video(v, output_dir=out_dir, point=pt,
                             detect_method=method, max_frames=max_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Video Prediction Sample")
    parser.add_argument("--video", default=None, help="비디오 파일 경로")
    parser.add_argument("--output", default=None, help="결과 저장 디렉토리")
    parser.add_argument("--point", default=None, help="프롬프트 좌표 'x,y'")
    parser.add_argument("--detect", default="center", choices=["center", "red"],
                        help="객체 감지 방법")
    parser.add_argument("--max_frames", type=int, default=None, help="최대 프레임 수")
    parser.add_argument("--all", action="store_true", help="모든 샘플 비디오 예측")
    args = parser.parse_args()

    if args.all:
        predict_all_samples(max_frames=args.max_frames)
    elif args.video:
        pt = None
        if args.point:
            parts = args.point.split(",")
            pt = (int(parts[0]), int(parts[1]))
        predict_single_video(
            args.video, output_dir=args.output, point=pt,
            detect_method=args.detect, max_frames=args.max_frames,
        )
    else:
        parser.print_help()
