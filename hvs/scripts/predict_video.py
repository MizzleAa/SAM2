"""
비디오 예측 스크립트 (Video Prediction)

■ 사용법:
  # .avi / .mp4 비디오 파일 입력
  python hvs/scripts/predict_video.py --video input.avi
  python hvs/scripts/predict_video.py --video input.mp4 --point 320,240

  # 프레임 디렉토리 입력 (기존 호환)
  python hvs/scripts/predict_video.py --video_dir .dataset/video_samples/moving_circle

  # 모든 샘플 비디오
  python hvs/scripts/predict_video.py --all

■ 출력:
  .dataset/results/video/{video_name}/
  ├── result.avi                     — 마스크 오버레이 비디오
  ├── result_mask.avi                — 이진 마스크 비디오
  ├── frame_00000_overlay.jpg        — 프레임별 이미지 (선택)
  └── result_info.txt                — 추적 결과 요약
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# SAM2 루트를 path에 추가
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _ROOT)

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"
VIDEO_SAMPLES_DIR = os.path.join(_ROOT, ".dataset", "video_samples")
RESULT_DIR = os.path.join(_ROOT, ".dataset", "results", "video")

MASK_COLORS = [
    (0, 200, 100),   # Green
    (200, 100, 0),    # Orange
    (100, 0, 200),    # Purple
    (200, 200, 0),    # Yellow
]


def load_video_file(video_path: str, max_frames: int = None):
    """
    비디오 파일(.avi, .mp4 등)을 프레임 리스트로 로드

    Args:
        video_path: 비디오 파일 경로
        max_frames: 최대 프레임 수 (None이면 전체)

    Returns:
        frames: [(H, W, 3) numpy] 프레임 리스트
        fps: 원본 FPS
        video_info: 비디오 메타 정보
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()

    video_info = {
        "path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "loaded_frames": len(frames),
        "width": width,
        "height": height,
    }
    return frames, fps, video_info


def create_overlay_frame(frame, mask_data, frame_idx, obj_ids):
    """
    프레임에 마스크 오버레이 + 경계선 + 정보 표시

    Args:
        frame: (H, W, 3) 원본 프레임
        mask_data: (num_obj, ..., H, W) 마스크 logits
        frame_idx: 프레임 인덱스
        obj_ids: 객체 ID 리스트

    Returns:
        overlay: (H, W, 3) 오버레이 프레임
        combined_binary: (H, W) bool 통합 마스크
    """
    overlay = frame.copy()

    # 마스크 정리
    if isinstance(mask_data, np.ndarray):
        if mask_data.ndim == 2:
            all_masks = [mask_data]
        elif mask_data.ndim == 3:
            all_masks = [mask_data[i] for i in range(mask_data.shape[0])]
        elif mask_data.ndim == 4:
            all_masks = [mask_data[i].squeeze() for i in range(mask_data.shape[0])]
        else:
            all_masks = [mask_data.squeeze()]
    else:
        return overlay, np.zeros(frame.shape[:2], dtype=bool)

    combined_binary = np.zeros(frame.shape[:2], dtype=bool)

    for mi, single_mask in enumerate(all_masks):
        if single_mask.ndim > 2:
            single_mask = single_mask.squeeze()
        binary = (single_mask > 0).astype(bool)
        combined_binary |= binary

        color = MASK_COLORS[mi % len(MASK_COLORS)]
        overlay[binary] = (
            0.5 * overlay[binary].astype(float) +
            0.5 * np.array(color)
        ).astype(np.uint8)

        # 경계선
        edge_img = Image.fromarray((binary * 255).astype(np.uint8))
        edge = np.array(edge_img.filter(ImageFilter.FIND_EDGES)) > 128
        overlay[edge] = color

    # 프레임 정보 표시
    pil_overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_overlay)
    area = int(combined_binary.sum())
    draw.text((10, 10), f"Frame {frame_idx}", fill=(255, 255, 255))
    draw.text((10, 30), f"Area: {area}px", fill=(255, 255, 0))

    return np.array(pil_overlay), combined_binary


def predict_video(
    video_path: str = None,
    video_dir: str = None,
    output_dir: str = None,
    checkpoint_path: str = CKPT_PATH,
    point: str = None,
    max_frames: int = None,
    save_frames: bool = False,
):
    """
    비디오 예측 + 결과 비디오 저장

    Args:
        video_path: .avi/.mp4 비디오 파일 경로
        video_dir: 프레임 이미지 디렉토리 (기존 호환)
        output_dir: 결과 저장 경로
        checkpoint_path: 체크포인트 경로
        point: 프롬프트 점 좌표 "x,y" (None이면 중앙)
        max_frames: 최대 프레임 수
        save_frames: 프레임별 이미지도 저장할지
    """
    from hvs.predictor.video_predictor import VideoPredictor

    # ── 1. 프레임 로드 ──
    fps = 30.0
    video_info = {}

    if video_path and os.path.isfile(video_path):
        # .avi / .mp4 비디오 파일
        frames, fps, video_info = load_video_file(video_path, max_frames)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"  Video file: {video_path}")
        print(f"  Loaded: {len(frames)} frames, {video_info['width']}x{video_info['height']}, {fps:.1f} FPS")
    elif video_dir and os.path.isdir(video_dir):
        # 프레임 디렉토리
        video_name = os.path.basename(video_dir)
        frame_files = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) and "mask" not in f.lower()
        ])
        frames = []
        for i, f in enumerate(frame_files):
            if max_frames and i >= max_frames:
                break
            img = np.array(Image.open(os.path.join(video_dir, f)).convert("RGB"))
            frames.append(img)
        print(f"  Video dir: {video_dir}")
        print(f"  Loaded: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
    else:
        # 기본: 첫 번째 샘플 비디오
        if os.path.exists(VIDEO_SAMPLES_DIR):
            videos = sorted(os.listdir(VIDEO_SAMPLES_DIR))
            if videos:
                video_dir = os.path.join(VIDEO_SAMPLES_DIR, videos[0])
                return predict_video(video_dir=video_dir, output_dir=output_dir,
                                    checkpoint_path=checkpoint_path, point=point)
        print("  비디오 파일 또는 디렉토리를 지정하세요.")
        return None

    if not frames:
        print("  프레임이 없습니다.")
        return None

    # ── 2. 출력 디렉토리 ──
    if output_dir is None:
        output_dir = os.path.join(RESULT_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── 3. 프롬프트 설정 ──
    h, w = frames[0].shape[:2]
    prompt_point = [w // 2, h // 2]
    prompt_label = 1
    meta = {}

    # CLI에서 point 지정
    if point:
        parts = point.split(",")
        prompt_point = [int(parts[0]), int(parts[1])]

    # meta.json에서 프롬프트 로드 (비디오 파일 또는 프레임 디렉토리)
    meta_search_dirs = []
    if video_path:
        meta_search_dirs.append(os.path.dirname(os.path.abspath(video_path)))
    if video_dir:
        meta_search_dirs.append(video_dir)
    for meta_dir in meta_search_dirs:
        meta_path = os.path.join(meta_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            break

    # ── 4. Predictor 초기화 ──
    use_ckpt = os.path.exists(checkpoint_path)
    print(f"  Checkpoint: {'loaded' if use_ckpt else 'scratch'}")

    predictor = VideoPredictor(
        model_size="tiny",
        image_size=1024 if use_ckpt else 256,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        checkpoint_path=checkpoint_path if use_ckpt else None,
        init_mode="finetune" if use_ckpt else "scratch",
    )
    print(f"  Device: {predictor.device}")

    # ── 5. init_state + add_points ──
    t0 = time.time()
    state = predictor.init_state(frames, async_loading_frames=True)

    if "objects" in meta:
        for obj in meta["objects"]:
            predictor.add_points(
                state, frame_idx=0, obj_id=obj["id"],
                points=np.array([obj["point"]], dtype=np.float32),
                labels=np.array([obj["label"]]),
            )
            print(f"  Added obj {obj['id']}: point={obj['point']}")
    else:
        if "prompt" in meta:
            prompt_point = meta["prompt"]["point"]
            prompt_label = meta["prompt"]["label"]
        predictor.add_points(
            state, frame_idx=0, obj_id=1,
            points=np.array([prompt_point], dtype=np.float32),
            labels=np.array([prompt_label]),
        )
        print(f"  Added prompt: point={prompt_point}")

    t_init = time.time() - t0
    print(f"  Init: {t_init:.2f}s")

    # ── 6. 전파 (인코딩+추적 통합) ──
    t0 = time.time()
    results = list(predictor.propagate(state))
    t_prop = time.time() - t0
    prop_fps = len(results) / t_prop if t_prop > 0 else 0
    print(f"  Inference: {len(results)} frames in {t_prop:.2f}s ({prop_fps:.1f} FPS)")

    # ── 7. 비디오 결과 생성 ──
    print(f"  Generating output video...")

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_video_path = os.path.join(output_dir, "result.avi")
    out_mask_path = os.path.join(output_dir, "result_mask.avi")
    video_writer = None
    mask_writer = None

    info_lines = [
        f"SAM2 Video Prediction: {video_name}",
        "=" * 50,
        f"Input: {video_path or video_dir}",
        f"Frames: {len(frames)}, Size: {w}x{h}",
        f"FPS: {fps:.1f}",
        f"Checkpoint: {'loaded' if use_ckpt else 'scratch'}",
        f"Device: {predictor.device}",
        f"Inference: {t_prop:.2f}s ({prop_fps:.1f} FPS)",
        "",
    ]

    t_save = time.time()
    for frame_idx, obj_ids, mask_data in results:
        frame = frames[frame_idx]

        # 오버레이 생성
        overlay, combined_binary = create_overlay_frame(frame, mask_data, frame_idx, obj_ids)

        # 비디오 라이터 초기화 (첫 프레임에서)
        if video_writer is None:
            oh, ow = overlay.shape[:2]
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (ow, oh))
            mask_writer = cv2.VideoWriter(out_mask_path, fourcc, fps, (ow, oh), isColor=False)

        # RGB → BGR (OpenCV)
        video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 마스크 비디오 (그레이스케일)
        mask_frame = (combined_binary * 255).astype(np.uint8)
        mask_writer.write(mask_frame)

        # 프레임별 이미지 저장 (옵션)
        if save_frames:
            Image.fromarray(overlay).save(
                os.path.join(output_dir, f"frame_{frame_idx:05d}_overlay.jpg")
            )
            Image.fromarray(mask_frame).save(
                os.path.join(output_dir, f"frame_{frame_idx:05d}_mask.png")
            )

        area = int(combined_binary.sum())
        info_lines.append(f"Frame {frame_idx}: objs={obj_ids}, area={area}")

    t_save = time.time() - t_save
    t_total = t_init + t_prop + t_save
    print(f"  Saving: {t_save:.2f}s")
    print(f"  Pipeline total: {t_total:.2f}s")
    info_lines.insert(11, f"Saving: {t_save:.2f}s")
    info_lines.insert(12, f"Pipeline total: {t_total:.2f}s")

    # 비디오 라이터 종료
    if video_writer:
        video_writer.release()
    if mask_writer:
        mask_writer.release()

    # 결과 요약 저장
    with open(os.path.join(output_dir, "result_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))

    print(f"\n  === Results ===")
    print(f"  Overlay video: {out_video_path}")
    print(f"  Mask video:    {out_mask_path}")
    print(f"  Info:          {os.path.join(output_dir, 'result_info.txt')}")
    return output_dir


def predict_all_videos():
    """모든 샘플 비디오에 대해 예측"""
    if not os.path.exists(VIDEO_SAMPLES_DIR):
        print("  No video samples found. Generating...")
        from hvs.scripts.prepare_video_samples import prepare_all
        prepare_all()

    videos = sorted([
        d for d in os.listdir(VIDEO_SAMPLES_DIR)
        if os.path.isdir(os.path.join(VIDEO_SAMPLES_DIR, d))
    ])

    print(f"\n=== Predicting {len(videos)} videos ===\n")
    for v in videos:
        video_dir = os.path.join(VIDEO_SAMPLES_DIR, v)
        print(f"\n--- {v} ---")
        predict_video(video_dir=video_dir)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Video Prediction")
    parser.add_argument("--video", default=None, help="비디오 파일 경로 (.avi, .mp4)")
    parser.add_argument("--video_dir", default=None, help="프레임 이미지 디렉토리")
    parser.add_argument("--output", default=None, help="결과 저장 디렉토리")
    parser.add_argument("--point", default=None, help="프롬프트 좌표 'x,y'")
    parser.add_argument("--max_frames", type=int, default=None, help="최대 프레임 수")
    parser.add_argument("--save_frames", action="store_true", help="프레임별 이미지도 저장")
    parser.add_argument("--all", action="store_true", help="모든 샘플 비디오 예측")
    args = parser.parse_args()

    if args.all:
        predict_all_videos()
    else:
        predict_video(
            video_path=args.video,
            video_dir=args.video_dir,
            output_dir=args.output,
            point=args.point,
            max_frames=args.max_frames,
            save_frames=args.save_frames,
        )
