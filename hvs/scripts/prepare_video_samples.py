"""
샘플 비디오 데이터 생성 스크립트

■ 목적:
  .dataset/video_samples/ 에 3개의 샘플 비디오 시퀀스를 생성합니다.
  실제 비디오 파일이 없어도 VideoPredictor를 테스트할 수 있습니다.

■ 생성되는 비디오:
  1. moving_circle  — 원이 좌→우 이동 (10프레임, 640×480)
  2. rotating_box   — 사각형이 회전 (8프레임, 640×480)
  3. multi_objects   — 3개 객체 이동 (12프레임, 640×480)

사용법:
  python hvs/scripts/prepare_video_samples.py
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "video_samples")


def create_moving_circle(output_dir: str, num_frames: int = 10, w: int = 640, h: int = 480):
    """이동하는 원 비디오"""
    video_dir = os.path.join(output_dir, "moving_circle")
    os.makedirs(video_dir, exist_ok=True)

    gt_masks_dir = os.path.join(video_dir, "gt_masks")
    os.makedirs(gt_masks_dir, exist_ok=True)

    for i in range(num_frames):
        # 배경 (그라데이션)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            img[y, :] = [30 + y // 10, 20, 40 + y // 10]

        # 원 위치 (좌→우)
        cx = int(80 + (w - 160) * i / (num_frames - 1))
        cy = h // 2
        radius = 50

        # 원 그리기
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist < radius
        img[mask] = [220, 140, 60]  # 주황

        # 테두리
        edge = (dist >= radius - 3) & (dist < radius)
        img[edge] = [255, 200, 100]

        # 저장
        Image.fromarray(img).save(os.path.join(video_dir, f"frame_{i:05d}.jpg"))
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(gt_masks_dir, f"mask_{i:05d}.png")
        )

    # 메타 정보
    meta = {
        "name": "moving_circle",
        "num_frames": num_frames,
        "size": [w, h],
        "description": "원이 좌측에서 우측으로 이동",
        "prompt": {"point": [80, cy], "label": 1},
    }
    import json
    with open(os.path.join(video_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [1] moving_circle: {num_frames} frames, {w}x{h}")
    return video_dir


def create_rotating_box(output_dir: str, num_frames: int = 8, w: int = 640, h: int = 480):
    """회전하는 사각형 비디오"""
    video_dir = os.path.join(output_dir, "rotating_box")
    os.makedirs(video_dir, exist_ok=True)

    gt_masks_dir = os.path.join(video_dir, "gt_masks")
    os.makedirs(gt_masks_dir, exist_ok=True)

    cx, cy = w // 2, h // 2
    size = 80

    for i in range(num_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 25
        # 배경 노이즈
        noise = np.random.randint(0, 10, (h, w, 3), dtype=np.uint8)
        img = img + noise

        # 회전 각도
        angle = i * 360 / num_frames
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # 회전된 사각형 꼭짓점 계산
        rad = math.radians(angle)
        corners = []
        for dx, dy in [(-size, -size), (size, -size), (size, size), (-size, size)]:
            rx = dx * math.cos(rad) - dy * math.sin(rad) + cx
            ry = dx * math.sin(rad) + dy * math.cos(rad) + cy
            corners.append((rx, ry))

        draw.polygon(corners, fill=(100, 180, 220), outline=(150, 220, 255))
        img = np.array(pil_img)

        # GT 마스크 (폴리곤 내부)
        mask_img = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.polygon(corners, fill=255)
        mask = np.array(mask_img)

        Image.fromarray(img).save(os.path.join(video_dir, f"frame_{i:05d}.jpg"))
        Image.fromarray(mask).save(os.path.join(gt_masks_dir, f"mask_{i:05d}.png"))

    import json
    meta = {
        "name": "rotating_box",
        "num_frames": num_frames,
        "size": [w, h],
        "description": "사각형이 중심에서 회전",
        "prompt": {"point": [cx, cy], "label": 1},
    }
    with open(os.path.join(video_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [2] rotating_box: {num_frames} frames, {w}x{h}")
    return video_dir


def create_multi_objects(output_dir: str, num_frames: int = 12, w: int = 640, h: int = 480):
    """3개 객체 이동 비디오"""
    video_dir = os.path.join(output_dir, "multi_objects")
    os.makedirs(video_dir, exist_ok=True)

    gt_masks_dir = os.path.join(video_dir, "gt_masks")
    os.makedirs(gt_masks_dir, exist_ok=True)

    objects = [
        {"color": [220, 80, 60], "radius": 35, "y_center": h // 4, "direction": 1},
        {"color": [60, 180, 120], "radius": 45, "y_center": h // 2, "direction": -1},
        {"color": [80, 100, 220], "radius": 30, "y_center": 3 * h // 4, "direction": 1},
    ]

    for i in range(num_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 20
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        yy, xx = np.ogrid[:h, :w]

        for oid, obj in enumerate(objects):
            if obj["direction"] > 0:
                cx = int(60 + (w - 120) * i / (num_frames - 1))
            else:
                cx = int(w - 60 - (w - 120) * i / (num_frames - 1))

            cy = obj["y_center"]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = dist < obj["radius"]

            img[mask] = obj["color"]
            combined_mask[mask] = (oid + 1) * 80  # 80, 160, 240

        Image.fromarray(img).save(os.path.join(video_dir, f"frame_{i:05d}.jpg"))
        Image.fromarray(combined_mask).save(os.path.join(gt_masks_dir, f"mask_{i:05d}.png"))

    import json
    meta = {
        "name": "multi_objects",
        "num_frames": num_frames,
        "size": [w, h],
        "description": "3개 객체가 좌우로 이동",
        "objects": [
            {"id": 1, "point": [60, h // 4], "label": 1, "color": "red"},
            {"id": 2, "point": [w - 60, h // 2], "label": 1, "color": "green"},
            {"id": 3, "point": [60, 3 * h // 4], "label": 1, "color": "blue"},
        ],
    }
    with open(os.path.join(video_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [3] multi_objects: {num_frames} frames, {w}x{h}, 3 objects")
    return video_dir


def copy_sam2_bedroom(output_dir: str, max_frames: int = 10):
    """
    Facebook SAM2 공식 bedroom 비디오를 .dataset에 복사

    200프레임 중 max_frames개만 균등 간격으로 선택합니다.
    """
    import json
    import shutil

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "facebook", "notebooks", "videos", "bedroom"
    )
    if not os.path.exists(src_dir):
        print(f"  [4] bedroom: SKIP (source not found)")
        return None

    video_dir = os.path.join(output_dir, "sam2_bedroom")
    os.makedirs(video_dir, exist_ok=True)

    # 원본 프레임 목록
    all_frames = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
    total = len(all_frames)

    # 균등 간격 선택
    step = max(1, total // max_frames)
    selected = all_frames[::step][:max_frames]

    for idx, fname in enumerate(selected):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(video_dir, f"frame_{idx:05d}.jpg")
        shutil.copy2(src, dst)

    # 첫 프레임 크기 확인
    first = Image.open(os.path.join(video_dir, "frame_00000.jpg"))
    w, h = first.size

    # 메타 정보 — Facebook 원본 프롬프트: 아이 위치 [400, 150]
    meta = {
        "name": "sam2_bedroom",
        "source": "Facebook SAM2 official sample",
        "num_frames": len(selected),
        "total_original_frames": total,
        "size": [w, h],
        "description": "SAM2 공식 bedroom 비디오 - 아이 추적 (Facebook 원본 프롬프트)",
        "prompt": {"point": [400, 150], "label": 1},
    }
    with open(os.path.join(video_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [4] sam2_bedroom: {len(selected)}/{total} frames, {w}x{h} (SAM2 공식)")
    return video_dir


def prepare_all():
    """전체 샘플 비디오 생성"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== Preparing video samples ===")
    print(f"  Output: {OUTPUT_DIR}\n")

    create_moving_circle(OUTPUT_DIR)
    create_rotating_box(OUTPUT_DIR)
    create_multi_objects(OUTPUT_DIR)
    copy_sam2_bedroom(OUTPUT_DIR)

    count = len([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    print(f"\n  Done! {count} video samples in {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    prepare_all()
