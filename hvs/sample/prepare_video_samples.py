"""
비디오 샘플 준비 스크립트

■ 기능:
  1. 합성 비디오 생성 (60fps × 10초 = 600프레임)
     - moving_circle: 원형 궤적으로 이동하는 빨간 원
     - multi_objects: 서로 다른 방향으로 이동하는 3개 객체
     - rotating_box: 회전하는 사각형
  2. Facebook 데모 비디오 복사
     - 01_dog, 02_cups, 03_blocks, 04_coffee, 05_default_juggle
  3. sam2_bedroom 삭제

■ 사용법:
  python hvs/sample/prepare_video_samples.py
"""
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

SAMPLES_DIR = os.path.join(_ROOT, ".dataset", "video_samples")
FACEBOOK_GALLERY = os.path.join(_ROOT, "facebook", "demo", "data", "gallery")

FPS = 60
DURATION = 10  # seconds
TOTAL_FRAMES = FPS * DURATION  # 600
WIDTH, HEIGHT = 640, 480


def generate_moving_circle():
    """빨간 원이 원형 궤적으로 이동 (60fps, 10초)"""
    name = "moving_circle"
    out_dir = os.path.join(SAMPLES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    avi_path = os.path.join(out_dir, f"{name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(avi_path, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(TOTAL_FRAMES):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 30  # 어두운 배경

        # 배경 그라데이션
        for y in range(HEIGHT):
            frame[y, :, 0] = min(30 + y // 8, 60)
            frame[y, :, 1] = min(30 + y // 10, 50)
            frame[y, :, 2] = min(40 + y // 6, 70)

        # 빨간 원: 원형 궤적
        t = i / TOTAL_FRAMES * 2 * math.pi * 3  # 3바퀴
        cx = int(WIDTH // 2 + 150 * math.cos(t))
        cy = int(HEIGHT // 2 + 100 * math.sin(t))
        cv2.circle(frame, (cx, cy), 35, (0, 0, 220), -1)
        cv2.circle(frame, (cx, cy), 35, (50, 50, 255), 2)

        # 그림자
        cv2.circle(frame, (cx + 3, cy + 3), 35, (15, 15, 15), -1)
        cv2.circle(frame, (cx, cy), 35, (0, 0, 220), -1)

        writer.write(frame)

    writer.release()

    # 메타 정보
    meta = {
        "name": name,
        "description": "빨간 원이 원형 궤적으로 이동",
        "fps": FPS,
        "frames": TOTAL_FRAMES,
        "size": [WIDTH, HEIGHT],
        "prompt": {"point": [WIDTH // 2 + 150, HEIGHT // 2], "label": 1},
        "detect_method": "red",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [OK] {name}: {avi_path} ({os.path.getsize(avi_path) // 1024}KB)")


def generate_multi_objects():
    """3개 객체(원, 사각형, 삼각형)가 서로 다른 방향으로 이동 (60fps, 10초)"""
    name = "multi_objects"
    out_dir = os.path.join(SAMPLES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    avi_path = os.path.join(out_dir, f"{name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(avi_path, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(TOTAL_FRAMES):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 25

        # 격자 배경
        for x in range(0, WIDTH, 40):
            cv2.line(frame, (x, 0), (x, HEIGHT), (35, 35, 35), 1)
        for y in range(0, HEIGHT, 40):
            cv2.line(frame, (0, y), (WIDTH, y), (35, 35, 35), 1)

        t = i / TOTAL_FRAMES

        # 객체 1: 빨간 원 (좌→우 직선)
        cx1 = int(50 + t * (WIDTH - 100))
        cy1 = int(HEIGHT // 4 + 30 * math.sin(t * 2 * math.pi * 5))
        cv2.circle(frame, (cx1, cy1), 30, (0, 0, 200), -1)

        # 객체 2: 녹색 사각형 (대각선 바운스)
        cx2 = int(abs(((i * 2) % (2 * WIDTH)) - WIDTH))
        cy2 = int(abs(((i * 3) % (2 * HEIGHT)) - HEIGHT))
        cx2 = max(30, min(cx2, WIDTH - 30))
        cy2 = max(30, min(cy2, HEIGHT - 30))
        cv2.rectangle(frame, (cx2 - 25, cy2 - 25), (cx2 + 25, cy2 + 25), (0, 180, 0), -1)

        # 객체 3: 파란 삼각형 (원형 궤적, 반대 방향)
        angle = -t * 2 * math.pi * 2
        cx3 = int(WIDTH // 2 + 120 * math.cos(angle))
        cy3 = int(HEIGHT * 3 // 4 + 60 * math.sin(angle))
        pts = np.array([
            [cx3, cy3 - 25],
            [cx3 - 22, cy3 + 20],
            [cx3 + 22, cy3 + 20],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (200, 100, 0))

        writer.write(frame)

    writer.release()

    meta = {
        "name": name,
        "description": "빨간원+녹색사각형+파란삼각형 3객체 이동",
        "fps": FPS,
        "frames": TOTAL_FRAMES,
        "size": [WIDTH, HEIGHT],
        "objects": [
            {"id": 1, "point": [50, HEIGHT // 4], "label": 1, "desc": "빨간 원"},
            {"id": 2, "point": [WIDTH // 2, HEIGHT // 2], "label": 1, "desc": "녹색 사각형"},
            {"id": 3, "point": [WIDTH // 2 + 120, HEIGHT * 3 // 4], "label": 1, "desc": "파란 삼각형"},
        ],
        "detect_method": "center",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [OK] {name}: {avi_path} ({os.path.getsize(avi_path) // 1024}KB)")


def generate_rotating_box():
    """회전하는 사각형 + 크기 변화 (60fps, 10초)"""
    name = "rotating_box"
    out_dir = os.path.join(SAMPLES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    avi_path = os.path.join(out_dir, f"{name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(avi_path, fourcc, FPS, (WIDTH, HEIGHT))

    cx, cy = WIDTH // 2, HEIGHT // 2

    for i in range(TOTAL_FRAMES):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 20

        # 방사형 배경
        for r in range(0, max(WIDTH, HEIGHT), 60):
            cv2.circle(frame, (cx, cy), r, (25, 25, 30), 1)

        t = i / TOTAL_FRAMES
        angle = t * 360 * 4  # 4바퀴 회전
        size = int(40 + 30 * math.sin(t * 2 * math.pi * 3))  # 크기 맥동

        # 회전 사각형
        rect = ((cx, cy), (size * 2, size * 2), angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(frame, [box], (0, 140, 200))
        cv2.polylines(frame, [box], True, (0, 180, 255), 2)

        # 중심 점
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        writer.write(frame)

    writer.release()

    meta = {
        "name": name,
        "description": "중앙에서 회전+크기변화하는 사각형",
        "fps": FPS,
        "frames": TOTAL_FRAMES,
        "size": [WIDTH, HEIGHT],
        "prompt": {"point": [cx, cy], "label": 1},
        "detect_method": "center",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [OK] {name}: {avi_path} ({os.path.getsize(avi_path) // 1024}KB)")


def copy_facebook_demos():
    """Facebook 데모 비디오를 .dataset/video_samples/에 복사"""
    demos = [
        ("01_dog.mp4", "red", "개와 공놀이", [461, 432]),
        ("02_cups.mp4", "center", "컵 쌓기 - 노란색 공 추적", [430, 530]),
        ("03_blocks.mp4", "center", "블록 놀이 - T 알파벳 블록 추적", [840, 250]),
        ("04_coffee.mp4", "center", "커피 제조 - 양손 추적", None),  # 다중 객체
        ("05_default_juggle.mp4", "center", "저글링 - 축구공 추적", [560, 390]),
    ]

    for filename, method, desc, point in demos:
        src = os.path.join(FACEBOOK_GALLERY, filename)
        name = os.path.splitext(filename)[0]
        dst_dir = os.path.join(SAMPLES_DIR, name)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)

        if not os.path.exists(src):
            print(f"  [SKIP] {filename}: not found at {src}")
            continue

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        # 비디오 정보
        cap = cv2.VideoCapture(dst)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        meta = {
            "name": name,
            "description": desc,
            "fps": fps,
            "frames": total,
            "size": [w, h],
            "detect_method": method,
        }
        if point:
            meta["prompt"] = {"point": point, "label": 1}
        else:
            meta["prompt"] = {"point": [w // 2, h // 2], "label": 1}

        with open(os.path.join(dst_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        sz = os.path.getsize(dst) // 1024
        print(f"  [OK] {name}: {total} frames, {w}x{h}, {fps:.0f}FPS ({sz}KB)")


def remove_old_samples():
    """sam2_bedroom 삭제"""
    old = os.path.join(SAMPLES_DIR, "sam2_bedroom")
    if os.path.exists(old):
        shutil.rmtree(old)
        print(f"  [DEL] sam2_bedroom")


def prepare_all():
    """전체 비디오 샘플 준비"""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("=== 1. 기존 샘플 정리 ===")
    remove_old_samples()

    print("\n=== 2. 합성 비디오 생성 (60fps × 10초) ===")
    generate_moving_circle()
    generate_multi_objects()
    generate_rotating_box()

    print("\n=== 3. Facebook 데모 비디오 복사 ===")
    copy_facebook_demos()

    print(f"\n{'='*50}")
    print(f"  전체 샘플 준비 완료: {SAMPLES_DIR}")
    entries = sorted(os.listdir(SAMPLES_DIR))
    for e in entries:
        full = os.path.join(SAMPLES_DIR, e)
        if os.path.isdir(full):
            files = os.listdir(full)
            print(f"   {e}/ ({len(files)} files)")
    print()


if __name__ == "__main__":
    prepare_all()
