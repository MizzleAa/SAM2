"""
이미지 예측 예시 스크립트

■ 사용법:
  python hvs/scripts/predict_image.py
  python hvs/scripts/predict_image.py --image path/to/image.jpg --point 320,240

■ 기능:
  1. COCO 또는 지정 이미지에서 마스크 예측
  2. 결과를 시각화하여 .dataset/results/image/ 에 저장
  3. 점 프롬프트 + 마스크 오버레이를 육안 확인 가능

■ 출력 파일:
  .dataset/results/image/
  ├── input.jpg           — 원본 이미지
  ├── prompt.jpg          — 프롬프트 점 표시
  ├── mask_overlay.jpg    — 마스크 오버레이
  ├── mask_binary.png     — 이진 마스크
  └── result_info.txt     — 예측 결과 정보
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

# SAM2 루트를 path에 추가
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _ROOT)

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "results", "image")
COCO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini", "images")


def predict_image(
    image_path: str = None,
    point: tuple = None,
    output_dir: str = RESULT_DIR,
    checkpoint_path: str = CKPT_PATH,
):
    """
    이미지 예측 + 결과 저장

    Args:
        image_path: 입력 이미지 경로 (None이면 COCO에서 선택)
        point: (x, y) 프롬프트 점
        output_dir: 결과 저장 경로
    """
    from hvs.predictor.image_predictor import ImagePredictor

    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    if image_path and os.path.exists(image_path):
        image = np.array(Image.open(image_path).convert("RGB"))
        img_name = os.path.basename(image_path)
    elif os.path.exists(COCO_DIR):
        files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
        image_path = os.path.join(COCO_DIR, files[0])
        image = np.array(Image.open(image_path).convert("RGB"))
        img_name = files[0]
    else:
        # 합성 이미지
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:380, 150:490] = [200, 150, 80]
        img_name = "synthetic.jpg"

    h, w = image.shape[:2]
    print(f"  Image: {img_name} ({w}x{h})")

    # 프롬프트 점
    if point is None:
        point = (w // 2, h // 2)
    px, py = point

    # 1. 원본 저장
    Image.fromarray(image).save(os.path.join(output_dir, "input.jpg"))

    # 2. 프롬프트 점 표시
    prompt_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(prompt_img)
    r = 8
    draw.ellipse([px-r, py-r, px+r, py+r], fill=(0, 255, 0), outline=(255, 255, 255), width=2)
    draw.text((px+12, py-10), f"({px},{py})", fill=(255, 255, 0))
    prompt_img.save(os.path.join(output_dir, "prompt.jpg"))

    # 3. 예측
    predictor = ImagePredictor(
        model_size="tiny", image_size=1024,
        device="cpu",
        checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
        init_mode="finetune" if os.path.exists(checkpoint_path) else "scratch",
    )
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # 최고 점수 마스크 선택
    best_idx = scores.argmax()
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    # 이진화
    binary_mask = (best_mask > 0).astype(bool)
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[0]

    # 4. 마스크 오버레이
    overlay = image.copy()
    overlay[binary_mask] = (
        0.5 * overlay[binary_mask].astype(float) +
        0.5 * np.array([0, 200, 100])
    ).astype(np.uint8)

    # 마스크 경계
    from PIL import ImageFilter
    edge_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    edge = np.array(edge_img.filter(ImageFilter.FIND_EDGES)) > 128
    overlay[edge] = [0, 255, 0]

    # 프롬프트 점도 표시
    overlay_pil = Image.fromarray(overlay)
    draw2 = ImageDraw.Draw(overlay_pil)
    draw2.ellipse([px-r, py-r, px+r, py+r], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    overlay_pil.save(os.path.join(output_dir, "mask_overlay.jpg"))

    # 5. 이진 마스크 저장
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "mask_binary.png")
    )

    # 6. 결과 정보
    info = f"""SAM2 Image Prediction Result
============================
Image: {img_name} ({w}x{h})
Point: ({px}, {py})
Checkpoint: {checkpoint_path}
Best mask score: {best_score:.4f}
Mask area: {binary_mask.sum()} pixels ({binary_mask.sum()/(h*w)*100:.1f}%)
All scores: {[f'{s:.4f}' for s in scores]}
"""
    with open(os.path.join(output_dir, "result_info.txt"), "w") as f:
        f.write(info)

    print(info)
    print(f"  Results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Image Prediction")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--point", default=None, help="Prompt point x,y")
    parser.add_argument("--output", default=RESULT_DIR)
    args = parser.parse_args()

    pt = tuple(map(int, args.point.split(","))) if args.point else None
    predict_image(image_path=args.image, point=pt, output_dir=args.output)
