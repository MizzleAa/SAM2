"""
COCO 미니 데이터셋 다운로드 및 구축

소형 COCO 데이터셋을 다운로드하여 .dataset/coco_mini/에 저장합니다.
COCO val2017에서 처음 20장 이미지와 해당 어노테이션만 추출.

사용법:
    python hvs/scripts/prepare_coco_mini.py
"""

import json
import os
import sys
import urllib.request
import zipfile
import shutil


DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".dataset", "coco_mini")
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/val2017/"

NUM_IMAGES = 20  # 소형 테스트용


def download_file(url: str, dest: str):
    """파일 다운로드"""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest}")


def prepare_mini_dataset():
    """COCO 미니 데이터셋 구축"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    images_dir = os.path.join(DATASET_DIR, "images")
    annotations_dir = os.path.join(DATASET_DIR, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    ann_file = os.path.join(annotations_dir, "instances.json")
    if os.path.exists(ann_file) and len(os.listdir(images_dir)) >= NUM_IMAGES:
        print(f"Mini dataset already exists at {DATASET_DIR}")
        return

    # 1. COCO annotations 다운로드
    tmp_dir = os.path.join(DATASET_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ann_zip = os.path.join(tmp_dir, "annotations.zip")

    print("Step 1: Downloading COCO annotations...")
    download_file(COCO_ANNOTATIONS_URL, ann_zip)

    # 2. annotations 압축 해제
    print("Step 2: Extracting annotations...")
    full_ann_file = os.path.join(tmp_dir, "annotations", "instances_val2017.json")
    if not os.path.exists(full_ann_file):
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extract("annotations/instances_val2017.json", tmp_dir)

    # 3. 전체 annotations 로드
    print("Step 3: Loading full annotations...")
    with open(full_ann_file, "r") as f:
        coco = json.load(f)

    # 4. 처음 N개 이미지 선택
    selected_images = coco["images"][:NUM_IMAGES]
    selected_ids = {img["id"] for img in selected_images}

    # 5. 해당 이미지의 어노테이션 필터링
    selected_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in selected_ids
    ]

    print(f"  Selected {len(selected_images)} images, {len(selected_anns)} annotations")

    # 6. 미니 annotations 저장
    mini_coco = {
        "images": selected_images,
        "annotations": selected_anns,
        "categories": coco["categories"],
    }
    with open(ann_file, "w") as f:
        json.dump(mini_coco, f)
    print(f"  Saved: {ann_file}")

    # 7. 이미지 다운로드
    print(f"Step 4: Downloading {len(selected_images)} images...")
    for i, img in enumerate(selected_images):
        fname = img["file_name"]
        dest = os.path.join(images_dir, fname)
        if not os.path.exists(dest):
            url = COCO_VAL_IMAGES_URL + fname
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"  Warning: Failed to download {fname}: {e}")
                continue
        if (i + 1) % 5 == 0:
            print(f"  Downloaded {i+1}/{len(selected_images)}")

    # 8. 임시 파일 정리
    print("Step 5: Cleanup...")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 9. 요약
    num_images = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    print(f"\nDone! Mini COCO dataset:")
    print(f"  Images: {num_images}")
    print(f"  Annotations: {len(selected_anns)}")
    print(f"  Location: {os.path.abspath(DATASET_DIR)}")


if __name__ == "__main__":
    prepare_mini_dataset()
