"""
Windows 배포 패키징

■ 역할:
  SAM2 모델을 Windows 소프트웨어에 탑재 가능한 형태로 패키징합니다.
  ONNX Runtime을 사용하여 Python 없이도 추론 가능합니다.

■ 배포 구조:
  deploy/
  ├── models/
  │   ├── sam2_image_encoder.onnx
  │   ├── sam2_prompt_encoder.onnx
  │   ├── sam2_mask_decoder.onnx
  │   └── config.json
  ├── sam2_inference.py        ← ONNX Runtime 추론 래퍼
  ├── requirements_deploy.txt  ← 최소 의존성
  └── README.md

■ 사용법:
  python hvs/scripts/deploy_windows.py --model_size tiny --checkpoint path/to/ckpt

■ 함수 목록:
  1. create_deploy_package()   — 전체 패키지 생성
  2. create_inference_wrapper() — ONNX 추론 래퍼 생성
  3. create_deploy_requirements() — 최소 의존성
  4. create_deploy_readme()     — 사용 가이드
  5. test_deploy_package()     — 패키지 검증
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────
# 1. 전체 패키지 생성
# ─────────────────────────────────────────────────

def create_deploy_package(
    model_size: str = "tiny",
    image_size: int = 1024,
    checkpoint_path: str = None,
    output_dir: str = "./deploy",
) -> str:
    """
    Windows 배포 패키지 생성

    Returns:
        패키지 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. ONNX 내보내기
    from hvs.scripts.export_model import export_all
    paths = export_all(
        model_size=model_size,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        output_dir=models_dir,
    )

    # 2. 추론 래퍼 생성
    create_inference_wrapper(output_dir, image_size)

    # 3. 의존성 파일
    create_deploy_requirements(output_dir)

    # 4. README
    create_deploy_readme(output_dir, model_size, image_size)

    print(f"\n=== Deploy Package Created ===")
    print(f"  Location: {os.path.abspath(output_dir)}")
    return output_dir


# ─────────────────────────────────────────────────
# 2. ONNX 추론 래퍼
# ─────────────────────────────────────────────────

def create_inference_wrapper(output_dir: str, image_size: int = 1024):
    """ONNX Runtime 추론 래퍼 Python 파일 생성"""
    code = f'''"""
SAM2 ONNX Runtime 추론 래퍼

■ 사용법:
  from sam2_inference import SAM2Predictor

  predictor = SAM2Predictor("models/")
  predictor.set_image(image)
  masks, scores = predictor.predict(points=[[x, y]], labels=[1])

■ Windows 소프트웨어 통합:
  1. deploy/ 폴더를 소프트웨어 디렉토리에 복사
  2. pip install -r requirements_deploy.txt
  3. from sam2_inference import SAM2Predictor
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np


# ─── 이미지 정규화 상수 ───
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
IMAGE_SIZE = {image_size}


class SAM2Predictor:
    """
    SAM2 ONNX Runtime 추론기

    ■ 함수 목록:
      set_image()   — 이미지 인코딩
      predict()     — 마스크 예측
      reset()       — 상태 초기화

    Args:
        model_dir: ONNX 모델 디렉토리 경로
        device: "cpu" 또는 "cuda"
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime 필요: pip install onnxruntime 또는 pip install onnxruntime-gpu"
            )

        # 프로바이더 설정
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # 모델 로드
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.ie_session = ort.InferenceSession(
            os.path.join(model_dir, self.config["files"]["image_encoder"]),
            providers=providers,
        )
        self.pe_session = ort.InferenceSession(
            os.path.join(model_dir, self.config["files"]["prompt_encoder"]),
            providers=providers,
        )
        self.md_session = ort.InferenceSession(
            os.path.join(model_dir, self.config["files"]["mask_decoder"]),
            providers=providers,
        )

        self.image_embed = None
        self.orig_hw = None

    def set_image(self, image: np.ndarray):
        """
        이미지 인코딩

        Args:
            image: (H, W, 3) uint8 RGB
        """
        self.orig_hw = image.shape[:2]

        # 전처리
        img = image.astype(np.float32)
        img = np.array(
            __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(image)
            .resize((IMAGE_SIZE, IMAGE_SIZE))
        ).astype(np.float32)
        img = (img - PIXEL_MEAN) / PIXEL_STD
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        # 인코딩
        outputs = self.ie_session.run(None, {{"image": img}})
        self.fpn_0, self.fpn_1, self.fpn_2 = outputs

    def predict(
        self,
        points: List[List[float]] = None,
        labels: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        마스크 예측

        Args:
            points: [[x, y], ...] 좌표 (원본 이미지 기준)
            labels: [1, 0, ...] 전경/배경

        Returns:
            (masks, scores): masks (N, H, W) bool, scores (N,)
        """
        assert self.fpn_2 is not None, "set_image() 먼저 호출"

        # 좌표 변환
        coords = np.array(points, dtype=np.float32)
        orig_h, orig_w = self.orig_hw
        coords[:, 0] *= IMAGE_SIZE / orig_w
        coords[:, 1] *= IMAGE_SIZE / orig_h
        coords = coords[np.newaxis]  # (1, N, 2)

        labs = np.array(labels, dtype=np.int32)[np.newaxis]  # (1, N)

        # 프롬프트 인코딩
        sparse, dense, image_pe = self.pe_session.run(
            None, {{"point_coords": coords, "point_labels": labs}}
        )

        # 고해상도 특징 (conv_s0/s1은 이미 ONNX에 포함되지 않으므로
        # 원본 FPN을 직접 사용)
        # 마스크 디코딩
        masks, iou_pred = self.md_session.run(
            None, {{
                "image_embed": self.fpn_2,
                "image_pe": image_pe,
                "sparse": sparse,
                "dense": dense,
                "high_res_0": self.fpn_0,
                "high_res_1": self.fpn_1,
            }}
        )

        # 후처리
        from PIL import Image as PILImage
        masks_full = []
        for m in masks[0]:
            pil_mask = PILImage.fromarray(m.astype(np.float32))
            pil_mask = pil_mask.resize((orig_w, orig_h))
            masks_full.append(np.array(pil_mask) > 0)

        return np.array(masks_full), iou_pred[0]

    def reset(self):
        """상태 초기화"""
        self.fpn_0 = None
        self.fpn_1 = None
        self.fpn_2 = None
        self.orig_hw = None


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    predictor = SAM2Predictor(sys.argv[1] if len(sys.argv) > 1 else "models/")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    predictor.set_image(test_image)
    masks, scores = predictor.predict(points=[[320, 240]], labels=[1])
    print(f"Masks: {{masks.shape}}, Scores: {{scores}}")
'''

    wrapper_path = os.path.join(output_dir, "sam2_inference.py")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(code)


# ─────────────────────────────────────────────────
# 3. 의존성
# ─────────────────────────────────────────────────

def create_deploy_requirements(output_dir: str):
    """최소 배포 의존성"""
    req = """# SAM2 배포 최소 의존성
numpy>=1.24.0
Pillow>=10.0.0
onnxruntime>=1.16.0
# onnxruntime-gpu>=1.16.0  # CUDA GPU 사용 시
"""
    req_path = os.path.join(output_dir, "requirements_deploy.txt")
    with open(req_path, "w") as f:
        f.write(req)


# ─────────────────────────────────────────────────
# 4. README
# ─────────────────────────────────────────────────

def create_deploy_readme(output_dir: str, model_size: str, image_size: int):
    """배포 README"""
    readme = f"""# SAM2 Windows 배포 패키지

## 모델 정보
- Model: SAM2 {model_size}
- Input size: {image_size}×{image_size}

## 설치
```
pip install -r requirements_deploy.txt
```

## 사용법
```python
from sam2_inference import SAM2Predictor

# 초기화
predictor = SAM2Predictor("models/", device="cpu")  # 또는 "cuda"

# 이미지 설정
import numpy as np
from PIL import Image
image = np.array(Image.open("test.jpg"))
predictor.set_image(image)

# 마스크 예측 (점 프롬프트)
masks, scores = predictor.predict(
    points=[[100, 200]],  # (x, y) 좌표
    labels=[1],           # 1=전경, 0=배경
)

# 결과
print(f"마스크 수: {{masks.shape[0]}}")
print(f"IoU 점수: {{scores}}")
```

## 파일 구조
```
deploy/
├── models/
│   ├── sam2_image_encoder.onnx   # 이미지 인코더
│   ├── sam2_prompt_encoder.onnx  # 프롬프트 인코더
│   ├── sam2_mask_decoder.onnx    # 마스크 디코더
│   └── config.json               # 모델 설정
├── sam2_inference.py              # 추론 래퍼
├── requirements_deploy.txt        # 의존성
└── README.md                      # 이 파일
```

## Windows 소프트웨어 통합
1. `deploy/` 폴더를 소프트웨어 디렉토리에 복사
2. `requirements_deploy.txt` 설치
3. `sam2_inference.py`의 `SAM2Predictor` 클래스 사용
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)


# ─────────────────────────────────────────────────
# 5. 패키지 검증
# ─────────────────────────────────────────────────

def test_deploy_package(deploy_dir: str) -> bool:
    """배포 패키지 검증"""
    models_dir = os.path.join(deploy_dir, "models")

    # 파일 존재 확인
    required = [
        "models/sam2_image_encoder.onnx",
        "models/sam2_prompt_encoder.onnx",
        "models/sam2_mask_decoder.onnx",
        "models/config.json",
        "sam2_inference.py",
        "requirements_deploy.txt",
        "README.md",
    ]
    all_ok = True
    for f in required:
        path = os.path.join(deploy_dir, f)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {f} ({size / 1024:.0f} KB)")
        if not exists:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Windows Deploy")
    parser.add_argument("--model_size", default="tiny")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="./deploy")
    args = parser.parse_args()

    create_deploy_package(
        model_size=args.model_size,
        image_size=args.image_size,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
