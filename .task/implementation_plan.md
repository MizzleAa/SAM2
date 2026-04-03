# SAM 2 모델 개발 계획 — Facebook vs Ultralytics 비교 기반

## 개요

Facebook 공식 SAM 2 코드와 Ultralytics의 SAM 2 래핑 코드를 분석하고, 이를 기반으로 `hvs/` 경로에 독립적인 학습 및 예측 코드를 구성합니다.

---

## 1. 두 코드베이스 비교 분석

### 1.1 Facebook SAM 2 (`facebook/`)

| 항목 | 설명 |
|---|---|
| **모델 구성** | Hydra + OmegaConf YAML 설정 기반, `build_sam.py`로 모델 빌드 |
| **Image Encoder** | Hiera 백본 (MAE 사전학습), `hieradet.py` + `image_encoder.py` |
| **Memory 모듈** | `memory_attention.py`, `memory_encoder.py` — 비디오 처리의 핵심 |
| **Prompt/Mask Decoder** | `prompt_encoder.py`, `mask_decoder.py`, `transformer.py` (Two-way) |
| **Base Model** | `sam2_base.py` (SAM2Base) — 47KB, 전체 파이프라인 로직 |
| **Image Predictor** | `sam2_image_predictor.py` — set_image → predict 패턴 |
| **Video Predictor** | `sam2_video_predictor.py` — init_state → add_points → propagate_in_video |
| **학습** | `training/` — Hydra config 기반, DDP/submitit, `SAM2Train` 모델 |
| **Loss 함수** | Focal + Dice + IoU + Object Score (Multi-step Multi-mask) |
| **데이터셋** | VOS (DAVIS-style PNG), SA-V (JSON), SA-1B (이미지) |
| **의존성** | hydra-core, omegaconf, iopath, submitit, fvcore 등 무거운 의존성 |
| **체크포인트** | `{"model": state_dict}` 형태로 저장 |

### 1.2 Ultralytics SAM 2 (`ultralytics/`)

| 항목 | 설명 |
|---|---|
| **모델 구성** | 파이썬 코드로 직접 빌드 (`build.py`), YAML 없음 |
| **모듈 파일** | `modules/` — sam.py, encoders.py, decoders.py, blocks.py, memory_attention.py, transformer.py |
| **SAM2Model** | `modules/sam.py` — SAM2Base와 동등하지만 Ultralytics 스타일로 재구성 |
| **Predictor** | `predict.py` — `SAM2Predictor` 클래스, BasePredictor 상속 |
| **학습** | Ultralytics 통합 엔진 사용, **SAM2 전용 학습 미지원** |
| **장점** | 간결한 인터페이스, `SAM("sam2_b.pt")` 한 줄로 사용, auto-download |
| **단점** | 비디오 프롬프트 추적(propagation) 기능이 Facebook 대비 제한적 |
| **라이선스** | AGPL-3.0 (상용 주의) |

### 1.3 핵심 차이점 요약

| 비교 항목 | Facebook SAM 2 | Ultralytics SAM 2 |
|---|---|---|
| **설정 방식** | Hydra YAML | Python 코드 직접 |
| **모델 빌드** | `hydra.instantiate()` | `build_sam2_*()` 함수 |
| **비디오 추적** | 완전한 Memory Bank + Propagation | 이미지 중심, 비디오 제한적 |
| **학습 지원** | 전용 trainer, loss, optimizer | 미지원 (추론 전용) |
| **사용 편의성** | 복잡 (Hydra 설정 필요) | 매우 간단 |
| **체크포인트 호환** | Facebook 공식 `.pt` | Facebook `.pt` 변환 사용 |
| **리팩토링 수준** | 원본 연구 코드 | 산업용 리팩토링 |

---

## 2. GPU 환경 계획

> [!NOTE]
> 현재 개발 환경은 RTX 5060, 실제 학습은 RTX 4090에서 수행 예정

| 항목 | RTX 5060 (개발) | RTX 4090 (학습) |
|---|---|---|
| **용도** | 코드 개발, 소규모 테스트, 추론 | 본격 학습, 파인튜닝 |
| **VRAM** | 8~16GB (예상) | 24GB |
| **학습 배치 사이즈** | 1 (테스트용) | 2~4 |
| **AMP (Mixed Precision)** | 필수 | 권장 |
| **모델 크기** | Tiny/Small (테스트) | Base+/Large (본 학습) |

### GPU 메모리 관리 전략
- **Gradient Checkpointing**: 메모리 절약을 위해 활성화 (속도 ↓ 20%, 메모리 ↓ 40%)
- **AMP (bfloat16/float16)**: 항상 활성화
- **학습 프레임 수**: 4~8 프레임 (4090 기준)
- **이미지 해상도**: 512 (개발) → 1024 (본 학습)

---

## 3. 제안 폴더 구조 (모듈 분리 기준)

> [!IMPORTANT]
> **Backbone / Neck / Head 패턴**으로 모듈을 분리하여 재사용성을 극대화합니다.
> 이 패턴은 딥러닝 모델의 표준 구조로, 각 부분을 독립적으로 교체/확장할 수 있습니다.

```
c:\workspace\SAM2\hvs\
│
├── README.md                          # HVS SAM2 프로젝트 문서
│
├── configs/                           # 설정 파일 (YAML, Hydra 의존성 없음)
│   ├── model/
│   │   ├── sam2_hiera_t.yaml          # Tiny: 개발/테스트용 (5060에서 사용)
│   │   ├── sam2_hiera_b+.yaml         # Base+: 기본 학습용
│   │   └── sam2_hiera_l.yaml          # Large: 고정밀 학습용 (4090 전용)
│   ├── train/
│   │   ├── default_train.yaml         # 기본 학습 하이퍼파라미터
│   │   ├── finetune_mose.yaml         # MOSE 파인튜닝 (검증용)
│   │   └── finetune_custom.yaml       # 자사 데이터 파인튜닝 (산업 결함)
│   └── predict/
│       ├── image_predict.yaml         # 이미지 예측 설정
│       └── video_predict.yaml         # 비디오 예측 설정
│
│   ┌──────────────────────────────────────────────────┐
│   │  models/ — 모델 계층 구조 (Backbone/Neck/Head)    │
│   └──────────────────────────────────────────────────┘
├── models/
│   ├── __init__.py
│   ├── build.py                       # ★ 모델 빌드 팩토리 (한 줄로 모델 생성)
│   │
│   ├── backbone/                      # ★ BACKBONE: 입력 → 특징 추출
│   │   ├── __init__.py
│   │   ├── hiera.py                   # Hiera 트랜스포머 백본 (계층적 ViT)
│   │   │                              #   - ViT를 계층적으로 만든 것
│   │   │                              #   - 이미지를 패치로 나누고 Self-Attention으로 처리
│   │   │                              #   - 4단계로 해상도 줄이며 특징 추출
│   │   ├── blocks.py                  # 트랜스포머 기본 블록들
│   │   │                              #   - MultiHeadAttention: 여러 관점에서 관계 파악
│   │   │                              #   - MLP: 특징 변환용 완전연결 계층
│   │   │                              #   - LayerNorm: 학습 안정화
│   │   └── position_encoding.py       # 위치 인코딩 (Sine, RoPE)
│   │                                  #   - 트랜스포머는 순서 정보가 없으므로
│   │                                  #   - 위치 정보를 별도로 주입해야 함
│   │
│   ├── neck/                          # ★ NECK: 백본 특징 → 다중 스케일 정제
│   │   ├── __init__.py
│   │   ├── fpn_neck.py                # FPN (Feature Pyramid Network)
│   │   │                              #   - 백본의 여러 해상도 특징을 결합
│   │   │                              #   - 작은 결함(스크래치)과 큰 결함을 동시에 감지
│   │   │                              #   - 고해상도(세밀) + 저해상도(문맥) 정보 융합
│   │   └── image_encoder.py           # 이미지 인코더 (Backbone + Neck 조합)
│   │                                  #   - Backbone(Hiera) + Neck(FPN) = Image Encoder
│   │
│   ├── head/                          # ★ HEAD: 정제된 특징 → 최종 출력
│   │   ├── __init__.py
│   │   ├── prompt_encoder.py          # 프롬프트 인코더
│   │   │                              #   - 사용자 입력(점, 박스, 마스크)을 임베딩으로 변환
│   │   │                              #   - "여기를 세그멘테이션해줘"라는 지시를 모델이 이해하는 형태로
│   │   ├── mask_decoder.py            # 마스크 디코더
│   │   │                              #   - 이미지 특징 + 프롬프트 → 세그멘테이션 마스크 생성
│   │   │                              #   - IoU 점수도 함께 예측 (마스크 품질 판단)
│   │   └── transformer.py            # Two-way Transformer 블록
│   │                                  #   - 이미지↔프롬프트 간 양방향 정보 교환
│   │                                  #   - 프롬프트가 이미지 정보를 읽고
│   │                                  #   - 이미지가 프롬프트 정보를 읽어서
│   │                                  #   - 정확한 마스크 위치를 결정
│   │
│   ├── memory/                        # ★ MEMORY: 비디오 시간적 처리 전용
│   │   ├── __init__.py
│   │   ├── memory_attention.py        # 메모리 어텐션
│   │   │                              #   - 현재 프레임이 과거 프레임 정보를 참조
│   │   │                              #   - Self-Attention: 현재 프레임 내부 관계
│   │   │                              #   - Cross-Attention: 현재↔과거 프레임 간 관계
│   │   │                              #   - 물체가 움직여도 과거 기억으로 추적 가능
│   │   ├── memory_encoder.py          # 메모리 인코더
│   │   │                              #   - 예측 결과를 메모리로 변환하여 저장
│   │   │                              #   - 마스크 + 이미지 특징 → 압축된 메모리 형태
│   │   └── memory_bank.py             # 메모리 뱅크 관리
│   │                                  #   - FIFO 큐로 최근 N프레임 메모리 보관
│   │                                  #   - 프롬프트 프레임 메모리 별도 보관
│   │                                  #   - Object Pointer: 경량 고수준 객체 정보
│   │
│   └── sam2_base.py                   # SAM2 통합 모델 (전체 파이프라인)
│                                      #   - Backbone + Neck + Head + Memory 조합
│                                      #   - 이미지 모드: Memory 비활성 → SAM처럼 동작
│                                      #   - 비디오 모드: Memory 활성 → 프레임 간 추적
│
│   ┌──────────────────────────────────────────────────┐
│   │  training/ — 학습 코드                            │
│   └──────────────────────────────────────────────────┘
├── training/
│   ├── __init__.py
│   ├── train.py                       # ★ 학습 진입점 (단일 GPU 전용)
│   │                                  #   python train.py --config configs/train/...
│   ├── trainer.py                     # 학습 루프 (train/eval/save 관리)
│   ├── sam2_train.py                  # SAM2Train: 학습 전용 모델 확장
│   │                                  #   - 인터랙티브 프롬프팅 시뮬레이션
│   │                                  #   - 8프레임 시퀀스 샘플링
│   │                                  #   - 보정 클릭 자동 생성
│   ├── loss_fns.py                    # 손실 함수
│   │                                  #   - Focal Loss: 어려운 픽셀에 집중 학습
│   │                                  #   - Dice Loss: 마스크 영역 겹침 최적화
│   │                                  #   - IoU Loss: 예측 품질 점수 정확도
│   │                                  #   - Object Score Loss: 객체 존재 여부 판단
│   ├── optimizer.py                   # 옵티마이저 + 스케줄러
│   │                                  #   - AdamW + Cosine LR + Layer-wise decay
│   │                                  #   - Backbone은 더 작은 학습률 사용
│   └── dataset/
│       ├── __init__.py
│       ├── vos_dataset.py             # VOS 비디오 데이터셋 (DAVIS/MOSE 포맷)
│       ├── image_dataset.py           # 이미지 데이터셋 (단일 프레임)
│       ├── custom_dataset.py          # ★ 자사 산업 결함 데이터셋
│       │                              #   - 이미지 + 마스크 쌍 로드
│       │                              #   - 불규칙 시간 간격 이미지 시퀀스 지원
│       │                              #   - 결함 유형별 라벨링 지원
│       ├── transforms.py              # 데이터 증강
│       │                              #   - 산업 데이터용 특화 증강 포함:
│       │                              #     - 조명 변화 시뮬레이션
│       │                              #     - 결함 크기/위치 랜덤 변환
│       │                              #     - 노이즈/블러 추가
│       └── sampler.py                 # 프레임 샘플러 (비디오에서 학습 프레임 선택)
│
│   ┌──────────────────────────────────────────────────┐
│   │  predictor/ — 예측(추론) 코드                     │
│   └──────────────────────────────────────────────────┘
├── predictor/
│   ├── __init__.py
│   ├── image_predictor.py             # 이미지 세그멘테이션 예측
│   │                                  #   사용법:
│   │                                  #   predictor.set_image(img)
│   │                                  #   masks = predictor.predict(points=[[x,y]])
│   ├── video_predictor.py             # 비디오 세그멘테이션 예측
│   │                                  #   사용법:
│   │                                  #   state = predictor.init_state(video)
│   │                                  #   predictor.add_new_points(state, frame, points)
│   │                                  #   for frame, masks in predictor.propagate(state):
│   │                                  #       process(masks)
│   ├── auto_mask_generator.py         # 자동 전체 마스크 생성
│   │                                  #   - 프롬프트 없이 이미지 내 모든 물체 검출
│   └── utils.py                       # 예측 후처리 유틸
│
│   ┌──────────────────────────────────────────────────┐
│   │  utils/ — 공통 유틸리티                            │
│   └──────────────────────────────────────────────────┘
├── utils/
│   ├── __init__.py
│   ├── transforms.py                  # 이미지 전처리 (리사이즈, 정규화)
│   ├── misc.py                        # 기타 유틸
│   ├── checkpoint.py                  # 체크포인트 로드/저장
│   │                                  #   - Facebook 공식 체크포인트 호환
│   │                                  #   - 자사 학습 체크포인트 관리
│   ├── distributed.py                 # 분산 학습 유틸 (향후 멀티 GPU 대비)
│   └── visualization.py              # 결과 시각화 (마스크 오버레이, 비교 플롯)
│
│   ┌──────────────────────────────────────────────────┐
│   │  scripts/ — 실행 스크립트                          │
│   └──────────────────────────────────────────────────┘
├── scripts/
│   ├── train_mose.py                  # MOSE 학습 → Facebook 결과 재현 검증
│   ├── train_custom.py                # 자사 결함 데이터 학습
│   ├── predict_image.py               # 이미지 예측 (산업 결함 검출)
│   ├── predict_video.py               # 비디오/시퀀스 예측
│   ├── download_checkpoints.py        # 체크포인트 다운로드
│   ├── compare_facebook_ultralytics.py # 구현 비교 검증
│   └── export_model.py               # 모델 내보내기 (ONNX 등)
│
├── checkpoints/                       # 체크포인트 저장 폴더
├── data/                              # 데이터 참조 폴더
├── logs/                              # 학습 로그/텐서보드
├── tests/                             # 테스트 코드
│   ├── test_model_build.py
│   ├── test_backbone.py               # Backbone 단독 테스트
│   ├── test_neck.py                   # Neck 단독 테스트
│   ├── test_head.py                   # Head 단독 테스트
│   ├── test_memory.py                 # Memory 단독 테스트
│   ├── test_image_predict.py
│   ├── test_video_predict.py
│   └── test_training.py
│
└── requirements.txt                   # 의존성 (Hydra/submitit 없음)
```

---

## 4. Backbone / Neck / Head 모듈 분리 설계

> [!NOTE]
> **딥러닝 모델은 일반적으로 3개 계층으로 분리됩니다.** 이 구조를 따르면 각 부분을 독립적으로 교체하거나 재사용할 수 있습니다.

### 4.1 전체 데이터 흐름도

```
입력 이미지 (H×W×3)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  BACKBONE (Hiera)                                               │
│  "눈": 이미지에서 특징을 추출하는 역할                              │
│                                                                 │
│  이미지 → [패치 분할] → [트랜스포머 블록 ×N] → 다중 스케일 특징맵    │
│                                                                 │
│  출력: 4개 해상도의 특징맵                                         │
│  - Stage 1: H/4 × W/4   (고해상도, 세밀한 경계)                    │
│  - Stage 2: H/8 × W/8   (중해상도)                               │
│  - Stage 3: H/16 × W/16 (중저해상도)                              │
│  - Stage 4: H/32 × W/32 (저해상도, 전체 맥락)                      │
└─────────────────────────────────────────────────────────────────┘
    │ 4개 스케일 특징맵
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  NECK (FPN: Feature Pyramid Network)                            │
│  "목": 여러 해상도의 특징을 통합하는 역할                            │
│                                                                 │
│  - Top-down: 저해상도(전체맥락)를 고해상도로 전파                    │
│  - 각 스케일을 256채널로 통일                                      │
│  - 결과: 세밀한 결함 경계 + 전체 문맥 정보 동시 보유                  │
│                                                                 │
│  ★ 산업 결함 검출에 중요:                                          │
│    작은 스크래치(고해상도) + 넓은 영역 변색(저해상도) 동시 감지       │
└─────────────────────────────────────────────────────────────────┘
    │ 통합된 3개 스케일 특징맵
    ├────────────────────────────────┐
    ▼                                ▼
┌────────────────────┐    ┌─────────────────────────────────────┐
│ MEMORY (비디오 전용) │    │  HEAD (Prompt Encoder + Mask Decoder) │
│                    │    │  "머리": 최종 판단을 내리는 역할         │
│ - Memory Attention │    │                                     │
│   (과거 프레임 참조) │    │  [프롬프트 인코더]                     │
│ - Memory Encoder   │    │    점/박스/마스크 → 임베딩 변환         │
│   (결과를 기억 저장) │    │                                     │
│ - Memory Bank      │    │  [마스크 디코더]                       │
│   (기억 큐 관리)    │    │    이미지 특징 + 프롬프트 → 마스크 출력  │
│                    │    │    + IoU 품질 점수 예측                │
│ ★ 이미지 모드에서는  │    │                                     │
│   비활성 (빈 메모리) │    │  ★ 재사용 가능:                       │
│ ★ 비디오 모드에서만  │    │    다른 백본과도 조합 가능              │
│   활성화            │    │                                     │
└────────────────────┘    └─────────────────────────────────────┘
                                     │
                                     ▼
                          출력: 세그멘테이션 마스크 + IoU 점수
```

### 4.2 재사용 시나리오

```python
# 시나리오 1: 기본 SAM2 모델 (Facebook 동일 구조)
model = build_sam2(backbone="hiera_b+", neck="fpn", head="sam2_decoder")

# 시나리오 2: 향후 백본만 교체 (예: ResNet, Swin 등)
model = build_sam2(backbone="resnet50", neck="fpn", head="sam2_decoder")

# 시나리오 3: Head만 변경 (예: 결함 분류 Head 추가)
model = build_sam2(backbone="hiera_b+", neck="fpn", head="defect_decoder")

# 시나리오 4: Backbone만 재사용 (다른 Task용)
backbone = build_backbone("hiera_b+")
features = backbone(image)  # 다른 용도로 활용 가능
```

---

## 5. ViT / Transformer 핵심 개념 가이드

> [!NOTE]
> 코드 작성 시 아래 개념에 대한 **상세 한글 주석**을 모든 관련 파일에 포함합니다.

### 5.1 핵심 용어 사전

```
┌─────────────────┬───────────────────────────────────────────────────────────┐
│ 용어             │ 설명 (비유 포함)                                           │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Transformer     │ 입력 데이터의 모든 부분 간 관계를 파악하는 신경망 구조.         │
│                 │ 비유: 회의실에서 모든 참석자가 서로의 발언을 듣고 종합 판단      │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ ViT             │ Vision Transformer. 이미지를 작은 조각(패치)으로 나누어        │ 
│ (Vision         │ Transformer에 넣는 방식. 기존 CNN이 지역적 특징만 보는 반면,   │
│  Transformer)   │ ViT는 이미지 전체의 관계를 동시에 파악할 수 있음.              │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Patch           │ 이미지를 격자 형태로 나눈 작은 조각.                          │
│ (패치)           │ 예: 1024×1024 이미지 → 16×16 크기 패치 64×64개              │
│                 │ 비유: 퍼즐 조각처럼 이미지를 나누는 것                         │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Embedding       │ 데이터(패치, 좌표 등)를 고정 길이 숫자 벡터로 변환한 것.        │
│ (임베딩)         │ 비유: 각 패치의 "신분증" — 해당 패치의 특성을 숫자로 요약       │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Attention       │ "어디에 집중할지"를 학습하는 메커니즘.                         │
│ (어텐션)         │ Query(질문) × Key(답변 후보) → 가중치 → Value(실제 정보)      │
│                 │ 비유: 시험에서 "중요한 부분에 밑줄 치기"                       │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Self-Attention  │ 같은 데이터 내부의 각 요소가 서로를 참조하는 것.               │
│                 │ 예: 이미지의 한 패치가 다른 모든 패치와의 관계를 계산           │
│                 │ 비유: 학급 학생들이 서로 누가 비슷한지 파악                    │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Cross-Attention │ 서로 다른 두 데이터 간 관계를 파악하는 것.                     │
│                 │ 예: 현재 프레임이 과거 프레임의 정보를 참조                    │
│                 │ 비유: 새 직원이 기존 직원에게 업무 인수인계 받기                │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Multi-Head      │ Attention을 여러 "관점"에서 동시에 수행하는 것.                │
│ Attention       │ 비유: 여러 전문가가 각자의 시각으로 동시에 분석                 │
│                 │ (한 명은 색깔, 한 명은 형태, 한 명은 질감 분석)                │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Positional      │ 트랜스포머는 입력 순서를 모르므로, 위치 정보를 별도로 추가.      │
│ Encoding        │ 비유: 퍼즐 조각 뒷면에 좌표 번호를 적어두는 것                 │
│ (위치 인코딩)    │ - Sine: 고정된 수학 함수로 위치 생성                          │
│                 │ - RoPE: 회전 기반 상대적 위치 (SAM2에서 사용)                  │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ FPN             │ Feature Pyramid Network. 여러 해상도의 특징을 결합             │
│ (특징 피라미드)   │ 비유: 지도를 여러 축척으로 겹쳐 보기                          │
│                 │ (세계지도 + 시도지도 + 동네지도 = 전체적+세부적 동시)            │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Mask Decoder    │ 프롬프트 + 이미지 특징 → 정밀한 세그멘테이션 마스크 생성         │
│ (마스크 디코더)   │ - IoU Head: 마스크 품질 점수 예측                            │
│                 │ - Multimask: 모호한 입력에 대해 3개 후보 마스크 제시            │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Memory Bank     │ 비디오에서 과거 프레임의 정보를 저장하는 큐.                    │
│ (메모리 뱅크)    │ 비유: 수사관의 수첩 — 이전에 본 것들을 기록해두고 참조          │
│                 │ ★ 이미지 모드: 수첩이 비어있음 → 현재 프레임만으로 판단         │
│                 │ ★ 비디오 모드: 수첩에 기록 축적 → 과거와 비교하며 추적          │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

### 5.2 SAM2의 Hiera Backbone 동작 원리

```
입력 이미지 (1024 × 1024 × 3)
         │
         ▼ ① 패치 분할 (Patch Embedding)
   이미지를 16×16 크기 패치로 분할
   각 패치를 벡터(숫자 배열)로 변환
   결과: 64×64개 패치, 각 112차원 벡터
         │
         ▼ ② Stage 1: 트랜스포머 블록 ×2
   각 패치가 주변 패치와 관계 파악 (Window Attention)
   결과: 64×64 해상도, 112채널 특징맵
         │
         ▼ ③ Stage 2: 해상도 절반 축소 + 트랜스포머 블록 ×3
   2×2 패치를 합쳐서 해상도 줄임
   결과: 32×32 해상도, 224채널 특징맵
         │
         ▼ ④ Stage 3: 해상도 절반 축소 + 트랜스포머 블록 ×16
   가장 많은 블록 → 핵심 특징 학습
   일부 블록에서 Global Attention (전체 이미지 참조)
   결과: 16×16 해상도, 448채널 특징맵
         │
         ▼ ⑤ Stage 4: 해상도 절반 축소 + 트랜스포머 블록 ×3
   결과: 8×8 해상도, 896채널 특징맵

   ★ Window Attention: 일부 영역(창문)만 보고 계산 → 빠름
   ★ Global Attention: 전체 이미지를 보고 계산 → 정확하지만 느림
   ★ Hiera는 Window를 기본, 필요한 곳만 Global 사용 → 효율적
```

### 5.3 코드 주석 작성 기준

모든 모델 코드에 다음 수준의 주석을 포함합니다:

```python
class MultiHeadAttention(nn.Module):
    """
    멀티 헤드 어텐션 (Multi-Head Attention)
    
    ■ 역할:
      입력 데이터의 각 요소가 다른 모든 요소와의 관계(중요도)를 계산하여
      가장 관련 있는 정보에 집중하도록 합니다.
    
    ■ 동작 원리:
      1. 입력에서 Query(질문), Key(답변 후보), Value(실제 정보) 생성
      2. Query와 Key의 유사도 계산 → 어텐션 가중치 (어디에 집중할지)
      3. 가중치를 Value에 적용 → 중요한 정보만 추출
      4. 이것을 여러 "관점(Head)"에서 병렬로 수행 후 결합
    
    ■ 비유:
      라인 검사원이 제품을 볼 때:
      - Head 1: 표면 색상 변화에 집중
      - Head 2: 형태/윤곽선에 집중  
      - Head 3: 질감/패턴에 집중
      각각의 관점을 종합하여 최종 판단
    
    ■ 입출력:
      입력: (B, N, D) — B개 배치, N개 패치, D차원 특징
      출력: (B, N, D) — 관계 정보가 반영된 특징
    
    Args:
        embedding_dim: 입력 특징의 차원 수 (예: 256)
        num_heads: 병렬 어텐션 관점 수 (예: 8)
        downsample_rate: 내부 연산 차원 축소 비율 (메모리 절약)
    """
    def __init__(self, embedding_dim: int, num_heads: int, ...):
        ...
```

---

## 6. 산업 결함 데이터 대응 전략

### 6.1 이미지 학습 → 비디오 추론 가능성

> [!IMPORTANT]
> **결론: 가능합니다.** SAM2의 설계가 이를 직접 지원합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│  SAM2의 이중 모드 동작                                           │
│                                                                 │
│  [이미지 모드] Memory Bank = 비어있음                              │
│  → Backbone + Neck + Head만으로 세그멘테이션                      │
│  → ★ 자사 이미지 데이터로 이 부분을 파인튜닝                       │
│                                                                 │
│  [비디오 모드] Memory Bank = 과거 프레임 축적                      │
│  → 위의 세그멘테이션 능력 + Memory 모듈로 시간적 추적               │
│  → ★ Memory 모듈은 Facebook 사전학습으로 이미 학습됨               │
│  → ★ 추가 학습 없이도 비디오 추론 가능                             │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 논거:**
- Facebook도 SA-1B(**이미지**)와 VOS(**비디오**) 를 혼합 학습
- 이미지 전용 학습만으로도 zero-shot 비디오 벤치마크 성능이 기본 수준 달성 (Table 7, Row 4: SA-V 62.9 J&F)
- **사전학습 체크포인트의 Memory 모듈**이 비디오 추적 능력을 보유
- 자사 이미지로 **Backbone+Head만 파인튜닝**하면 Memory 모듈은 보존됨

### 6.2 DLBackend2 데이터 포맷 분석 결과

> [!IMPORTANT]
> `C:\workspace\dltrainer\DLBackend2`의 데이터셋은 **COCO 형식이 아닌 HVS 자체 포맷**입니다.

#### HVS 자체 어노테이션 구조

```
DLBackend2/
├── datasets/
│   ├── seg_datasetV1.py       # 멀티라벨 바이너리 마스크 (C,H,W)
│   └── seg_datasetV2.py       # softmax 모드 추가 (zIndex 우선순위)
│
│  어노테이션 폴더 구조 (labelPaths):
│  {labelPath}/
│  ├── LabelInfo.json          # 클래스 정의: [{Name, Index, UseLabel}]
│  └── Images_*.json           # 이미지별 어노테이션 배열
│
│  각 이미지 어노테이션 구조:
│  {
│    "FileName": "image.jpg",
│    "LabelPolygonShapes": [{ClassLabel:{Name}, PointCollection:["x,y",...]}],
│    "LabelStrokeShapes":  [{ClassLabel:{Name}, PointsCollection:["x,y",...], Thinkness}],
│    "LabelCircleShapes":  [{ClassLabel:{Name}, CenterX, CenterY, Radius}],
│    "LabelBoxShapes":     [{ClassLabel:{Name}, X, Y, Width, Height}],
│    "LabelLineShapes":    [{ClassLabel:{Name}, SX, SY, EX, EY, Thinkness}]
│  }
```

#### SAM2가 요구하는 데이터 형식

```
SAM2 학습 입력:
  - 이미지: (H, W, 3) RGB
  - 마스크: (H, W) 바이너리 또는 (N, H, W) 인스턴스별
  - 프롬프트: 점(x,y) / 박스(x1,y1,x2,y2) / 이전 마스크
```

#### 데이터 변환 전략: HVS → SAM2 어댑터

```python
# hvs/training/dataset/hvs_adapter.py

class HVStoSAM2Adapter:
    """
    DLBackend2의 HVS 어노테이션을 SAM2 학습용 포맷으로 변환
    
    변환 과정:
    1. LabelInfo.json → 클래스 목록 파싱
    2. Images_*.json → 이미지별 Shape 리스트 파싱
    3. Polygon/Stroke/Circle/Box/Line → cv2로 바이너리 마스크 렌더링
       (SegDatasetV1의 마스크 생성 로직 재사용)
    4. 바이너리 마스크에서 SAM2 프롬프트 자동 생성
       - 마스크 중심점 → point prompt
       - 마스크 바운딩박스 → box prompt
    """
    
    # ★ DLBackend2의 cv2 렌더링 로직을 그대로 재사용
    # seg_datasetV1.py L185-251의 마스크 생성 코드를 공유
```

#### SAM2용 통합 데이터 구조

```
c:\workspace\SAM2\hvs\data\
├── hvs_format/                    # HVS 원본 데이터 (심볼릭 링크)
│   ├── {ImageGroupName}/
│   │   ├── images/               # imagePaths
│   │   └── labels/               # labelPaths
│   │       ├── LabelInfo.json
│   │       └── Images_*.json
│
├── sam2_format/                   # 변환된 SAM2 학습 데이터
│   ├── images/                   # 이미지 복사/링크
│   ├── masks/                    # 렌더링된 바이너리 마스크 (PNG)
│   ├── prompts/                  # 자동 생성된 프롬프트 (JSON)
│   │   └── {image_id}.json      # {points, boxes, labels}
│   └── metadata.json            # 전체 데이터셋 메타정보
│
└── convert_hvs_to_sam2.py        # 변환 스크립트
```

> [!NOTE]
> - DLBackend2의 `seg_datasetV1.py`의 마스크 렌더링 로직을 **hvs_adapter.py에서 직접 import하여 재사용**
> - 변환은 1회성 전처리 스크립트로 수행 (학습 시 매번 변환하지 않음)
> - 마스크 → 프롬프트 자동 생성: 마스크 중심점/bbox를 SAM2의 point/box 프롬프트로 변환

---

### 6.3 테스트 케이스 점진적 확장 전략

> [!NOTE]
> 각 시나리오(Phase)마다 test case를 누적 확장하여, 이전 단계의 테스트가 깨지지 않으면서 새 기능을 추가합니다.

```
┌────────────────────────────────────────────────────────────────────┐
│  Test Expansion Strategy                                          │
│                                                                   │
│  Phase 0 (가능성 검증):                                             │
│  └─ test_01_model_build.py       모델 빌드 + forward shape 검증     │
│  └─ test_02_overfit_tiny.py      소량 데이터 overfitting 검증        │
│                                                                   │
│  Phase 1 (모델 코드):                                               │
│  └─ test_03_backbone.py          Backbone forward + output shape   │
│  └─ test_04_neck.py              Neck forward + FPN multi-scale    │
│  └─ test_05_head.py              Head forward + mask output        │
│  └─ test_06_memory.py            Memory 모듈 forward               │
│  └─ test_07_checkpoint.py        체크포인트 로드/저장 검증            │
│                                                                   │
│  Phase 2 (예측 코드):                                               │
│  └─ test_08_image_predict.py     이미지 예측 파이프라인               │
│  └─ test_09_video_predict.py     비디오 예측 파이프라인               │
│  └─ test_10_compare_fb_ultra.py  Facebook/Ultralytics 비교          │
│                                                                   │
│  Phase 3 (학습 코드):                                               │
│  └─ test_11_dataset_hvs.py       HVS 데이터 로드 + 변환 검증         │
│  └─ test_12_dataset_sam2.py      SAM2 포맷 데이터셋 검증             │
│  └─ test_13_loss_fns.py          Loss 함수 출력 검증                 │
│  └─ test_14_train_step.py        1-step 학습 → loss 정상 출력        │
│  └─ test_15_train_loop.py        N-epoch 학습 → loss 수렴 확인       │
│                                                                   │
│  Phase 4 (자사 데이터):                                              │
│  └─ test_16_custom_data.py       자사 데이터 로드 + 전처리 검증       │
│  └─ test_17_finetune.py          파인튜닝 결과 비교                  │
│  └─ test_18_video_inference.py   이미지 학습 → 비디오 추론 검증       │
│                                                                   │
│  ★ 각 Phase에서 기존 테스트는 반드시 통과해야 함 (회귀 방지)           │
│  ★ pytest로 관리: pytest tests/ -v --tb=short                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. 스크래치 학습 가능성 분석

> [!CAUTION]
> **사용자 요구: 사전학습 모델 없이 처음부터 학습, 데이터 수량 제한적**
> 이 조건에서의 현실적 분석과 가능성 검증 전략을 아래에 정리합니다.

### 7.1 스크래치 학습의 현실적 난이도

```
┌──────────────────────────────────────────────────────────────────┐
│  Facebook의 SAM2 학습 조건 (논문 기준)                             │
│                                                                  │
│  데이터: SA-1B (11M 이미지) + SA-V (50.9K 비디오) + 기타           │
│  GPU:   256 × A100 (80GB)                                        │
│  시간:  수일~수주                                                  │
│  비용:  수천만원 이상 (클라우드 기준)                                │
│                                                                  │
│  ★ 모델 파라미터 수:                                                │
│    - Tiny:   ~39M (3,900만 파라미터)                                │
│    - Small:  ~46M                                                  │
│    - Base+:  ~81M                                                  │
│    - Large:  ~224M                                                 │
└──────────────────────────────────────────────────────────────────┘
```

| 항목 | Facebook (원본) | 자사 조건 | 격차 |
|---|---|---|---|
| **데이터 수** | 11M+ 이미지 | 수백~수천장 (추정) | **1,000~10,000배** |
| **GPU** | 256×A100 | 1×RTX 4090 | **256배 (VRAM은 1/3)** |
| **학습 시간** | 수일 | 동일 기간 학습 시 1/256 진행 | — |

### 7.2 스크래치 학습이 가능한 경우 vs 불가능한 경우

```
┌─────────────────────────────────────────────────────────────┐
│  ✅ 스크래치 학습이 가능할 수 있는 조건                        │
│                                                             │
│  1. 모델 크기를 대폭 축소 (Tiny 또는 더 작은 커스텀 모델)       │
│  2. 이미지 해상도 축소 (1024 → 256~512)                       │
│  3. 태스크 범위 축소 (범용 세그멘테이션 → 특정 결함 유형만)      │
│  4. 데이터 증강으로 실질 데이터 수 확대                         │
│  5. 비디오 기능(Memory) 제외, 이미지 전용 학습                  │
│                                                             │
│  결과 기대치: 제한된 도메인에서 "동작은 하지만"                  │
│  범용 모델 대비 성능은 크게 낮을 것                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ❌ 스크래치 학습이 비현실적인 경우                             │
│                                                             │
│  1. Base+/Large 모델을 스크래치로 학습 → 데이터 부족으로 과적합  │
│  2. 범용 세그멘테이션 성능 기대 → 불가능                       │
│  3. 비디오 Memory 모듈까지 스크래치 학습 → 비디오 데이터 필요    │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 권장 접근 전략 (3가지 경로)

> [!IMPORTANT]
> **Phase 0에서 세 가지 경로를 모두 테스트하여 가능성을 먼저 검증합니다.**

```
경로 A: 완전 스크래치 (사용자 요구 그대로)
────────────────────────────────────────
  - Tiny 모델 + 저해상도 (256~512)
  - 자사 데이터만으로 학습
  - Memory 모듈 제외 (이미지 전용)
  - 예상: loss 수렴 확인, 기본적 세그멘테이션 가능성 검증
  - 위험: 과적합, 범용성 부족

경로 B: ImageNet 사전학습 백본 + 나머지 스크래치
────────────────────────────────────────
  - Backbone만 ImageNet 사전학습 가중치 사용
  - Neck, Head, Memory는 스크래치
  - ★ 이것은 "SAM2 사전학습"이 아님 — 범용 시각 특징만 이용
  - 예상: 경로 A보다 수렴 속도 및 성능 크게 향상
  - 위험: 상대적으로 낮음

경로 C: SAM2 사전학습 + 파인튜닝 (비교 기준선)
────────────────────────────────────────
  - Facebook 체크포인트로 시작, 자사 데이터로 파인튜닝
  - Phase 0에서 "최선의 경우" 성능 기준선으로 사용
  - 경로 A/B의 결과를 이 기준선과 비교하여 판단

★ 세 경로 모두 구현하여 비교 → 실제 운영 가능한 경로 결정
```

### 7.4 소량 데이터 대응 전략

```
┌─────────────────────────────────────────────────────────────┐
│  소량 데이터로 학습 성능을 최대화하는 기법들                     │
│                                                             │
│  1. 강한 데이터 증강 (Heavy Augmentation)                      │
│     - 기하학적: 회전, 뒤집기, 아핀 변환, 크롭                   │
│     - 색상: 밝기, 대비, 채도, 그레이스케일 변환                  │
│     - 산업 특화: 조명 방향 변화, 표면 반사 시뮬, 노이즈           │
│     - Copy-Paste: 결함 영역을 잘라서 정상 이미지에 붙이기         │
│     → 데이터 100장 → 증강 후 실질 10,000장 이상 효과             │
│                                                             │
│  2. 정규화 (Regularization)                                    │
│     - Dropout, Weight Decay 강화                               │
│     - Label Smoothing                                          │
│     - Early Stopping (과적합 방지)                              │
│                                                             │
│  3. 모델 축소 (Model Downsizing)                               │
│     - Tiny 모델 사용 (39M → 충분히 학습 가능한 수준)              │
│     - 커스텀으로 더 작은 모델도 가능 (Backbone Stage 수 축소)      │
│                                                             │
│  4. Few-shot / Prompt Engineering                              │
│     - SAM2의 프롬프트 기반 특성 활용                             │
│     - 적은 데이터로도 프롬프트 → 마스크 매핑 학습 가능             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 개발 단계

### ★ Phase 0: 가능성 검증 (Feasibility Check) — 최우선

> [!WARNING]
> **본격 개발 전에 반드시 수행합니다.** 모델 코드(Phase 1)를 최소한으로 구성한 후, 즉시 학습 파이프라인을 붙여서 "학습이 되는가?"를 검증합니다.

**목표**: 제한된 데이터 + 단일 GPU + 스크래치 학습으로 loss가 수렴하고 의미 있는 마스크가 생성되는지 확인

1. **최소 모델 구성** (Tiny, 이미지 전용, Memory 제외)
2. **합성 데이터로 overfitting 테스트**
   - 10~20장의 이미지+마스크 쌍 (공개 데이터 또는 합성)
   - 이 소량 데이터에 과적합(overfitting) 시켜서 loss가 0에 수렴하는지 확인
   - → 수렴하면: 학습 파이프라인이 정상 동작하는 것
   - → 수렴 안 되면: 코드 또는 설정에 문제가 있는 것
3. **경로 A/B/C 비교 실험**
   - A: 완전 스크래치
   - B: ImageNet 백본 + 나머지 스크래치
   - C: Facebook 체크포인트 파인튜닝 (기준선)
   - 동일 데이터, 동일 epoch에서 loss 및 마스크 품질(IoU) 비교
4. **결과 보고서 작성** → 어떤 경로가 실현 가능한지 데이터 기반 판단

**산출물**: `scripts/feasibility_test.py`, 비교 결과 보고서

### Phase 1: 모델 코드 구성 (`models/`)
1. Backbone (`backbone/hiera.py`) — Hiera 트랜스포머 백본 분리
2. Neck (`neck/fpn_neck.py`) — FPN 특징 피라미드 분리
3. Head (`head/prompt_encoder.py`, `mask_decoder.py`) — 디코더 분리
4. Memory (`memory/`) — 메모리 모듈 분리
5. SAM2Base (`sam2_base.py`) — 전체 조합
6. Build (`build.py`) — 팩토리 함수 (Hydra 없이, YAML 선택적)
7. 스크래치 초기화 코드 (가중치 랜덤 초기화)
8. 체크포인트 로드 검증 (Facebook 체크포인트 경로 C용)

### Phase 2: 예측 코드 구성 (`predictor/`)
1. 이미지 Predictor — set_image → predict 패턴
2. 비디오 Predictor — init_state → add_points → propagate
3. 자동 마스크 생성기
4. Facebook/Ultralytics 비교 검증 스크립트

### Phase 3: 학습 코드 구성 (`training/`)
1. 데이터셋/데이터로더 (이미지 + 자사 커스텀)
2. Loss 함수 (Focal + Dice + IoU)
3. Trainer (단일 GPU, AMP, Gradient Checkpointing)
4. 데이터 증강 파이프라인 (소량 데이터 보완용)
5. 스크래치 학습 / 파인튜닝 모드 전환 지원

### Phase 4: 자사 데이터 학습 & 검증
1. 자사 결함 데이터셋 래퍼 구현
2. Phase 0 결과에 따른 최적 경로(A/B/C)로 본격 학습
3. 결과 비교: 각 경로별 성능
4. 비디오 추론 테스트 (이미지 학습 모델로)

---

## 확정 사항 (Open Questions 응답 반영)

> [!NOTE]
> 아래 사항은 사용자 확인을 거쳐 확정된 내용입니다. (2026-04-02)

| # | 질문 | 확정 답변 |
|---|---|---|
| 1 | **GPU 용도 분리** | 학습: RTX 4090 (24GB) / 추론(예측): RTX 5060 |
| 2 | **자사 데이터 현황** | 아직 미정 — 데이터가 확보되면 반영 |
| 3 | **추론 속도** | **30FPS 기준** (실시간 처리 목표) |
| 4 | **배포 환경** | **Windows**, 소프트웨어에 모듈로 탑재되는 형태 |
| 5 | **체크포인트 전략** | **선택형**: Facebook SAM2 체크포인트 사용 가능 + 스크래치(처음부터) 학습도 가능하게 **양쪽 모두 지원** |

---

## 9. 추론 성능 최적화 (30FPS 목표)

> [!WARNING]
> 추론 대상 GPU가 RTX 5060이므로, 30FPS 달성을 위한 최적화 전략이 필수입니다.

### 9.1 30FPS 달성을 위한 제약 분석

```
┌───────────────────────────────────────────────────────────────────┐
│  30FPS = 1프레임당 ~33ms 이내 처리 필요                             │
│                                                                   │
│  SAM2 추론 시간 구성 (추정, 1024×1024 기준):                         │
│  ┌─────────────────┬────────────┬──────────────────────┐          │
│  │ 구간             │ 시간 (ms)  │ 비고                  │          │
│  ├─────────────────┼────────────┼──────────────────────┤          │
│  │ Image Encoder   │ 15~40      │ 백본+FPN (가장 무거움)  │          │
│  │ Prompt Encoder  │ <1         │ 매우 가벼움            │          │
│  │ Mask Decoder    │ 3~8        │ 경량 디코더            │          │
│  │ Memory (비디오)  │ 5~15       │ 선택적                │          │
│  │ 후처리           │ 1~3        │ NMS, 리사이즈 등       │          │
│  └─────────────────┴────────────┴──────────────────────┘          │
│                                                                   │
│  ★ 핵심 병목: Image Encoder (Backbone + Neck)                      │
│  ★ 비디오 모드에서는 Memory 추가 → 더 큰 부담                       │
└───────────────────────────────────────────────────────────────────┘
```

### 9.2 최적화 전략

```
┌───────────────────────────────────────────────────────────────────┐
│  최적화 단계 (순서대로 적용)                                         │
│                                                                   │
│  1단계: 기본 최적화                                                 │
│  ├─ FP16/BF16 추론 (메모리 절반, 속도 1.5~2배)                       │
│  ├─ torch.compile() 적용 (PyTorch 2.x, 10~30% 향상)                │
│  └─ CUDA 그래프 캐싱 (반복 그래프 최적화)                             │
│                                                                   │
│  2단계: 모델 경량화                                                  │
│  ├─ Tiny 모델 사용 (39M 파라미터, 가장 빠름)                          │
│  ├─ 입력 해상도 축소 (1024→512, 속도 ~4배 향상)                       │
│  └─ 불필요 Head 제거 (multimask → single mask)                      │
│                                                                   │
│  3단계: 배포 최적화                                                  │
│  ├─ ONNX Runtime 변환 (Windows 네이티브 가속)                        │
│  ├─ TensorRT 변환 (NVIDIA GPU 전용 최적화)                           │
│  └─ 양자화 (INT8, 정확도 검증 후 적용)                                │
│                                                                   │
│  ★ 이미지 모드: Tiny + FP16 + 512 해상도 → 30FPS 달성 가능성 높음     │
│  ★ 비디오 모드: 동일 조건 + Memory 경량화 필요 → 20~30FPS 예상        │
└───────────────────────────────────────────────────────────────────┘
```

---

## 10. Windows 소프트웨어 배포 전략

### 10.1 배포 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│  Windows 소프트웨어 통합 구조                                      │
│                                                                 │
│  ┌───────────────────────────┐                                  │
│  │  기존 Windows 소프트웨어    │  ← C#/WPF 또는 기존 프레임워크     │
│  │  (HVS 메인 애플리케이션)    │                                  │
│  │                           │                                  │
│  │  ┌───────────────────┐    │                                  │
│  │  │  SAM2 추론 모듈    │    │  ← Python DLL 또는 REST API      │
│  │  │                   │    │                                  │
│  │  │  • ONNX Runtime   │    │  ← GPU 가속 (DirectML/CUDA)      │
│  │  │  • TensorRT       │    │  ← 선택적 추가 가속               │
│  │  │  • 전/후처리       │    │                                  │
│  │  └───────────────────┘    │                                  │
│  └───────────────────────────┘                                  │
│                                                                 │
│  배포 방식 후보:                                                   │
│  A) ONNX Runtime + DirectML (Windows 네이티브, 가장 호환성 높음)    │
│  B) TensorRT (NVIDIA 전용, 최고 성능)                              │
│  C) Python 프로세스 + REST API (가장 간단, 기존 앱과 분리)           │
│  D) PyInstaller 패키징 (독립 실행 파일)                             │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 모델 내보내기 포맷

```
hvs/exports/
├── onnx/
│   ├── sam2_image_encoder.onnx    # Backbone+Neck (가장 무거움)
│   ├── sam2_prompt_encoder.onnx   # 프롬프트 인코더
│   └── sam2_mask_decoder.onnx     # 마스크 디코더
├── tensorrt/
│   └── sam2_engine.trt            # TensorRT 엔진 (5060 최적)
└── config.json                    # 모델 메타정보 (입출력 shape 등)
```

---

## 11. 학습 모드 선택 시스템

> [!IMPORTANT]
> Facebook SAM2 체크포인트 기반 파인튜닝과 완전 스크래치 학습을 **사용자가 선택**할 수 있어야 합니다.

### 11.1 학습 모드 설정

```yaml
# configs/train/finetune_custom.yaml

training:
  # ★ 학습 모드 선택 (사용자가 변경)
  mode: "finetune"          # "finetune" | "scratch" | "backbone_only"
  
  # --- mode별 동작 ---
  # "finetune":     Facebook SAM2 체크포인트에서 시작, 전체 모델 파인튜닝
  # "scratch":      모든 가중치를 랜덤 초기화, 처음부터 학습 
  # "backbone_only": ImageNet 백본만 사용, Neck/Head/Memory는 스크래치

  checkpoint:
    facebook_checkpoint: "checkpoints/sam2.1_hiera_tiny.pt"  # finetune 모드용
    imagenet_backbone: "checkpoints/hiera_tiny_imagenet.pt"  # backbone_only 모드용
    resume_from: null        # 이전 학습 이어서 할 때 경로 지정
  
  # 모드별 학습률 자동 조정
  finetune:
    backbone_lr_scale: 0.1   # 백본은 낮은 학습률 (이미 학습됨)
    head_lr_scale: 1.0       # 헤드는 기본 학습률
  scratch:
    backbone_lr_scale: 1.0   # 모두 동일 학습률
    head_lr_scale: 1.0
```

### 11.2 빌드 팩토리 모드 선택 API

```python
# hvs/models/build.py

def build_sam2(
    model_size: str = "tiny",      # "tiny" | "base+" | "large"
    mode: str = "finetune",        # "finetune" | "scratch" | "backbone_only"
    checkpoint_path: str = None,   # Facebook 체크포인트 경로
    device: str = "cuda",
):
    """
    ★ 사용자 선택에 따라 3가지 초기화 모드를 지원
    
    - finetune:     Facebook 체크포인트 로드 → 미세 조정
    - scratch:      완전 랜덤 초기화 → 처음부터 학습
    - backbone_only: ImageNet 백본만 로드 → Neck/Head 스크래치
    """
    model = SAM2Base(...)
    
    if mode == "finetune":
        load_facebook_checkpoint(model, checkpoint_path)
    elif mode == "backbone_only":
        load_imagenet_backbone(model.backbone, checkpoint_path)
    elif mode == "scratch":
        initialize_weights(model)  # Xavier/Kaiming 초기화
    
    return model
```

---

## Verification Plan

### Phase 0 검증 (가능성 판단)
- **Overfitting 테스트**: 10~20장 데이터에 loss → 0 수렴 확인
- **경로 A/B/C 비교**: 동일 조건에서 50 epoch 학습 후 IoU 비교
- **GPU 메모리 프로파일링**: 5060 / 4090에서 모델별 메모리 사용량 측정

### 추론 성능 검증 (30FPS 목표)
- **FPS 벤치마크**: 5060에서 모델 크기별(Tiny/Base+) × 해상도(512/1024) FPS 측정
- **ONNX 변환 후 FPS 비교**: PyTorch vs ONNX Runtime vs TensorRT
- **30FPS 미달 시**: 모델 경량화 / 해상도 축소 / 파이프라인 최적화 순차 적용

### Automated Tests
- **모듈별 단위 테스트**: Backbone, Neck, Head, Memory 각각의 forward pass 검증
- **통합 테스트**: 모델 빌드 → 랜덤 초기화 → 학습 1 step → loss 정상 출력 검증
- **비교 테스트**: Facebook 원본 vs HVS 구현 동일 입력 shape 검증
- **학습 모드 테스트**: finetune / scratch / backbone_only 3가지 모드 전환 검증
- **ONNX 내보내기 테스트**: 변환 → 추론 → PyTorch 결과와 오차 비교

### Manual Verification
- Phase 0 결과에 따른 최적 경로 결정
- 자사 이미지 결함 검출 정성 평가
- 이미지 학습 모델의 비디오 추론 성능 확인
- 5060 추론 FPS 실측 및 30FPS 달성 여부 확인
- Windows 소프트웨어 통합 시 호환성 검증
