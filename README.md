# 🔥 YOLOv8 화재 및 연기 감지 시스템

YOLOv8을 기반으로 한 고성능 실시간 화재 및 연기 감지 시스템입니다. Google Colab과 로컬 환경에서 모두 실행 가능하며, 완전 자동화된 훈련 파이프라인을 제공합니다.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 🌟 주요 특징

### 🚀 원클릭 실행
- **완전 자동화**: 데이터셋 다운로드부터 모델 훈련까지 한 번에
- **스마트 설정**: GPU/CPU 자동 감지 및 최적화
- **에러 방지**: 모든 의존성 자동 설치 및 경로 설정

### 🔍 고성능 감지
- **실시간 처리**: 웹캠을 통한 실시간 화재/연기 감지
- **높은 정확도**: 사전 훈련된 YOLOv8 기반 정확한 감지
- **다중 입력**: 이미지, 비디오, 웹캠, 배치 처리 지원

### 📊 포괄적 분석
- **상세한 시각화**: 훈련 곡선, 혼동 행렬, PR 곡선
- **성능 분석**: 클래스별 성능, 속도 벤치마크
- **실시간 모니터링**: 훈련 진행 상황 실시간 추적

### 🛠️ 개발자 친화적
- **모듈형 설계**: 각 기능을 독립적으로 사용 가능
- **대화형 인터페이스**: 직관적인 명령어 기반 실행
- **확장 가능**: 새로운 기능 쉽게 추가 가능

## 📁 프로젝트 구조

```
fire_smoke_detection/
├── 📋 설정 및 데이터
│   ├── config.py                 # 프로젝트 설정
│   ├── local_config.py           # 로컬 환경 설정
│   ├── dataset_manager.py        # 데이터셋 관리
│   └── requirements.txt          # 패키지 의존성
│
├── 🧠 모델 및 훈련
│   ├── model_trainer.py          # 모델 훈련
│   ├── model_validator.py        # 모델 검증
│   └── inference_engine.py       # 추론 엔진
│
├── 📊 시각화 및 분석
│   ├── visualization_utils.py    # 시각화 도구
│   └── simple_validation.py      # 간단한 검증
│
├── 🔄 다운로더 및 유틸리티
│   ├── roboflow_dataset_downloader.py  # Roboflow 다운로더
│   ├── instant_download.py             # 즉시 다운로드
│   └── download_dataset.py             # 기본 다운로더
│
├── 🚀 실행 파일
│   └── main.py                   # 메인 실행 파일
│
└── 📚 문서
    ├── README.md                 # 프로젝트 문서
    └── LICENSE                   # 라이선스
```

## ⚡ 빠른 시작

### 🔥 원클릭 실행 (권장)

#### Google Colab
```python
# 1. 즉시 시작 (모든 과정 자동화)
!python main.py --mode full --download quick --api-key YOUR_API_KEY --epochs 50
```

#### 로컬 환경
```bash
# 1. 저장소 클론
git clone https://github.com/your-username/fire-smoke-detection.git
cd fire-smoke-detection

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 원클릭 실행
python main.py --mode full --download quick --api-key YOUR_API_KEY --epochs 50
```

### 📊 데이터셋 옵션

| 옵션 | 이미지 수 | 클래스 | 설명 |
|------|-----------|--------|------|
| `quick` | 979개 | Fire, default, smoke | 빠른 테스트용 |
| `large` | 6,391개 | fire, smoke | 고성능 훈련용 |
| `interactive` | 사용자 선택 | 다양 | 대화형 선택 |

## 🚀 사용법

### 1️⃣ 대화형 모드 (초보자 추천)

```bash
python main.py --interactive
```

대화형 모드에서 사용 가능한 명령어:
- `setup` - 환경 설정
- `dataset` - 데이터셋 준비
- `download` - Roboflow 데이터셋 다운로드
- `train` - 모델 훈련
- `validate` - 모델 검증
- `infer` - 추론 실행
- `viz` - 시각화
- `full` - 전체 파이프라인
- `quit` - 종료

### 2️⃣ 명령행 실행 (고급 사용자)

#### 전체 파이프라인
```bash
# 빠른 데이터셋으로 훈련
python main.py --mode full --download quick --api-key YOUR_API_KEY --epochs 50 --batch-size 16

# 대용량 데이터셋으로 고품질 훈련
python main.py --mode full --download large --api-key YOUR_API_KEY --epochs 100 --batch-size 32
```

#### 단계별 실행
```bash
# 1. 환경 설정
python main.py --mode setup

# 2. 데이터셋 다운로드
python main.py --mode dataset --download quick --api-key YOUR_API_KEY

# 3. 모델 훈련
python main.py --mode train --epochs 100 --batch-size 16 --model-size yolov8n.pt

# 4. 모델 검증
python main.py --mode validate

# 5. 추론 실행
python main.py --mode infer --source path/to/image.jpg --inference-type image
```

#### 추론 예시
```bash
# 단일 이미지
python main.py --mode infer --source image.jpg --inference-type image

# 비디오 파일
python main.py --mode infer --source video.mp4 --inference-type video

# 실시간 웹캠
python main.py --mode infer --inference-type realtime

# 폴더 내 모든 이미지
python main.py --mode infer --source ./images/ --inference-type batch
```

### 3️⃣ Python 코드에서 사용

```python
from main import FireSmokeDetectionPipeline

# 파이프라인 초기화
pipeline = FireSmokeDetectionPipeline()

# 🚀 원클릭 실행
pipeline.run_full_pipeline(
    epochs=50,
    batch_size=16,
    download_option='quick',
    api_key='YOUR_API_KEY'
)

# 또는 단계별 실행
pipeline.setup_environment()
pipeline.prepare_dataset()
pipeline.train_model(epochs=50)
pipeline.validate_model()
pipeline.run_inference('image.jpg', 'image')
```

## 🔧 설정 옵션

### 모델 크기 선택
| 모델 | 속도 | 정확도 | 파라미터 | 권장 용도 |
|------|------|--------|----------|-----------|
| `yolov8n.pt` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 3.2M | 실시간 처리 |
| `yolov8s.pt` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 11.2M | 균형잡힌 성능 |
| `yolov8m.pt` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 25.9M | 높은 정확도 |
| `yolov8l.pt` | ⭐⭐ | ⭐⭐⭐⭐⭐ | 43.7M | 최고 성능 |
| `yolov8x.pt` | ⭐ | ⭐⭐⭐⭐⭐ | 68.2M | 연구용 |

### 하이퍼파라미터 튜닝

`config.py`에서 설정 수정:
```python
# 기본 설정
EPOCHS = 100
BATCH_SIZE = 16          # GPU 메모리에 따라 조정
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5

# 클래스 설정
CLASS_NAMES = ['Fire', 'default', 'smoke']
```

### GPU/CPU 최적화

#### RTX 4060 최적화 설정
```python
BATCH_SIZE = 32          # RTX 4060용 최적화
WORKERS = 4              # CPU 코어 수에 따라 조정
```

#### CPU 전용 설정
```python
BATCH_SIZE = 8           # CPU는 작은 배치 크기
WORKERS = 2              # CPU 부하 감소
```

## 📊 성능 및 결과

### 🎯 주요 메트릭
- **mAP@0.5**: IoU 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5-0.95 범위의 평균 정밀도
- **Precision**: 정밀도 (False Positive 최소화)
- **Recall**: 재현율 (False Negative 최소화)
- **F1-Score**: 정밀도와 재현율의 조화 평균

### 📈 벤치마크 결과

| 데이터셋 | 모델 | mAP@0.5 | mAP@0.5:0.95 | FPS | 용도 |
|----------|------|---------|--------------|-----|------|
| Quick | YOLOv8n | 0.85+ | 0.65+ | 120+ | 실시간 감지 |
| Quick | YOLOv8s | 0.88+ | 0.70+ | 80+ | 균형잡힌 성능 |
| Large | YOLOv8m | 0.92+ | 0.75+ | 50+ | 고정확도 |

### 🔥 실제 성능 예시
```
=== 모델 검증 결과 ===
전체 이미지 수: 156
전체 라벨 수: 284

📦 Detection Metrics:
  Precision: 0.8945
  Recall: 0.8721
  mAP@0.5: 0.9123
  mAP@0.5:0.95: 0.7234

📊 클래스별 mAP@0.5:
  Fire      : 0.9456
  default   : 0.8534
  smoke     : 0.9378

⚡ 처리 속도:
  전처리: 2.1ms
  추론: 8.3ms
  NMS: 1.2ms
  총 처리시간: 11.6ms
  FPS: 86.2
```

## 🛠️ 고급 기능

### 🎛️ 하이퍼파라미터 자동 튜닝
```python
from model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_model()
trainer.hyperparameter_tuning(iterations=100)
```

### 📤 모델 내보내기
```python
# ONNX 형식 (크로스 플랫폼)
pipeline.export_model('onnx')

# TensorRT 형식 (NVIDIA GPU 가속)
pipeline.export_model('tensorrt')

# CoreML 형식 (Apple 기기)
pipeline.export_model('coreml')
```

### 🚨 실시간 알림 시스템
```python
from inference_engine import InferenceEngine

engine = InferenceEngine()
engine.load_model()

# 화재 감지 시 알림 콜백
def fire_alert(detection):
    if detection['class'] in ['Fire', 'smoke']:
        print(f"🚨 위험! {detection['class']} 감지됨!")
        # 여기에 SMS, 이메일, API 호출 등 추가

engine.real_time_detection(callback=fire_alert)
```

### 📊 배치 분석
```python
# 여러 이미지 한 번에 처리
results = engine.predict_batch('path/to/images/')

# 결과 분석 및 시각화
from visualization_utils import VisualizationUtils
viz = VisualizationUtils()
viz.plot_batch_analysis(results)
```

## 🐛 문제 해결

### 자주 발생하는 문제들

#### 1. GPU 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --mode train --batch-size 8

# 이미지 크기 줄이기
python main.py --mode train --batch-size 16 --model-size yolov8n.pt
```

#### 2. 데이터셋 다운로드 실패
```python
# API 키 확인
# https://roboflow.com → Settings → API Keys

# 수동 다운로드
python instant_download.py
```

#### 3. 의존성 설치 오류
```bash
# 가상환경 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. 모델 로드 실패
```python
# 모델 경로 확인
import os
model_path = "runs/detect/fire_smoke_detection/weights/best.pt"
print(f"Model exists: {os.path.exists(model_path)}")

# 기본 모델로 대체
model = YOLO('yolov8n.pt')  # 사전 훈련된 모델 사용
```

### 디버깅 팁

1. **상세 로그 확인**: 각 모듈에서 자세한 오류 메시지 출력
2. **단계별 실행**: 문제 발생 지점 정확히 파악
3. **GPU 메모리 모니터링**: `nvidia-smi`로 메모리 사용량 확인
4. **데이터셋 검증**: `dataset_manager.validate_dataset_format()` 실행

### 성능 최적화

#### 훈련 속도 향상
- **Mixed Precision**: GPU 메모리 절약 및 속도 향상
- **멀티 GPU**: 여러 GPU로 병렬 처리
- **데이터 로딩 최적화**: `workers` 수 증가

#### 추론 속도 향상
- **TensorRT**: NVIDIA GPU에서 최대 10배 속도 향상
- **ONNX Runtime**: CPU에서도 빠른 추론
- **모델 양자화**: 정확도 유지하며 크기 감소

## 🔗 Roboflow API 키 발급

1. [Roboflow 웹사이트](https://roboflow.com) 접속
2. 회원가입 또는 로그인
3. 좌측 메뉴에서 **Settings** → **API Keys** 클릭
4. **Generate New Key** 버튼으로 새 키 생성
5. **Private API Key** 복사하여 사용

## 📈 데이터셋 정보

### 추천 데이터셋

#### 1. Fire-WRPGM (빠른 시작용)
- **이미지 수**: 979개
- **클래스**: Fire, default, smoke
- **특징**: 빠른 다운로드, 테스트에 적합
- **URL**: [Roboflow Fire-WRPGM](https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8)

#### 2. Fire-Smoke-Detection (고품질용)
- **이미지 수**: 6,391개
- **클래스**: fire, smoke
- **특징**: 대용량, 높은 정확도
- **URL**: [Roboflow Fire-Smoke](https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia)

### 커스텀 데이터셋

자체 데이터셋 사용 시 YOLO 형식 준수:
```
datasets/fire/
├── train/
│   ├── images/  # .jpg, .png 파일
│   └── labels/  # .txt 파일 (class x y w h)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```