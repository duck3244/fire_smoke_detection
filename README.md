# 🔥 YOLOv8 화재 및 연기 감지 시스템

YOLOv8을 사용한 실시간 화재 및 연기 감지 시스템입니다. 이 프로젝트는 Google Colab 환경에 최적화되어 있으며, 화재 안전 시스템에 적용할 수 있는 고성능 AI 모델을 제공합니다.

## 🎯 주요 기능

- **실시간 감지**: 웹캠을 통한 실시간 화재/연기 감지
- **배치 처리**: 여러 이미지를 한 번에 처리
- **비디오 분석**: 비디오 파일에서 화재/연기 감지
- **성능 시각화**: 상세한 훈련 및 검증 결과 분석
- **모델 내보내기**: ONNX, TensorRT 등 다양한 형식 지원

## 🏗️ 프로젝트 구조

```
fire_smoke_detection/
├── config.py              # 설정 및 환경 구성
├── dataset_manager.py     # 데이터셋 관리
├── model_trainer.py       # 모델 훈련
├── model_validator.py     # 모델 검증
├── inference_engine.py    # 추론 엔진
├── visualization_utils.py # 시각화 유틸리티
├── main.py               # 메인 실행 파일
├── requirements.txt      # 필수 패키지 목록
└── README.md            # 프로젝트 설명서
```

## 🚀 빠른 시작

### 1. 환경 설정

#### Google Colab에서 실행
```python
# 저장소 클론
!git clone <repository-url>
%cd fire_smoke_detection

# 패키지 설치
!pip install -r requirements.txt

# 메인 스크립트 실행
!python main.py --interactive
```

#### 로컬 환경에서 실행
```bash
# 저장소 클론
git clone <repository-url>
cd fire_smoke_detection

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt

# 대화형 모드 실행
python main.py --interactive
```

### 2. 데이터셋 준비

프로젝트는 다음과 같은 데이터셋 구조를 기대합니다:

```
datasets/fire/
├── train/
│   ├── images/  # 훈련용 이미지
│   └── labels/  # 훈련용 라벨 (.txt)
├── valid/
│   ├── images/  # 검증용 이미지
│   └── labels/  # 검증용 라벨 (.txt)
└── test/
    ├── images/  # 테스트용 이미지
    └── labels/  # 테스트용 라벨 (.txt)
```

**추천 데이터셋:**
- [Roboflow Fire Dataset](https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8)
- 직접 라벨링한 화재/연기 이미지

## 💻 사용법

### 🔥 완전 자동화 실행

#### 명령행에서 원클릭 실행
```bash
# 빠른 데이터셋으로 전체 파이프라인 (권장)
python main.py --mode full --download quick --api-key YOUR_API_KEY --epochs 50

# 대용량 데이터셋으로 실행
python main.py --mode full --download large --api-key YOUR_API_KEY --epochs 100

# 대화형 데이터셋 선택
python main.py --mode full --download interactive --epochs 50
```

#### 단계별 실행
```bash
# 1. 환경 설정
python main.py --mode setup

# 2. 데이터셋 다운로드
python main.py --mode dataset --download quick --api-key YOUR_API_KEY

# 3. 모델 훈련
python main.py --mode train --epochs 50 --batch-size 16

# 4. 모델 검증
python main.py --mode validate

# 5. 추론 실행
python main.py --mode infer --source path/to/image.jpg --inference-type image
```

### 대화형 모드 (추천)

```bash
python main.py --interactive
```

대화형 모드에서는 다음 명령어들을 사용할 수 있습니다:

1. `setup` - 환경 설정
2. `dataset` - 데이터셋 준비
3. `download` - Roboflow 데이터셋 다운로드
4. `train` - 모델 훈련
5. `validate` - 모델 검증
6. `infer` - 추론 실행
7. `viz` - 시각화
8. `full` - 전체 파이프라인
9. `quit` - 종료

### Python 코드에서 사용

```python
from main import FireSmokeDetectionPipeline

# 파이프라인 초기화
pipeline = FireSmokeDetectionPipeline()

# 🚀 원클릭 실행 (권장)
pipeline.run_full_pipeline(
    epochs=50,
    download_option='quick',  # 'large', 'interactive', None
    api_key='YOUR_API_KEY'
)

# 또는 단계별 실행
pipeline.setup_environment()
pipeline.prepare_dataset(download_option='quick', api_key='YOUR_API_KEY')
pipeline.train_model(epochs=50, batch_size=16)
pipeline.validate_model()
pipeline.run_inference('path/to/image.jpg', 'image')
```

### Roboflow 데이터셋만 다운로드

```python
# 빠른 다운로드
from dataset_manager import DatasetManager
manager = DatasetManager()

# 옵션 1: 빠른 다운로드 (979개 이미지)
manager.quick_download_fire_dataset('YOUR_API_KEY')

# 옵션 2: 대용량 다운로드 (6,391개 이미지)  
manager.quick_download_large_dataset('YOUR_API_KEY')

# 옵션 3: 대화형 선택
manager.download_roboflow_dataset(interactive=True)
```

## 🎛️ 설정 옵션

### 모델 크기 선택
- `yolov8n.pt` - Nano (가장 빠름, 정확도 보통)
- `yolov8s.pt` - Small (빠름, 정확도 좋음)
- `yolov8m.pt` - Medium (보통, 정확도 높음)
- `yolov8l.pt` - Large (느림, 정확도 매우 높음)
- `yolov8x.pt` - Extra Large (가장 느림, 최고 정확도)

### 하이퍼파라미터 조정

`config.py`에서 다음 설정들을 수정할 수 있습니다:

```python
# 모델 설정
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5

# 클래스 설정
CLASS_NAMES = ['Fire', 'default', 'smoke']
```

## 📊 성능 평가

### 주요 메트릭
- **mAP@0.5**: IoU 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5-0.95 범위의 평균 정밀도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1-Score**: F1 점수

### 시각화
시스템은 다음과 같은 시각화를 제공합니다:

- 훈련 곡선 (손실, 정확도)
- 혼동 행렬
- PR 곡선
- 클래스별 성능 분석
- 감지 결과 시각화

## 🔧 고급 기능

### 하이퍼파라미터 튜닝
```python
from model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_model()
trainer.hyperparameter_tuning(iterations=100)
```

### 모델 내보내기
```python
# ONNX 형식으로 내보내기
pipeline.export_model('onnx')

# TensorRT 형식으로 내보내기 (NVIDIA GPU 환경)
pipeline.export_model('tensorrt')
```

### 실시간 알림 시스템
```python
from inference_engine import InferenceEngine

engine = InferenceEngine()
engine.load_model()

# 실시간 감지 시 알림 콜백 설정
def fire_detected_callback(detection):
    if detection['class'] in ['Fire', 'smoke']:
        print(f"🚨 ALERT: {detection['class']} detected!")
        # 여기에 알림 로직 추가 (이메일, SMS 등)

engine.real_time_detection()
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. CUDA 메모리 부족
```python
# 배치 크기 줄이기
python main.py --mode train --batch-size 8
```

#### 2. 데이터셋 형식 오류
```python
# 데이터셋 검증 실행
from dataset_manager import DatasetManager
manager = DatasetManager()
manager.validate_dataset_format()
```

#### 3. 모델 로드 실패
```python
# 모델 경로 확인
import os
model_path = "/content/runs/detect/fire_smoke_detection/weights/best.pt"
print(f"Model exists: {os.path.exists(model_path)}")
```

### 로그 및 디버깅
자세한 로그는 각 모듈에서 출력됩니다. 문제 발생 시 다음을 확인하세요:

1. GPU 메모리 사용량
2. 데이터셋 형식 및 경로
3. 모델 파일 존재 여부
4. Python 패키지 버전

## 📈 성능 최적화

### 훈련 속도 향상
- Mixed Precision 사용: `--half` 옵션
- 더 작은 이미지 크기 사용: `--imgsz 416`
- 데이터 로딩 워커 수 증가: `--workers 8`

### 추론 속도 향상
- TensorRT 엔진 사용 (NVIDIA GPU)
- ONNX Runtime 사용
- 모델 양자화 적용

---

**⚠️ 주의사항**: 이 시스템은 교육 및 연구 목적으로 개발되었습니다. 실제 화재 안전 시스템에 적용하기 전에 충분한 테스트와 검증이 필요합니다.