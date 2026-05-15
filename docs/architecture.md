# 아키텍처 (Architecture)

> Fire & Smoke Detection — YOLOv8 기반 화재/연기 감지 시스템 (단일 사용자 MVP)

본 문서는 본 프로젝트의 전체 아키텍처, 디렉토리 구성, 주요 컴포넌트, 데이터 흐름을 정리한다.

---

## 1. 시스템 개요

- **목적**: 업로드된 이미지/비디오에 대해 YOLOv8 모델로 화재(Fire) · 연기(smoke)를 감지하고 결과를 시각화/다운로드 제공.
- **형태**: SPA 프론트엔드 ↔ REST API 백엔드 ↔ 추론 엔진 + 파일 스토리지.
- **사용자 모델**: 단일 사용자 MVP. GPU 점유 직렬화를 위한 `asyncio.Lock` 사용.

### 1.1 기술 스택

| 레이어 | 기술 |
| --- | --- |
| Frontend | React 18 + TypeScript + Vite + TailwindCSS |
| Backend API | FastAPI (Python 3) + Uvicorn |
| ML Runtime | Ultralytics YOLOv8 + PyTorch (CUDA 우선, CPU 폴백) |
| Image / Video I/O | OpenCV (cv2) |
| Dataset | Roboflow 다운로더 (fire-wrpgm v8) |
| Storage | 로컬 파일시스템 (`backend/storage/`) |
| Inter-process Queue | In-memory dict + `asyncio.create_task` |

---

## 2. 컴포넌트 구성도

```
┌────────────────────────────┐        HTTP / JSON         ┌──────────────────────────────────────┐
│        Frontend            │  ─────────────────────►    │              Backend                 │
│  React + Vite (5173)       │                            │           FastAPI (8000)             │
│                            │                            │                                      │
│  ┌────────────┐            │                            │  ┌────────────┐  ┌────────────────┐  │
│  │  App.tsx   │            │                            │  │  routes/   │  │   services/    │  │
│  │  Tab UI    │            │                            │  │  detect    │  │  inference     │  │
│  └────┬───────┘            │                            │  │  jobs      │  │  jobs (queue)  │  │
│       │                    │                            │  │  model     │  └───────┬────────┘  │
│  ┌────▼──────┐ ┌─────────┐ │                            │  └─────┬──────┘          │           │
│  │ImageDetect│ │VideoDet.│ │                            │        │                 │           │
│  └────┬──────┘ └────┬────┘ │                            │        ▼                 ▼           │
│       │             │      │                            │  ┌──────────────────────────────┐    │
│  ┌────▼─────────────▼────┐ │                            │  │   InferenceEngine (1회 로드) │    │
│  │   api/client.ts       │ │   /api/detect/image       │  │   Ultralytics YOLO (best.pt) │    │
│  │   fetch + types       │ │   /api/detect/video       │  └──────────────┬───────────────┘    │
│  └────────┬──────────────┘ │   /api/jobs/{id}          │                 │                    │
│           │                │   /api/jobs/{id}/result   │                 ▼                    │
│           │                │   /api/model/info         │  ┌──────────────────────────────┐    │
│           ▼                │   /api/files/{name}       │  │       storage/              │    │
│   브라우저 (사용자)         │ ◄─────────────────────    │  │   uploads/  results/  jobs/ │    │
└────────────────────────────┘                            └──────────────────────────────────────┘
```

---

## 3. 디렉토리 구조

```
fire_smoke_detection/
├── backend/                        # FastAPI 서버 + ML 파이프라인
│   ├── main.py                     # FastAPI 진입점 (lifespan에서 엔진 로드)
│   ├── config.py                   # 전역 설정 (경로, 클래스, 모델 크기 등)
│   ├── schemas.py                  # Pydantic 응답/요청 모델
│   ├── data.yaml                   # Ultralytics 데이터셋 설정
│   ├── requirements.txt
│   │
│   ├── routes/                     # FastAPI 라우터 (HTTP 레이어)
│   │   ├── detect.py               # POST /api/detect/{image,video}
│   │   ├── jobs.py                 # GET  /api/jobs/{id}, /result
│   │   └── model.py                # GET  /api/model/info, /api/files/{name}
│   │
│   ├── services/                   # 도메인 서비스 (HTTP 비의존)
│   │   ├── inference.py            # InferenceEngine 싱글톤 관리
│   │   └── jobs.py                 # 비디오 잡 큐 + 진행률 트래킹
│   │
│   ├── storage/                    # 로컬 파일 스토리지 (런타임 생성)
│   │   ├── uploads/                # 원본 업로드 파일
│   │   ├── results/                # 감지 결과 (annotated 이미지/비디오)
│   │   └── jobs/                   # (예약) 잡 메타 정보
│   │
│   ├── inference_engine.py         # YOLO 래퍼 — 이미지/비디오/실시간 추론
│   ├── model_trainer.py            # 학습 파이프라인
│   ├── model_validator.py          # 모델 검증 + 메트릭 산출
│   ├── dataset_manager.py          # 데이터셋 구조/검증
│   ├── roboflow_dataset_downloader.py
│   ├── instant_download.py         # 빠른 데이터셋 부트스트랩
│   ├── download_dataset.py
│   ├── simple_validation.py
│   ├── visualization_utils.py      # plot/saving 헬퍼
│   └── cli.py                      # 학습/검증/추론 CLI 통합 인터페이스
│
├── frontend/                       # React + Vite SPA
│   ├── src/
│   │   ├── main.tsx                # React 부트스트랩
│   │   ├── App.tsx                 # 탭 라우팅 (이미지 / 비디오)
│   │   ├── types.ts                # 백엔드 schemas.py 의 TS 미러
│   │   ├── api/client.ts           # fetch 래퍼 + 엔드포인트 함수
│   │   ├── pages/
│   │   │   ├── ImageDetect.tsx     # 이미지 업로드 → 결과 표시
│   │   │   └── VideoDetect.tsx     # 비디오 업로드 → 잡 폴링
│   │   └── components/
│   │       ├── DropZone.tsx        # 파일 드래그&드롭
│   │       ├── ModelInfoCard.tsx   # 모델 메트릭 카드
│   │       ├── DetectionTable.tsx  # 감지 결과 표
│   │       └── JobProgress.tsx     # 진행률 표시
│   ├── vite.config.ts              # /api → 127.0.0.1:8000 프록시
│   └── package.json
│
├── datasets/                       # 학습 데이터셋 (Roboflow)
├── runs/                           # 학습/추론 산출물 (Ultralytics)
└── docs/
    ├── architecture.md             # 본 문서
    └── uml.md                      # UML 다이어그램
```

---

## 4. 레이어 책임

### 4.1 Backend — 3-Tier 분리

| 레이어 | 모듈 | 책임 |
| --- | --- | --- |
| **Presentation** (HTTP) | `routes/*.py` | URL 라우팅, 요청 검증, 응답 직렬화, 업로드 처리 (`UploadFile`) |
| **Service** (도메인) | `services/inference.py`, `services/jobs.py` | 모델 싱글톤 생명주기, 잡 큐/상태, 동시성 락 |
| **Domain / Core** | `inference_engine.py`, `model_trainer.py`, `model_validator.py`, `dataset_manager.py` | YOLO 호출, 파싱, 학습/검증, 데이터셋 구조 |

라우터는 도메인 클래스를 직접 호출하지 않고 항상 `services/`를 거친다 → HTTP 레이어와 ML 레이어의 결합도 분리.

### 4.2 Frontend — Page / Component / API 분리

- **Page (`pages/`)**: 화면 단위 상태/오케스트레이션. (`ImageDetect`, `VideoDetect`)
- **Component (`components/`)**: 재사용 가능한 표시 단위 (props만 받음).
- **API (`api/client.ts`)**: 모든 fetch 호출의 단일 집결지. 백엔드 스키마 변경 시 `types.ts`와 동시 수정.

---

## 5. 데이터 흐름

### 5.1 이미지 감지 (동기 1회)

```
[브라우저]  파일 선택
   │  POST /api/detect/image  (multipart)
   ▼
[routes/detect.py]
   ├── 확장자/용량 검증 → uploads/{uuid}.{ext} 저장
   ├── _infer_lock 획득
   ├── asyncio.to_thread(eng.model.predict, ...)   ← Ultralytics 동기 호출
   ├── _parse_detections() → [{cls, conf, bbox}]
   ├── result.plot() → results/{uuid}_annotated.jpg
   └── ImageDetectionResult JSON 응답
   ▼
[브라우저]  annotated 이미지 + 감지 테이블 렌더
```

### 5.2 비디오 감지 (비동기 잡 + 폴링)

```
[브라우저]  파일 선택
   │  POST /api/detect/video  (multipart)
   ▼
[routes/detect.py]
   ├── uploads/{job_id}.{ext} 저장
   ├── enqueue_video_job(job_id, ...)        ← _jobs dict 등록 + create_task
   └── 202 Accepted  {job_id}
   ▼
[services/jobs.py::_process_video]   (백그라운드 코루틴)
   ├── 메타 추출 (fps/wh/total_frames) — cv2.VideoCapture
   ├── _infer_lock 획득
   ├── asyncio.to_thread(_run)               ← Ultralytics stream=True
   │     for each frame:
   │       writer.write(r.plot())
   │       det_total += len(parse(r))
   │       매 5프레임 → _update(progress, ...)
   └── status="done", result_name 등록
        ▲
        │  GET /api/jobs/{id}   (800ms 주기 폴링)
[브라우저]  진행률 바 갱신
        │  status=="done" 도달 시
        │  GET /api/files/{name} 또는 /api/jobs/{id}/result
        ▼
[브라우저]  annotated 비디오 재생 + 다운로드 링크
```

### 5.3 모델 메타 정보

```
[브라우저 마운트]  GET /api/model/info
   ▼
[routes/model.py]
   ├── get_engine() → 가중치 경로 / device
   └── test_validation_report.json 있으면 map50/precision/recall 동봉
   ▼
[브라우저]  ModelInfoCard 표시
```

---

## 6. 동시성 모델

- **모델 1 인스턴스**: `services/inference.py`가 더블체크 락으로 `InferenceEngine` 싱글톤 보장.
- **GPU 직렬화**: `routes/detect.py`의 모듈 레벨 `asyncio.Lock _infer_lock` 사용. 이미지/비디오 모두 동일 락 사용 → 한 번에 하나만 추론.
- **동기 호출 격리**: Ultralytics는 동기 API이므로 `asyncio.to_thread()`로 워커 스레드에서 실행, 이벤트 루프 블로킹 방지.
- **잡 상태 보호**: `services/jobs.py`의 `_jobs` dict는 `threading.Lock`으로 보호 (스레드와 코루틴이 함께 접근).
- **진행률 갱신 빈도**: 매 프레임이 아닌 5프레임마다 갱신 → 락 경합 완화.

---

## 7. 스토리지 규약

| 디렉토리 | 내용 | 명명 규칙 |
| --- | --- | --- |
| `storage/uploads/` | 원본 업로드 | `{image_id|job_id}.{ext}` |
| `storage/results/` | 결과 이미지/비디오 | `{image_id}_annotated.jpg`, `{job_id}_annotated.mp4` |
| `storage/jobs/` | (예약) 잡 메타데이터 | — (현재는 in-memory) |

- 외부 노출은 `/api/files/{filename}` (storage/results 또는 storage/uploads 탐색).
- 파일 용량 한도: 이미지 20MB, 비디오 200MB. 초과 시 HTTP 413.

---

## 8. 설정 / 환경

- **`Config.CLASS_NAMES`** = `['Fire', 'smoke']`, `NUM_CLASSES = 2`
- **모델 검색 순서** (`InferenceEngine.load_model`):
  1. `runs/detect/fire_smoke_detection/weights/best.pt`
  2. `runs/detect/train/weights/best.pt`
  3. `yolov8n.pt` (사전훈련 폴백)
- **CORS**: `http://localhost:5173`만 허용 (Vite dev server).
- **Vite dev proxy**: `/api` → `http://127.0.0.1:8000`로 포워딩.

---

## 9. 학습 / 검증 파이프라인 (CLI)

브라우저 UI와 별도로 `backend/cli.py`가 학습/검증/추론 일괄 실행을 제공한다.

```
FireSmokeDetectionPipeline
  ├── setup_environment()         → initialize_project() (config.py)
  ├── prepare_dataset()           → DatasetManager
  ├── train_model()               → ModelTrainer (Ultralytics)
  ├── validate_model()            → ModelValidator → metrics JSON
  └── run_inference()             → InferenceEngine (이미지/비디오/실시간)
```

- 학습 산출물: `runs/detect/.../weights/best.pt`
- 검증 결과: `backend/test_validation_report.json` (← API의 ModelInfoCard 메트릭 출처)

---

## 10. 확장 방향 (현재는 미구현)

| 영역 | 현재 | 확장 시 후보 |
| --- | --- | --- |
| 잡 큐 | in-memory dict | Redis / Celery / RQ |
| 인증 | 없음 (단일 사용자) | API Key / OAuth |
| 스토리지 | 로컬 FS | S3 / MinIO + presigned URL |
| 잡 영속성 | 메모리 (재시작 시 손실) | SQLite / Postgres + `storage/jobs/` |
| 동시 추론 | Lock으로 직렬화 | 모델 N 인스턴스 풀 + GPU 큐 |
| 실시간 진행률 | 폴링 (800ms) | Server-Sent Events / WebSocket |
