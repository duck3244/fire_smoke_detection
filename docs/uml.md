# UML 다이어그램 (UML Diagrams)

> Fire & Smoke Detection — 컴포넌트/클래스/시퀀스/상태/배포 다이어그램

본 문서는 [Mermaid](https://mermaid.js.org/) 기반 UML 다이어그램을 모은다. GitHub/IDE 마크다운 프리뷰에서 그대로 렌더링된다.

---

## 1. 컴포넌트 다이어그램 (시스템 수준)

```mermaid
flowchart LR
    subgraph BROWSER["사용자 브라우저"]
        UI["React SPA<br/>Vite Dev 5173"]
    end

    subgraph FE["Frontend (src/)"]
        APP["App.tsx"]
        PG_IMG["pages/ImageDetect"]
        PG_VID["pages/VideoDetect"]
        CMP["components/<br/>DropZone, JobProgress,<br/>DetectionTable, ModelInfoCard"]
        API_CLIENT["api/client.ts"]
    end

    subgraph BE["Backend (FastAPI 8000)"]
        RT_DETECT["routes/detect.py"]
        RT_JOBS["routes/jobs.py"]
        RT_MODEL["routes/model.py"]
        SV_INF["services/inference.py<br/>InferenceEngine Singleton"]
        SV_JOBS["services/jobs.py<br/>In-memory Job Queue"]
        ENGINE["inference_engine.py<br/>InferenceEngine"]
    end

    subgraph ML["ML Runtime"]
        YOLO["Ultralytics YOLOv8"]
        TORCH["PyTorch (CUDA / CPU)"]
    end

    subgraph FS["Local File Storage"]
        UP["storage/uploads"]
        RES["storage/results"]
        WTS["runs/detect/.../best.pt"]
        VAL["test_validation_report.json"]
    end

    UI --> APP
    APP --> PG_IMG
    APP --> PG_VID
    PG_IMG --> CMP
    PG_VID --> CMP
    PG_IMG --> API_CLIENT
    PG_VID --> API_CLIENT
    CMP --> API_CLIENT

    API_CLIENT -- "/api/detect/image" --> RT_DETECT
    API_CLIENT -- "/api/detect/video" --> RT_DETECT
    API_CLIENT -- "/api/jobs/{id}" --> RT_JOBS
    API_CLIENT -- "/api/model/info" --> RT_MODEL
    API_CLIENT -- "/api/files/{name}" --> RT_MODEL

    RT_DETECT --> SV_INF
    RT_DETECT --> SV_JOBS
    RT_JOBS --> SV_JOBS
    RT_MODEL --> SV_INF
    RT_MODEL --> VAL

    SV_INF --> ENGINE
    SV_JOBS --> ENGINE
    ENGINE --> YOLO
    YOLO --> TORCH
    ENGINE --> WTS

    RT_DETECT --> UP
    RT_DETECT --> RES
    SV_JOBS --> UP
    SV_JOBS --> RES
    RT_MODEL --> RES
    RT_MODEL --> UP
```

---

## 2. 클래스 다이어그램 — Backend Core

```mermaid
classDiagram
    direction LR

    class Config {
        <<static>>
        +HOME : str
        +DATASET_BASE_PATH : str
        +RESULTS_PATH : str
        +MODEL_SIZE : str = "yolov8n.pt"
        +EPOCHS : int = 100
        +BATCH_SIZE : int = 32
        +IMAGE_SIZE : int = 640
        +CONFIDENCE_THRESHOLD : float = 0.5
        +CLASS_NAMES : List[str] = ["Fire","smoke"]
        +NUM_CLASSES : int = 2
        +ROBOFLOW_CONFIG : dict
        +get_dataset_paths() dict
        +get_label_paths() dict
    }

    class InferenceEngine {
        -model_path : str
        -model : YOLO
        -confidence : float
        -iou_threshold : float
        -class_names : List[str]
        -colors : dict
        -device : int|str
        +load_model(path?) bool
        +predict_image(path, save, show) dict
        +predict_batch(folder, out?) list
        +predict_video(path, out?, live?) dict
        +real_time_detection(camera, save, out) dict
        +set_confidence_threshold(c) void
        +set_iou_threshold(i) void
        +get_model_info() dict
        -_generate_colors() dict
        -_parse_detections(result) list
        -_print_detection_results(dets) void
        -_print_batch_summary(results) void
        -_visualize_result(result, path) void
    }

    class ModelTrainer {
        -model_size : str
        -model : YOLO
        -training_results : object
        -best_model_path : str
        +load_model(pretrained) bool
        +train(yaml, epochs, batch, ...) object
    }

    class ModelValidator {
        -model_path : str
        -model : YOLO
        -validation_results : object
        +load_model(path?) bool
        +validate(yaml, split, save_json, batch) dict
    }

    class DatasetManager {
        -base_path : str
        -dataset_paths : dict
        -label_paths : dict
        +create_directory_structure() void
        +check_dataset_status() dict
        +validate_dataset_format() bool
    }

    class FireSmokeDetectionPipeline {
        -config : Config
        -dataset_manager : DatasetManager
        -trainer : ModelTrainer
        -validator : ModelValidator
        -inference_engine : InferenceEngine
        -visualizer : VisualizationUtils
        +setup_environment() dict
        +prepare_dataset() dict
        +train_model() object
        +validate_model() dict
        +run_inference(...) dict
    }

    class VisualizationUtils {
        +plot_training_curves(...) void
        +plot_confusion_matrix(...) void
        +save_inference_grid(...) void
    }

    FireSmokeDetectionPipeline --> Config
    FireSmokeDetectionPipeline --> DatasetManager
    FireSmokeDetectionPipeline --> ModelTrainer
    FireSmokeDetectionPipeline --> ModelValidator
    FireSmokeDetectionPipeline --> InferenceEngine
    FireSmokeDetectionPipeline --> VisualizationUtils
    InferenceEngine ..> Config : reads CLASS_NAMES
    ModelTrainer ..> Config
    ModelValidator ..> Config
    DatasetManager ..> Config
```

---

## 3. 클래스 다이어그램 — Backend API Layer

```mermaid
classDiagram
    direction TB

    class Detection {
        +cls : str
        +confidence : float
        +bbox : List~float~
    }
    class ImageDetectionResult {
        +image_id : str
        +annotated_url : str
        +detections : List~Detection~
        +inference_ms : float
        +width : int
        +height : int
    }
    class JobCreated {
        +job_id : str
        +status : "queued"
    }
    class JobInfo {
        +job_id : str
        +status : JobStatus
        +progress : float
        +processed_frames : int
        +total_frames : int
        +detection_frames : int
        +detections_total : int
        +result_url : str?
        +error : str?
    }
    class ModelInfo {
        +weights_path : str
        +classes : List~str~
        +num_classes : int
        +device : str
        +map50 : float?
        +map50_95 : float?
        +precision : float?
        +recall : float?
    }
    class _JobState {
        +job_id : str
        +status : JobStatus
        +progress : float
        +processed_frames : int
        +total_frames : int
        +detection_frames : int
        +detections_total : int
        +result_name : str?
        +error : str?
    }

    class DetectRouter {
        <<FastAPI router>>
        +detect_image(file) ImageDetectionResult
        +detect_video(file) JobCreated
        -_infer_lock : asyncio.Lock
        -_save_upload(...) int
    }
    class JobsRouter {
        <<FastAPI router>>
        +get_job(job_id) JobInfo
        +get_job_result(job_id) FileResponse
    }
    class ModelRouter {
        <<FastAPI router>>
        +health() dict
        +model_info() ModelInfo
        +get_file(filename) FileResponse
    }
    class InferenceService {
        <<module>>
        -_engine : InferenceEngine?
        -_lock : threading.Lock
        +get_engine() InferenceEngine
        +reset_engine() void
    }
    class JobsService {
        <<module>>
        -_jobs : Dict~str,_JobState~
        -_lock : threading.Lock
        +enqueue_video_job(id, src, lock) void
        +get_job_info(id) JobInfo?
        +get_result_path(id) Path?
        +list_jobs() List~_JobState~
        -_process_video(id, src, lock) coroutine
        -_update(id, **fields) void
    }

    DetectRouter --> InferenceService : get_engine()
    DetectRouter --> JobsService : enqueue_video_job()
    DetectRouter ..> ImageDetectionResult : returns
    DetectRouter ..> JobCreated : returns
    JobsRouter --> JobsService : get_job_info / get_result_path
    JobsRouter ..> JobInfo : returns
    ModelRouter --> InferenceService : get_engine()
    ModelRouter ..> ModelInfo : returns
    JobsService --> InferenceService : get_engine()
    JobsService o-- _JobState : owns map
    InferenceService o-- InferenceEngine : singleton
    ImageDetectionResult o-- Detection
```

---

## 4. 클래스 다이어그램 — Frontend

```mermaid
classDiagram
    direction TB

    class App {
        <<component>>
        -tab : Tab
        +render()
    }
    class TabButton {
        <<component>>
        +active : bool
        +onClick : Function
        +children : ReactNode
    }
    class ImageDetect {
        <<page>>
        -busy : bool
        -result : ImageDetectionResult
        -error : string
        -filename : string
        +handle(file) async
    }
    class VideoDetect {
        <<page>>
        -busy : bool
        -job : JobInfo
        -error : string
        -filename : string
        -pollRef : timeoutId
        +handle(file) async
        +pollEffect() void
    }
    class DropZone {
        <<component>>
        +accept : string
        +label : string
        +disabled : bool
        +onFile : Function
        -over : bool
    }
    class ModelInfoCard {
        <<component>>
        -info : ModelInfo
        -error : string
    }
    class DetectionTable {
        <<component>>
        +detections : Detection[]
    }
    class JobProgress {
        <<component>>
        +job : JobInfo
    }

    class ApiClient {
        <<module>>
        +getModelInfo() Promise~ModelInfo~
        +detectImage(file) Promise~ImageDetectionResult~
        +startVideoJob(file) Promise~JobCreated~
        +getJob(id) Promise~JobInfo~
        +jobResultUrl(id) string
        -request~T~(url, init) Promise~T~
    }

    class Detection_TS {
        <<type>>
        +cls : string
        +confidence : number
        +bbox : [number,number,number,number]
    }
    class ImageDetectionResult_TS {
        <<type>>
    }
    class JobInfo_TS {
        <<type>>
    }
    class ModelInfo_TS {
        <<type>>
    }

    App --> TabButton
    App --> ModelInfoCard
    App --> ImageDetect
    App --> VideoDetect
    ImageDetect --> DropZone
    ImageDetect --> DetectionTable
    VideoDetect --> DropZone
    VideoDetect --> JobProgress
    ImageDetect ..> ApiClient : detectImage
    VideoDetect ..> ApiClient : startVideoJob, getJob
    ModelInfoCard ..> ApiClient : getModelInfo
    ApiClient ..> ImageDetectionResult_TS
    ApiClient ..> JobInfo_TS
    ApiClient ..> ModelInfo_TS
    ImageDetectionResult_TS o-- Detection_TS
```

---

## 5. 시퀀스 다이어그램 — 이미지 감지

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant UI as Browser (ImageDetect.tsx)
    participant Api as api/client.ts
    participant Det as routes/detect.py
    participant Inf as services/inference.py
    participant Eng as InferenceEngine
    participant FS as storage/

    User->>UI: 이미지 파일 드롭
    UI->>Api: detectImage(file)
    Api->>Det: POST /api/detect/image (multipart)
    Det->>Det: 확장자 / 용량 검증
    Det->>FS: uploads/{uuid}.{ext} 저장
    Det->>Inf: get_engine()
    Inf-->>Det: InferenceEngine (singleton)
    Det->>Det: async with _infer_lock
    Det->>Eng: to_thread(model.predict(source=path))
    Eng-->>Det: results[0]
    Det->>Eng: _parse_detections(result)
    Eng-->>Det: [{cls, confidence, bbox}]
    Det->>FS: results/{uuid}_annotated.jpg (result.plot())
    Det-->>Api: 200 ImageDetectionResult
    Api-->>UI: ImageDetectionResult
    UI->>UI: <img src=annotated_url/> + DetectionTable
    UI-->>User: 결과 표시
```

---

## 6. 시퀀스 다이어그램 — 비디오 감지 (비동기 잡 + 폴링)

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant UI as Browser (VideoDetect.tsx)
    participant Api as api/client.ts
    participant Det as routes/detect.py
    participant Jobs as services/jobs.py
    participant Eng as InferenceEngine
    participant FS as storage/

    User->>UI: 비디오 파일 드롭
    UI->>Api: startVideoJob(file)
    Api->>Det: POST /api/detect/video
    Det->>FS: uploads/{job_id}.{ext} 저장
    Det->>Jobs: enqueue_video_job(job_id, src, lock)
    Jobs->>Jobs: _jobs[id] = _JobState(queued)
    Jobs->>Jobs: loop.create_task(_process_video)
    Det-->>Api: 202 {job_id, status:"queued"}
    Api-->>UI: JobCreated

    Note over Jobs,Eng: 백그라운드 코루틴 시작

    Jobs->>Jobs: status="running", total_frames 추출
    Jobs->>Jobs: async with _infer_lock
    Jobs->>Eng: to_thread(_run) — stream=True

    loop 각 프레임
        Eng-->>Jobs: result (yield)
        Jobs->>FS: writer.write(result.plot())
        Jobs->>Jobs: counts 갱신
        alt processed % 5 == 0
            Jobs->>Jobs: _update(progress, processed_frames, ...)
        end
    end

    Jobs->>Jobs: status="done", result_name 설정

    par 폴링 루프
        loop 800ms 주기, 완료까지
            UI->>Api: getJob(job_id)
            Api->>Det: GET /api/jobs/{id}
            Det->>Jobs: get_job_info(id)
            Jobs-->>Det: JobInfo
            Det-->>Api: 200 JobInfo
            Api-->>UI: JobInfo
            UI->>UI: JobProgress 갱신
        end
    end

    UI->>UI: status=="done" 감지
    UI->>FS: <video src=result_url/>
    User->>UI: 다운로드 클릭
    UI->>Det: GET /api/jobs/{id}/result
    Det->>Jobs: get_result_path(id)
    Det-->>UI: FileResponse (video/mp4)
```

---

## 7. 시퀀스 다이어그램 — 모델 정보 조회

```mermaid
sequenceDiagram
    autonumber
    participant UI as ModelInfoCard
    participant Api as api/client.ts
    participant Mod as routes/model.py
    participant Inf as services/inference.py
    participant Eng as InferenceEngine
    participant FS as test_validation_report.json

    UI->>Api: getModelInfo()
    Api->>Mod: GET /api/model/info
    Mod->>Inf: get_engine()
    Inf-->>Mod: InferenceEngine
    Mod->>FS: read validation report
    FS-->>Mod: {map50, map50_95, precision, recall}
    Mod-->>Api: 200 ModelInfo
    Api-->>UI: ModelInfo
    UI->>UI: 카드 렌더
```

---

## 8. 상태 다이어그램 — 비디오 잡 라이프사이클

```mermaid
stateDiagram-v2
    [*] --> queued: enqueue_video_job()
    queued --> running: _process_video() 시작
    running --> running: _update(progress)
    running --> done: 모든 프레임 처리
    running --> failed: 예외 발생
    done --> [*]
    failed --> [*]

    note right of running
        async with _infer_lock
        매 5프레임마다 progress 갱신
    end note
    note right of done
        result_name = {job_id}_annotated.mp4
        result_url 응답에 포함
    end note
```

---

## 9. 상태 다이어그램 — Frontend 페이지 상태

### 9.1 ImageDetect

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> uploading: handle(file)
    uploading --> success: detectImage 성공
    uploading --> error: HTTPError
    success --> uploading: 새 파일 드롭
    error --> uploading: 새 파일 드롭
```

### 9.2 VideoDetect

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> uploading: handle(file)
    uploading --> polling: startVideoJob → getJob(first)
    uploading --> error: HTTPError
    polling --> polling: setTimeout(getJob, 800ms)
    polling --> done: status == "done"
    polling --> failed: status == "failed"
    done --> uploading: 새 파일 드롭
    failed --> uploading: 새 파일 드롭
    error --> uploading: 새 파일 드롭
```

---

## 10. 시퀀스 다이어그램 — 학습 파이프라인 (CLI)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 개발자
    participant CLI as backend/cli.py
    participant Cfg as config.py
    participant DM as DatasetManager
    participant Tr as ModelTrainer
    participant Vl as ModelValidator
    participant YOLO as Ultralytics YOLO
    participant FS as runs/detect/...

    Dev->>CLI: python cli.py --mode full
    CLI->>Cfg: initialize_project()
    Cfg-->>CLI: env / yaml ready
    CLI->>DM: create_directory_structure()
    CLI->>DM: check_dataset_status()
    DM-->>CLI: status
    CLI->>Tr: load_model(pretrained=True)
    Tr->>YOLO: YOLO('yolov8n.pt')
    CLI->>Tr: train(epochs=100, ...)
    Tr->>YOLO: model.train(data=data.yaml, ...)
    YOLO-->>FS: weights/best.pt 저장
    CLI->>Vl: load_model(best.pt)
    CLI->>Vl: validate()
    Vl->>YOLO: model.val(...)
    YOLO-->>Vl: metrics
    Vl-->>FS: validation_report.json
    Vl-->>CLI: metrics
```

---

## 11. 배포 다이어그램 (개발 환경)

```mermaid
flowchart TB
    subgraph DEV[개발자 머신 - Linux]
        subgraph NODE["Node.js + Vite Dev Server :5173"]
            REACT["React SPA<br/>HMR"]
        end

        subgraph PY["Python venv + Uvicorn :8000"]
            FAST["FastAPI app<br/>main.py"]
            ROUTES["routes/*"]
            SERVICES["services/*"]
            CORE["inference_engine + YOLO"]
        end

        subgraph GPU["NVIDIA GPU (CUDA) / CPU 폴백"]
            PYTORCH[PyTorch + CUDA]
        end

        subgraph DISK["로컬 디스크"]
            STORAGE["backend/storage/"]
            WEIGHTS["runs/detect/.../best.pt"]
            DATASETS["datasets/fire/"]
        end

        BROWSER["Browser localhost:5173"]
    end

    BROWSER -- HTTP --> REACT
    REACT -- "/api/* (vite proxy)" --> FAST
    FAST --> ROUTES
    ROUTES --> SERVICES
    SERVICES --> CORE
    CORE --> PYTORCH
    CORE --> WEIGHTS
    SERVICES --> STORAGE
    ROUTES --> STORAGE
    CORE --> DATASETS
```

---

## 12. API 시퀀스 요약 (엔드포인트 매트릭스)

| 메서드 | 경로 | 입력 | 출력 | 비고 |
| --- | --- | --- | --- | --- |
| GET | `/` | — | `{name, version}` | 헬스용 루트 |
| GET | `/api/health` | — | `{status:"ok"}` | 헬스체크 |
| GET | `/api/model/info` | — | `ModelInfo` | 메트릭은 `test_validation_report.json` 기반 |
| GET | `/api/files/{filename}` | path | binary | uploads/results 자동 탐색 |
| POST | `/api/detect/image` | multipart `file` | `ImageDetectionResult` | 20MB 한도, 동기 |
| POST | `/api/detect/video` | multipart `file` | `JobCreated` 202 | 200MB 한도, 비동기 잡 |
| GET | `/api/jobs/{job_id}` | path | `JobInfo` | 폴링용 |
| GET | `/api/jobs/{job_id}/result` | path | `FileResponse` mp4 | status=="done"만 |

---

## 13. 데이터 모델 (Pydantic ↔ TypeScript)

```mermaid
classDiagram
    direction LR

    class PydanticSchemas {
        <<schemas>>
        Detection
        ImageDetectionResult
        JobCreated
        JobStatus
        JobInfo
        ModelInfo
    }
    class TypeScriptTypes {
        <<types>>
        Detection
        ImageDetectionResult
        JobCreated
        JobStatus
        JobInfo
        ModelInfo
    }
    PydanticSchemas <--> TypeScriptTypes : mirrored 1 to 1
```

> 백엔드 `backend/schemas.py`와 프론트엔드 `frontend/src/types.ts`는 1:1로 미러링된다. 한쪽 변경 시 반드시 다른 쪽도 갱신할 것.
