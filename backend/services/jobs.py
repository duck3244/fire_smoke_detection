"""비디오 처리 잡 매니저 (in-memory, 단일 사용자 MVP)."""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from schemas import JobInfo, JobStatus
from services.inference import get_engine

logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
RESULTS_DIR = STORAGE_ROOT / "results"


@dataclass
class _JobState:
    job_id: str
    status: JobStatus = "queued"
    progress: float = 0.0
    processed_frames: int = 0
    total_frames: int = 0
    detection_frames: int = 0
    detections_total: int = 0
    result_name: str | None = None
    error: str | None = None


_jobs: dict[str, _JobState] = {}
_lock = threading.Lock()


def _get(job_id: str) -> _JobState | None:
    with _lock:
        return _jobs.get(job_id)


def _update(job_id: str, **fields) -> None:
    with _lock:
        st = _jobs.get(job_id)
        if st is None:
            return
        for k, v in fields.items():
            setattr(st, k, v)


def list_jobs() -> list[_JobState]:
    with _lock:
        return list(_jobs.values())


def get_job_info(job_id: str) -> JobInfo | None:
    st = _get(job_id)
    if st is None:
        return None
    return JobInfo(
        job_id=st.job_id,
        status=st.status,
        progress=round(st.progress, 4),
        processed_frames=st.processed_frames,
        total_frames=st.total_frames,
        detection_frames=st.detection_frames,
        detections_total=st.detections_total,
        result_url=f"/api/files/{st.result_name}" if st.result_name else None,
        error=st.error,
    )


def get_result_path(job_id: str) -> Path | None:
    st = _get(job_id)
    if st is None or st.result_name is None:
        return None
    return RESULTS_DIR / st.result_name


async def _process_video(job_id: str, src: Path, infer_lock: asyncio.Lock) -> None:
    """비디오 한 편을 stream 모드로 처리하면서 진행률을 갱신."""
    try:
        _update(job_id, status="running")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("VideoCapture open 실패")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        _update(job_id, total_frames=total)

        result_name = f"{job_id}_annotated.mp4"
        out_path = RESULTS_DIR / result_name
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        eng = get_engine()
        async with infer_lock:
            # ultralytics는 동기 → 스레드에서 stream 처리, 진행률은 thread-safe 콜백으로 갱신
            def _run():
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                processed = det_frames = det_total = 0
                try:
                    stream = eng.model.predict(
                        source=str(src),
                        conf=eng.confidence,
                        iou=eng.iou_threshold,
                        device=eng.device,
                        stream=True,
                        verbose=False,
                    )
                    for r in stream:
                        processed += 1
                        frame = r.plot()
                        writer.write(frame)
                        dets = eng._parse_detections(r)
                        if dets:
                            det_frames += 1
                            det_total += len(dets)
                        # 너무 자주 lock 잡지 않도록 5프레임마다 갱신
                        if processed % 5 == 0 or (total and processed == total):
                            _update(
                                job_id,
                                processed_frames=processed,
                                detection_frames=det_frames,
                                detections_total=det_total,
                                progress=(processed / total) if total else 0.0,
                            )
                finally:
                    writer.release()
                return processed, det_frames, det_total

            processed, det_frames, det_total = await asyncio.to_thread(_run)

        _update(
            job_id,
            status="done",
            progress=1.0,
            processed_frames=processed,
            detection_frames=det_frames,
            detections_total=det_total,
            result_name=result_name,
        )
        logger.info("job %s done: %d frames, %d detections", job_id, processed, det_total)
    except Exception as e:
        logger.exception("job %s failed", job_id)
        _update(job_id, status="failed", error=str(e))


def enqueue_video_job(job_id: str, src: Path, infer_lock: asyncio.Lock) -> None:
    """이벤트 루프에 비디오 처리 코루틴을 예약."""
    with _lock:
        _jobs[job_id] = _JobState(job_id=job_id)
    loop = asyncio.get_running_loop()
    loop.create_task(_process_video(job_id, src, infer_lock))
