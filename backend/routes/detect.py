"""감지(추론) 라우트 — 이미지/비디오 업로드."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from schemas import Detection, ImageDetectionResult, JobCreated
from services.inference import get_engine
from services.jobs import enqueue_video_job

router = APIRouter()
logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
UPLOAD_DIR = STORAGE_ROOT / "uploads"
RESULTS_DIR = STORAGE_ROOT / "results"

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB

# 단일 사용자 MVP: GPU 점유 직렬화
_infer_lock = asyncio.Lock()


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


async def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    """업로드를 청크 단위로 저장하고 바이트 크기 반환. 한도 초과 시 HTTP 413."""
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"file too large (> {max_bytes} bytes)")
            f.write(chunk)
    return written


@router.post("/api/detect/image", response_model=ImageDetectionResult)
async def detect_image(file: UploadFile = File(...)):
    ext = _ext(file.filename or "")
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported image type: {ext}")

    image_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{image_id}{ext}"
    await _save_upload(file, upload_path, MAX_IMAGE_BYTES)

    eng = get_engine()
    async with _infer_lock:
        t0 = time.perf_counter()
        # ultralytics는 동기 호출 → 별도 스레드에서 실행
        results = await asyncio.to_thread(
            eng.model.predict,
            source=str(upload_path),
            conf=eng.confidence,
            iou=eng.iou_threshold,
            device=eng.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = results[0]
    detections_raw = eng._parse_detections(result)
    detections = [
        Detection(cls=d["class"], confidence=d["confidence"], bbox=d["bbox"])
        for d in detections_raw
    ]

    # 어노테이트 이미지 저장
    annotated = result.plot()  # BGR ndarray
    annotated_name = f"{image_id}_annotated.jpg"
    annotated_path = RESULTS_DIR / annotated_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), annotated)

    h, w = annotated.shape[:2]
    return ImageDetectionResult(
        image_id=image_id,
        annotated_url=f"/api/files/{annotated_name}",
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        width=int(w),
        height=int(h),
    )


@router.post("/api/detect/video", response_model=JobCreated, status_code=202)
async def detect_video(file: UploadFile = File(...)):
    ext = _ext(file.filename or "")
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported video type: {ext}")

    job_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    await _save_upload(file, upload_path, MAX_VIDEO_BYTES)

    enqueue_video_job(job_id, upload_path, _infer_lock)
    return JobCreated(job_id=job_id)
