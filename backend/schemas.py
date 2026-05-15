"""Pydantic 스키마 — API 응답 타입 정의."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    cls: str = Field(..., description="클래스 이름 (Fire | smoke)")
    confidence: float
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] (픽셀)")


class ImageDetectionResult(BaseModel):
    image_id: str
    annotated_url: str
    detections: list[Detection]
    inference_ms: float
    width: int
    height: int


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


JobStatus = Literal["queued", "running", "done", "failed"]


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0  # 0.0 ~ 1.0
    processed_frames: int = 0
    total_frames: int = 0
    detection_frames: int = 0
    detections_total: int = 0
    result_url: str | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    weights_path: str
    classes: list[str]
    num_classes: int
    device: str
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
