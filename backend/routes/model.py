"""모델 정보 + 정적 파일 라우트."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import Config
from schemas import ModelInfo
from services.inference import get_engine

router = APIRouter()


@router.get("/api/health")
def health():
    return {"status": "ok"}


@router.get("/api/model/info", response_model=ModelInfo)
def model_info():
    eng = get_engine()
    weights = eng.model_path

    # validation_report.json 이 있으면 메트릭 동봉
    metrics = {}
    report_path = Path(__file__).resolve().parent.parent / "test_validation_report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            m = data.get("metrics", {})
            metrics = {
                "map50": m.get("map50"),
                "map50_95": m.get("map50_95"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
            }
        except Exception:
            pass

    return ModelInfo(
        weights_path=str(weights),
        classes=Config.CLASS_NAMES,
        num_classes=Config.NUM_CLASSES,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **metrics,
    )


@router.get("/api/files/{filename}")
def get_file(filename: str):
    """storage/results 또는 storage/uploads 의 파일 서빙."""
    storage = Path(__file__).resolve().parent.parent / "storage"
    for sub in ("results", "uploads"):
        candidate = storage / sub / filename
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="file not found")
