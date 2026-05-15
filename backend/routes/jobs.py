"""잡 상태 폴링 + 결과 다운로드."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from schemas import JobInfo
from services.jobs import get_job_info, get_result_path

router = APIRouter()


@router.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    info = get_job_info(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="job not found")
    return info


@router.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    info = get_job_info(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="job not found")
    if info.status != "done":
        raise HTTPException(status_code=409, detail=f"job not done (status={info.status})")
    p = get_result_path(job_id)
    if p is None or not p.exists():
        raise HTTPException(status_code=410, detail="result file missing")
    return FileResponse(p, media_type="video/mp4", filename=p.name)
