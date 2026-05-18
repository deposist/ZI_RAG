"""Indexing job routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..indexing.registry import JobStatus
from ..indexing.service import RagService
from ..runtime import get_service, require_api_key


router = APIRouter()


@router.get("/jobs")
def list_jobs(
    index_id: str = "",
    active: bool = False,
    status: str = "",
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    statuses: list[Any] = []
    if active:
        statuses = [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED]
    elif status:
        statuses = [part.strip() for part in status.split(",") if part.strip()]
    jobs = service.registry.list_jobs(index_id=index_id, statuses=statuses, limit=100)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    job = service.registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        job["result"] = json.loads(job.get("result_json") or "{}")
    except Exception:
        job["result"] = {}
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    job = service.registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs = service.registry.request_cancel_jobs(job_id=job_id)
    return {"jobs": jobs}


@router.post("/indexes/{index_id}/jobs/cancel")
def cancel_index_jobs(
    index_id: str,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    jobs = service.registry.request_cancel_jobs(index_id=index_id)
    return {"jobs": jobs}


__all__ = ["router"]
