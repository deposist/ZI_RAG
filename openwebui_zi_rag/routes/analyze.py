"""Retrieval and Deep RAG analysis routes."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..config import SidecarConfig
from ..indexing.service import RagService
from ..ollama_client import OllamaCancelled, OllamaError
from ..runtime import (
    analysis_jobs,
    analysis_jobs_lock,
    get_config,
    get_service,
    require_api_key,
)
from ..schemas import AnalyzeRequest, RetrieveRequest
from ..services.jobs import (
    ANALYSIS_JOB_GONE_DETAIL,
    analysis_job_event_stream,
    analysis_job_snapshot,
    cleanup_analysis_jobs,
    run_analysis_job,
)


router = APIRouter()


@router.post("/retrieve")
def retrieve(
    payload: RetrieveRequest,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    try:
        return service.retrieve(
            payload.query,
            index_ids=payload.index_ids or None,
            extra_index_ids=payload.extra_index_ids or None,
            embedding_model=payload.embedding_model or None,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/analyze/jobs")
def start_analyze_job(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    cleanup_analysis_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    job: dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "message": "Multi-pass анализ поставлен в очередь",
        "events": [],
        "result": None,
        "error": "",
        "cancel_event": threading.Event(),
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
    }
    with analysis_jobs_lock():
        analysis_jobs()[job_id] = job
    background_tasks.add_task(run_analysis_job, job_id, payload, cfg, service)
    return analysis_job_snapshot(job)


@router.get("/analyze/jobs/{job_id}")
def read_analyze_job(
    job_id: str,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    cleanup_analysis_jobs()
    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if not job:
            raise HTTPException(status_code=410, detail=ANALYSIS_JOB_GONE_DETAIL)
        return analysis_job_snapshot(job)


@router.get("/analyze/jobs/{job_id}/events")
def stream_analyze_job_events(
    job_id: str,
    _: None = Depends(require_api_key),
) -> StreamingResponse:
    cleanup_analysis_jobs()
    with analysis_jobs_lock():
        status_code = 410 if job_id not in analysis_jobs() else 200
    return StreamingResponse(
        analysis_job_event_stream(job_id),
        status_code=status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/analyze/jobs/{job_id}/cancel")
def cancel_analyze_job(
    job_id: str,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if not job:
            raise HTTPException(status_code=410, detail=ANALYSIS_JOB_GONE_DETAIL)
        cancel_event = job.get("cancel_event")
        if cancel_event is not None and hasattr(cancel_event, "set"):
            cancel_event.set()
        if str(job.get("status") or "") in {"queued", "running"}:
            job["status"] = "cancel_requested"
            job["message"] = "Cancel requested"
            job["updated_at"] = time.time()
        return analysis_job_snapshot(job)


@router.post("/analyze")
def analyze(
    payload: AnalyzeRequest,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    # Imported lazily through the server facade so monkeypatches like
    # ``monkeypatch.setattr(rag_server, "run_multi_pass_analysis", ...)`` keep
    # working after the route extraction.
    from .. import server as _server

    try:
        return _server.run_multi_pass_analysis(payload, cfg=cfg, service=service)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OllamaCancelled as exc:
        raise HTTPException(status_code=499, detail="Analysis canceled") from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


__all__ = ["router"]
