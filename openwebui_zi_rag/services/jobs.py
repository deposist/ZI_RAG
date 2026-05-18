"""Analysis job helpers (queueing, snapshots, SSE streaming)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from ..config import SidecarConfig
from ..indexing.service import RagService
from ..ollama_client import OllamaCancelled
from ..runtime import analysis_jobs, analysis_jobs_lock
from . import analysis as _analysis_service
from .analysis import AnalysisCancelled


ANALYSIS_JOB_TTL_SEC = 3600
ANALYSIS_TERMINAL_STATUSES = {"completed", "failed", "canceled"}
ANALYSIS_JOB_GONE_DETAIL = "Sidecar restarted, multi-pass job is gone. Retry from filter."


def cleanup_analysis_jobs() -> None:
    cutoff = time.time() - ANALYSIS_JOB_TTL_SEC
    jobs = analysis_jobs()
    with analysis_jobs_lock():
        stale = [
            job_id
            for job_id, job in jobs.items()
            if float(job.get("updated_at") or 0.0) < cutoff
            and str(job.get("status") or "") in ANALYSIS_TERMINAL_STATUSES
        ]
        for job_id in stale:
            jobs.pop(job_id, None)


def analysis_job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "message": job.get("message") or "",
        "events": list(job.get("events") or []),
        "result": job.get("result"),
        "error": job.get("error") or "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }


def sse_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def analysis_job_event_stream(job_id: str, *, poll_sec: float = 0.5) -> Iterator[str]:
    last_event_count = -1
    last_status = ""
    while True:
        with analysis_jobs_lock():
            job = analysis_jobs().get(job_id)
            if not job:
                yield sse_event("error", {"id": job_id, "error": ANALYSIS_JOB_GONE_DETAIL})
                return
            snapshot = analysis_job_snapshot(job)

        events = snapshot.get("events") or []
        status = str(snapshot.get("status") or "")
        event_count = len(events)
        changed = event_count != last_event_count or status != last_status
        if changed:
            yield sse_event("analysis", snapshot)
            last_event_count = event_count
            last_status = status
        if status in ANALYSIS_TERMINAL_STATUSES:
            yield sse_event("done", snapshot)
            return
        time.sleep(max(0.05, float(poll_sec)))


def analysis_cancel_requested(job_id: str) -> bool:
    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if not job:
            return True
        cancel_event = job.get("cancel_event")
        return bool(
            str(job.get("status") or "") in {"cancel_requested", "canceled"}
            or (cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set())
        )


def append_analysis_event(job_id: str, event: dict[str, Any]) -> None:
    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if not job:
            return
        events = list(job.get("events") or [])
        events.append(dict(event))
        if len(events) > 300:
            events = events[-300:]
        job["events"] = events
        job["message"] = str(event.get("message") or job.get("message") or "")
        job["updated_at"] = time.time()


def run_analysis_job(
    job_id: str,
    payload: Any,
    cfg: SidecarConfig,
    service: RagService,
) -> None:
    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if job:
            cancel_event = job.get("cancel_event")
            if str(job.get("status") or "") == "cancel_requested" or (
                cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set()
            ):
                job["status"] = "canceled"
                job["message"] = "Multi-pass анализ остановлен"
                job["error"] = "Multi-pass analysis canceled"
                job["updated_at"] = time.time()
                job["finished_at"] = time.time()
                return
            job["status"] = "running"
            job["message"] = "Multi-pass анализ запущен"
            job["updated_at"] = time.time()
    try:
        result = _analysis_service.run_multi_pass_analysis(
            payload,
            cfg=cfg,
            service=service,
            progress=lambda event: append_analysis_event(job_id, event),
            cancel_check=lambda: analysis_cancel_requested(job_id),
        )
    except (AnalysisCancelled, OllamaCancelled) as exc:
        append_analysis_event(
            job_id,
            {
                "stage": "canceled",
                "message": "Multi-pass анализ остановлен пользователем",
                "done": True,
            },
        )
        with analysis_jobs_lock():
            job = analysis_jobs().get(job_id)
            if job:
                job["status"] = "canceled"
                job["message"] = "Multi-pass анализ остановлен"
                job["error"] = str(exc)
                job["updated_at"] = time.time()
                job["finished_at"] = time.time()
        return
    except Exception as exc:
        with analysis_jobs_lock():
            job = analysis_jobs().get(job_id)
            if job:
                job["status"] = "failed"
                job["message"] = "Multi-pass анализ завершился ошибкой"
                job["error"] = str(exc)
                job["updated_at"] = time.time()
                job["finished_at"] = time.time()
        return

    with analysis_jobs_lock():
        job = analysis_jobs().get(job_id)
        if job:
            cancel_event = job.get("cancel_event")
            if str(job.get("status") or "") == "cancel_requested" or (
                cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set()
            ):
                job["status"] = "canceled"
                job["message"] = "Multi-pass анализ остановлен"
                job["error"] = "Multi-pass analysis canceled"
                job["result"] = None
            else:
                job["status"] = "completed"
                job["message"] = "Multi-pass анализ готов"
                job["result"] = result
            job["updated_at"] = time.time()
            job["finished_at"] = time.time()


__all__ = [
    "ANALYSIS_JOB_TTL_SEC",
    "ANALYSIS_TERMINAL_STATUSES",
    "ANALYSIS_JOB_GONE_DETAIL",
    "cleanup_analysis_jobs",
    "analysis_job_snapshot",
    "sse_event",
    "analysis_job_event_stream",
    "analysis_cancel_requested",
    "append_analysis_event",
    "run_analysis_job",
]
