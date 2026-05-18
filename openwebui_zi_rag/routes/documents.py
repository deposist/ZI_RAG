"""Document upload, listing, deletion, reindexing routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from ..config import SidecarConfig
from ..indexing.service import RagService
from ..runtime import (
    get_config,
    get_service,
    public_path_error,
    require_api_key,
)
from ..schemas import (
    AddPathRequest,
    DeleteDocumentsRequest,
    ReindexDocumentsRequest,
)


router = APIRouter()


def _job_document_ids(job: dict[str, Any]) -> set[str]:
    document_id = str(job.get("document_id") or "").strip()
    if document_id:
        return {document_id}
    try:
        payload = json.loads(job.get("result_json") or "{}")
    except Exception:
        return set()
    return {str(item).strip() for item in payload.get("document_ids", []) if str(item).strip()}


@router.get("/indexes/{index_id}/documents")
def list_documents(
    index_id: str,
    limit: int = 200,
    offset: int = 0,
    query: str = "",
    status: str = "",
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    return service.registry.list_documents_page(
        index_id,
        limit=limit,
        offset=offset,
        query=query,
        status=status,
    )


@router.post("/indexes/{index_id}/documents/upload")
async def upload_document(
    index_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    content = await file.read()
    max_bytes = int(cfg.max_upload_mb) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Upload is too large")
    document = service.save_upload(index_id, file.filename or "upload", content, file.content_type or "")
    job = service.registry.create_job("index_document", index_id=index_id, document_id=document["id"])
    background_tasks.add_task(service.run_job, job["id"], service.index_document_now, document["id"])
    return {"document": document, "job": job}


@router.post("/indexes/{index_id}/documents/upload-batch")
async def upload_documents_batch(
    index_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    if not files:
        raise HTTPException(status_code=400, detail="No files selected")
    max_bytes = int(cfg.max_upload_mb) * 1024 * 1024
    documents: list[dict[str, Any]] = []
    for file in files:
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload is too large: {file.filename}")
        documents.append(
            service.save_upload(
                index_id,
                file.filename or "upload",
                content,
                file.content_type or "",
            )
        )
    document_ids = [document["id"] for document in documents]
    if len(document_ids) == 1:
        job = service.registry.create_job("index_document", index_id=index_id, document_id=document_ids[0])
        background_tasks.add_task(service.run_job, job["id"], service.index_document_now, document_ids[0])
    else:
        job = service.registry.create_job(
            "index_documents",
            index_id=index_id,
            result_json=json.dumps(
                {"document_ids": document_ids, "documents": len(document_ids)},
                ensure_ascii=False,
            ),
        )
        background_tasks.add_task(
            service.run_job, job["id"], service.index_documents_now, index_id, document_ids
        )
    return {"documents": documents, "jobs": [job]}


@router.post("/indexes/{index_id}/documents/add-path")
def add_path(
    index_id: str,
    payload: AddPathRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    try:
        documents = service.add_path(
            index_id,
            payload.path,
            recursive=payload.recursive,
            include=payload.include,
            exclude=payload.exclude,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=public_path_error(exc, default="Permission denied"),
        ) from exc
    jobs: list[dict[str, Any]] = []
    if payload.index_now:
        document_ids = [document["id"] for document in documents]
        if len(document_ids) == 1:
            document = documents[0]
            job = service.registry.create_job(
                "index_document", index_id=index_id, document_id=document["id"]
            )
            background_tasks.add_task(service.run_job, job["id"], service.index_document_now, document["id"])
            jobs.append(job)
        elif document_ids:
            job = service.registry.create_job(
                "index_documents",
                index_id=index_id,
                result_json=json.dumps(
                    {"document_ids": document_ids, "documents": len(document_ids)},
                    ensure_ascii=False,
                ),
            )
            background_tasks.add_task(
                service.run_job, job["id"], service.index_documents_now, index_id, document_ids
            )
            jobs.append(job)
    return {"documents": documents, "jobs": jobs}


@router.delete("/indexes/{index_id}/documents/{document_id}")
def delete_document(
    index_id: str,
    document_id: str,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    document = service.registry.get_document(document_id)
    if not document or document["index_id"] != index_id:
        raise HTTPException(status_code=404, detail="Document not found")
    deleted = service.delete_document(document_id)
    service.remove_storage_file(deleted)
    return {"deleted": deleted}


@router.post("/indexes/{index_id}/documents/delete")
def delete_documents(
    index_id: str,
    payload: DeleteDocumentsRequest,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")

    document_ids = [str(item).strip() for item in payload.document_ids if str(item).strip()]
    document_ids = list(dict.fromkeys(document_ids))
    if not document_ids:
        raise HTTPException(status_code=400, detail="No documents selected")
    if len(document_ids) > 500:
        raise HTTPException(status_code=413, detail="Too many documents selected: max 500")

    documents: list[dict[str, Any]] = []
    missing: list[str] = []
    for document_id in document_ids:
        document = service.registry.get_document(document_id)
        if not document or document["index_id"] != index_id or document.get("deleted_at"):
            missing.append(document_id)
            continue
        documents.append(document)
    if not documents:
        raise HTTPException(status_code=404, detail="Selected documents not found")

    cancelled_by_id: dict[str, dict[str, Any]] = {}
    active_jobs = service.registry.active_jobs_for_documents(
        index_id,
        [document["id"] for document in documents],
    )
    for job in active_jobs:
        for cancelled_job in service.registry.request_cancel_jobs(job_id=str(job["id"])):
            cancelled_by_id[str(cancelled_job["id"])] = cancelled_job

    try:
        result = service.delete_documents(index_id, [document["id"] for document in documents])
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=public_path_error(exc, default="File not found")
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for deleted in result.get("deleted", []):
        service.remove_storage_file(deleted)

    return {
        **result,
        "missing": missing,
        "cancelled": list(cancelled_by_id.values()),
    }


@router.post("/indexes/{index_id}/documents/{document_id}/reindex")
def reindex_document(
    index_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    document = service.registry.get_document(document_id)
    if not document or document["index_id"] != index_id:
        raise HTTPException(status_code=404, detail="Document not found")
    job = service.registry.create_job("reindex_document", index_id=index_id, document_id=document_id)
    background_tasks.add_task(service.run_job, job["id"], service.index_document_now, document_id)
    return {"job": job}


@router.post("/indexes/{index_id}/documents/reindex")
def reindex_documents(
    index_id: str,
    payload: ReindexDocumentsRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")

    document_ids = [str(item).strip() for item in payload.document_ids if str(item).strip()]
    document_ids = list(dict.fromkeys(document_ids))
    if not document_ids:
        raise HTTPException(status_code=400, detail="No documents selected")
    if len(document_ids) > 500:
        raise HTTPException(status_code=413, detail="Too many documents selected: max 500")

    documents: list[dict[str, Any]] = []
    missing: list[str] = []
    for document_id in document_ids:
        document = service.registry.get_document(document_id)
        if not document or document["index_id"] != index_id or document.get("deleted_at"):
            missing.append(document_id)
            continue
        documents.append(document)

    active_jobs = service.registry.active_jobs_for_documents(
        index_id,
        [document["id"] for document in documents],
    )
    active_document_ids: set[str] = set()
    active_bulk_job_ids: set[str] = set()
    for job in active_jobs:
        job_ids = _job_document_ids(job)
        overlap = set(document["id"] for document in documents).intersection(job_ids)
        active_document_ids.update(overlap)
        if overlap and not str(job.get("document_id") or "").strip():
            active_bulk_job_ids.add(str(job["id"]))
    skipped: list[dict[str, Any]] = []
    cancelled: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    selected_document_ids: list[str] = []

    for document in documents:
        document_id = document["id"]
        if document_id in active_document_ids and not payload.force:
            skipped.append(
                {
                    "document_id": document_id,
                    "filename": document.get("filename") or document_id,
                    "reason": "active_job_exists",
                }
            )
            continue
        if payload.force and document_id in active_document_ids:
            cancelled.extend(
                service.registry.request_cancel_jobs(
                    index_id=index_id,
                    document_id=document_id,
                )
            )
        selected_document_ids.append(document_id)

    if payload.force:
        for job_id in sorted(active_bulk_job_ids):
            cancelled.extend(service.registry.request_cancel_jobs(job_id=job_id))

    if len(selected_document_ids) == 1:
        document_id = selected_document_ids[0]
        kind = "force_reindex_document" if payload.force else "reindex_document"
        job = service.registry.create_job(kind, index_id=index_id, document_id=document_id)
        background_tasks.add_task(service.run_job, job["id"], service.index_document_now, document_id)
        jobs.append(job)
    elif selected_document_ids:
        kind = "force_reindex_documents" if payload.force else "reindex_documents"
        job = service.registry.create_job(
            kind,
            index_id=index_id,
            result_json=json.dumps(
                {
                    "document_ids": selected_document_ids,
                    "documents": len(selected_document_ids),
                    "force": bool(payload.force),
                },
                ensure_ascii=False,
            ),
        )
        background_tasks.add_task(
            service.run_job,
            job["id"],
            service.index_documents_now,
            index_id,
            selected_document_ids,
        )
        jobs.append(job)

    return {
        "jobs": jobs,
        "document_count": len(selected_document_ids),
        "skipped": skipped,
        "cancelled": cancelled,
        "missing": missing,
        "force": bool(payload.force),
    }


__all__ = ["router"]
