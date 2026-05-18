"""Index management routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..config import save_config
from ..indexing.service import RagService
from ..runtime import (
    get_config,
    get_service,
    model_to_dict,
    refresh_service,
    require_api_key,
)
from ..schemas import IndexCreate, RebuildIndexRequest


router = APIRouter()


@router.get("/indexes")
def list_indexes(
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    return {"indexes": service.registry.list_indexes()}


@router.post("/indexes")
def create_index(
    payload: IndexCreate,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    return service.create_index(model_to_dict(payload))


@router.delete("/indexes/{index_id}")
def delete_index(
    index_id: str,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    try:
        deleted = service.delete_index(index_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cfg = get_config()
    if index_id in cfg.default_index_ids:
        cfg.default_index_ids = [item for item in cfg.default_index_ids if item != index_id]
        save_config(cfg)
        refresh_service(cfg)
    return {"deleted": deleted}


@router.post("/indexes/{index_id}/rebuild")
def rebuild_index(
    index_id: str,
    payload: RebuildIndexRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not service.registry.get_index(index_id):
        raise HTTPException(status_code=404, detail="Index not found")
    document_ids = [str(item).strip() for item in payload.document_ids if str(item).strip()]
    document_ids = list(dict.fromkeys(document_ids))
    job = service.registry.create_job(
        "rebuild_index",
        index_id=index_id,
        result_json=json.dumps(
            {"document_ids": document_ids, "documents": len(document_ids)},
            ensure_ascii=False,
        ),
    )
    background_tasks.add_task(
        service.run_job,
        job["id"],
        service.rebuild_index_documents_now,
        index_id,
        document_ids or None,
    )
    return {"job": job, "document_count": len(document_ids)}


__all__ = ["router"]
