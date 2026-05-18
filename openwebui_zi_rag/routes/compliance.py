"""Compliance analysis route."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..config import SidecarConfig
from ..indexing.service import RagService
from ..ollama_client import OllamaCancelled, OllamaError
from ..runtime import get_config, get_service, require_api_key
from ..schemas import ComplianceRequest


router = APIRouter()


@router.post("/compliance/analyze")
async def compliance_analyze(
    payload: str = Form(...),
    files: list[UploadFile] = File(...),
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    if not cfg.compliance_enabled:
        raise HTTPException(status_code=409, detail="Compliance Check is disabled")
    try:
        parsed_payload = json.loads(payload or "{}")
        request_payload = ComplianceRequest(**parsed_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid compliance payload: {exc}") from exc

    max_files = max(1, int(cfg.compliance_max_files or 10))
    if len(files) > max_files:
        raise HTTPException(status_code=413, detail=f"Too many files: max {max_files}")
    max_bytes = max(1, int(cfg.compliance_max_file_mb or 256)) * 1024 * 1024
    file_payloads: list[dict[str, Any]] = []
    for upload in files:
        content = await upload.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large: {upload.filename or 'upload'}",
            )
        file_payloads.append(
            {
                "filename": upload.filename or "upload",
                "content_type": upload.content_type or "",
                "content": content,
            }
        )

    # Late import through the server facade preserves monkeypatch compatibility
    # for tests that replace ``rag_server.run_compliance_analysis``.
    from .. import server as _server

    try:
        return await asyncio.to_thread(
            _server.run_compliance_analysis,
            request_payload,
            file_payloads,
            cfg=cfg,
            service=service,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OllamaCancelled as exc:
        raise HTTPException(status_code=499, detail="Analysis canceled") from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


__all__ = ["router"]
