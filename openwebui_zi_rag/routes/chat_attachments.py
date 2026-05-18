"""Chat attachment indexing route."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..config import SidecarConfig
from ..indexing.service import IndexingDeadlineExceeded, RagService
from ..ollama_client import OllamaError
from ..runtime import get_config, get_service, model_to_dict, require_api_key
from ..schemas import ChatAttachmentsRequest


router = APIRouter()


@router.post("/chat-attachments/index")
async def index_chat_attachments(
    payload: str = Form(...),
    files: list[UploadFile] = File(...),
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    if not cfg.chat_attachments_enabled:
        raise HTTPException(status_code=409, detail="Chat attachment indexing is disabled")
    try:
        parsed_payload = json.loads(payload or "{}")
        request_payload = ChatAttachmentsRequest(**parsed_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid chat attachment payload: {exc}") from exc

    max_files = max(1, int(cfg.chat_attachment_max_files or 10))
    if len(files) > max_files:
        raise HTTPException(status_code=413, detail=f"Too many files: max {max_files}")
    max_bytes = max(1, int(cfg.chat_attachment_max_file_mb or 256)) * 1024 * 1024
    metas = request_payload.files or []
    file_payloads: list[dict[str, Any]] = []
    for pos, upload in enumerate(files):
        content = await upload.read()
        filename = upload.filename or f"attachment-{pos + 1}"
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File is too large: {filename}")
        meta = model_to_dict(metas[pos]) if pos < len(metas) else {}
        file_payloads.append(
            {
                "filename": meta.get("name") or filename,
                "content_type": meta.get("content_type") or upload.content_type or "",
                "content": content,
                "external_id": meta.get("id") or "",
                "metadata": meta,
            }
        )

    scope_id = (
        request_payload.scope_id
        or request_payload.chat_id
        or request_payload.session_id
        or request_payload.message_id
    ).strip()
    if not scope_id:
        raise HTTPException(
            status_code=400,
            detail="chat_id, session_id, message_id or scope_id is required",
        )
    deadline = time.monotonic() + max(1, int(cfg.chat_attachment_timeout_sec or 900))
    try:
        result = await asyncio.to_thread(
            service.index_chat_attachments,
            scope_id,
            file_payloads,
            chat_id=request_payload.chat_id,
            user_id=request_payload.user_id,
            message_id=request_payload.message_id,
            deadline=deadline,
        )
    except IndexingDeadlineExceeded as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


__all__ = ["router"]
