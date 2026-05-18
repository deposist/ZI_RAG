"""Admin/system routes: health, metrics, config, OCR cache, model listings, UI."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import SidecarConfig, update_config
from ..indexing.service import RagService
from ..runtime import (
    get_config,
    get_service,
    make_ollama_client,
    model_cache,
    model_to_dict,
    refresh_service,
    require_api_key,
)
from ..schemas import ConfigUpdate
from ..services.health import build_full_health_payload, build_public_health_payload


router = APIRouter()
_WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _cached_payload(key: str, ttl_sec: int, loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    import time
    cache = model_cache()
    now = time.monotonic()
    cached = cache.get(key)
    if cached and now - cached[0] < ttl_sec:
        return cached[1]
    payload = loader()
    cache[key] = (now, payload)
    return payload


async def _cached_payload_async(
    key: str,
    ttl_sec: int,
    loader: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    import time
    cache = model_cache()
    now = time.monotonic()
    cached = cache.get(key)
    if cached and now - cached[0] < ttl_sec:
        return cached[1]
    payload = await loader()
    cache[key] = (time.monotonic(), payload)
    return payload


def _load_openai_embedding_models(cfg: SidecarConfig) -> dict[str, Any]:
    return _load_openai_models(
        cfg.embedding_base_url or "http://127.0.0.1:5010/v1",
        api_key=cfg.embedding_api_key,
        timeout=cfg.request_timeout_sec,
    )


def _load_openai_models(base_url: str, *, api_key: str = "", timeout: float = 120) -> dict[str, Any]:
    url = (base_url or "http://127.0.0.1:5010/v1").rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    data = payload.get("data")
    if not isinstance(data, list):
        data = payload.get("models") or []
    return {
        "models": [
            {"name": item.get("id") or item.get("name") or item.get("model") or ""}
            for item in data
            if isinstance(item, dict)
        ]
    }


@router.get("/health", response_model=None)
def health(
    cfg: SidecarConfig = Depends(get_config),
    service: RagService = Depends(get_service),
) -> Any:
    status_code, payload = build_public_health_payload(cfg, service)
    if status_code >= 400:
        return JSONResponse(status_code=status_code, content=payload)
    return payload


@router.get("/health/full", response_model=None)
def health_full(
    _: None = Depends(require_api_key),
    cfg: SidecarConfig = Depends(get_config),
    service: RagService = Depends(get_service),
) -> Any:
    status_code, payload = build_full_health_payload(cfg, service)
    if status_code >= 400:
        return JSONResponse(status_code=status_code, content=payload)
    return payload


@router.get("/metrics")
def metrics(
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    return {"metrics": service.metrics_snapshot()}


@router.post("/ocr/cache/clear")
def clear_ocr_cache(
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    return {"ocr_gpu_cache": service.clear_ocr_gpu_cache(unload_readers=True)}


@router.get("/config")
def read_config(
    _: None = Depends(require_api_key),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    return cfg.model_dump()


@router.put("/config")
def write_config(payload: ConfigUpdate, _: None = Depends(require_api_key)) -> dict[str, Any]:
    updates = {key: value for key, value in model_to_dict(payload).items() if value is not None}
    try:
        cfg = update_config(updates)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    refresh_service(cfg)
    return cfg.model_dump()


@router.get("/ollama/models")
def ollama_models(
    _: None = Depends(require_api_key),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    def load() -> dict[str, Any]:
        provider = str(getattr(cfg, "deep_generation_provider", "ollama") or "ollama").lower()
        base_url = str(getattr(cfg, "deep_generation_base_url", "") or "")
        if provider in {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"} or (
            base_url and provider != "ollama"
        ):
            return _load_openai_models(
                base_url or "http://127.0.0.1:8081/v1",
                api_key=str(getattr(cfg, "deep_generation_api_key", "") or ""),
                timeout=cfg.request_timeout_sec,
            )
        client = make_ollama_client(cfg)
        return {"models": client.list_models()}

    return _cached_payload(
        f"generation:{cfg.deep_generation_provider}:{cfg.deep_generation_base_url}:{cfg.ollama_base_url}",
        20,
        load,
    )


@router.get("/embedding/models")
async def embedding_models(
    _: None = Depends(require_api_key),
    cfg: SidecarConfig = Depends(get_config),
) -> dict[str, Any]:
    provider = str(cfg.embedding_provider or "ollama").lower()
    cache_key = f"embedding:{provider}:{cfg.ollama_base_url}:{cfg.embedding_base_url}"

    async def load() -> dict[str, Any]:
        if provider not in {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}:
            return await asyncio.to_thread(lambda: {"models": make_ollama_client(cfg).list_models()})
        return await asyncio.to_thread(_load_openai_embedding_models, cfg)

    return await _cached_payload_async(cache_key, 20, load)


@router.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse((_WEB_DIR / "index.html").read_text(encoding="utf-8"))


__all__ = ["router"]
