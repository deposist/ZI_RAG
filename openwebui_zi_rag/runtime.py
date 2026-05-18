"""Runtime state, dependency-injection helpers and shared utilities.

Pulled out of ``server.py`` so route packages can share runtime state without
importing the FastAPI app object directly.
"""

from __future__ import annotations

import hmac
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

try:  # pragma: no cover - degrade gracefully without FastAPI
    from fastapi import HTTPException, Header, Request
except Exception:  # pragma: no cover
    HTTPException = None  # type: ignore[assignment, misc]
    Header = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment, misc]

from .config import SidecarConfig, load_config
from .indexing.service import RagService
from .indexing.vector_store import clear_index_cache
from .ollama_client import OllamaClient, make_generation_client


_FALLBACK_STATE = SimpleNamespace()
_APP_REF: list[Any] = []
_GET_CONFIG_OVERRIDE: list[Callable[[], SidecarConfig]] = []


def set_get_config_provider(provider: Callable[[], SidecarConfig] | None) -> None:
    """Install a callable that overrides :func:`get_config`.

    Used by ``server`` so that ``monkeypatch.setattr(rag_server, "get_config",
    ...)`` propagates to dependencies that look up the config (notably
    :func:`require_api_key`).
    """
    _GET_CONFIG_OVERRIDE.clear()
    if provider is not None:
        _GET_CONFIG_OVERRIDE.append(provider)


def _resolve_config() -> "SidecarConfig":
    if _GET_CONFIG_OVERRIDE:
        return _GET_CONFIG_OVERRIDE[0]()
    return get_config()


def register_app(app: Any) -> None:
    """Remember the FastAPI app instance so :func:`_state_container` can locate it."""
    _APP_REF.clear()
    if app is not None:
        _APP_REF.append(app)


def _state_container() -> Any:
    if _APP_REF:
        return getattr(_APP_REF[0], "state", _FALLBACK_STATE)
    return _FALLBACK_STATE


def _install_runtime_state(
    container: Any,
    *,
    config: SidecarConfig | None = None,
    service: RagService | None = None,
) -> None:
    cfg = config or (service.config if service is not None else load_config())
    container.zi_rag_config = cfg
    container.zi_rag_service = service or RagService(cfg)
    container.zi_rag_model_cache = {}
    # Analysis jobs are in-memory only; installing runtime state is the explicit
    # restart boundary where stale multi-pass jobs become gone (HTTP 410).
    container.zi_rag_analysis_jobs = {}
    container.zi_rag_analysis_jobs_lock = threading.Lock()
    container.zi_rag_initialized = True


def _runtime_state() -> Any:
    container = _state_container()
    if not getattr(container, "zi_rag_initialized", False):
        _install_runtime_state(container)
    return container


def configure_runtime_state(
    *,
    config: SidecarConfig | None = None,
    service: RagService | None = None,
) -> None:
    """Reset runtime state with the supplied config/service.

    Used by tests to inject a fresh ``RagService`` without restarting the app.
    """
    _install_runtime_state(_state_container(), config=config, service=service)


def model_cache() -> dict[str, tuple[float, dict[str, Any]]]:
    return cast(dict[str, tuple[float, dict[str, Any]]], _runtime_state().zi_rag_model_cache)


def analysis_jobs() -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], _runtime_state().zi_rag_analysis_jobs)


def analysis_jobs_lock() -> Any:
    return _runtime_state().zi_rag_analysis_jobs_lock


def get_config() -> SidecarConfig:
    return cast(SidecarConfig, _runtime_state().zi_rag_config)


def get_service() -> RagService:
    return cast(RagService, _runtime_state().zi_rag_service)


def refresh_service(new_config: SidecarConfig) -> None:
    state = _runtime_state()
    old_service = getattr(state, "zi_rag_service", None)
    if old_service is not None:
        try:
            old_service.registry.close()
        except Exception:
            pass
    state.zi_rag_config = new_config
    state.zi_rag_service = RagService(new_config)
    state.zi_rag_model_cache.clear()
    clear_index_cache()


def make_ollama_client(
    cfg: SidecarConfig,
    *,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    stream_idle_timeout: float | None = None,
) -> OllamaClient:
    """Build an :class:`OllamaClient` honouring timeout fields on the config.

    The class is resolved through ``openwebui_zi_rag.server`` when that facade is
    already imported. Tests intentionally monkeypatch ``rag_server.OllamaClient``,
    so this small indirection keeps route-level behavior compatible while the
    runtime state lives outside ``server.py``.
    """
    resolved_request_timeout = float(
        request_timeout
        if request_timeout is not None
        else getattr(cfg, "request_timeout_sec", 120) or 120
    )
    # Keep this late lookup aligned with server.py's public monkeypatch surface.
    client_cls = OllamaClient
    try:
        import sys

        server_module = sys.modules.get("openwebui_zi_rag.server")
        if server_module is not None:
            client_cls = getattr(server_module, "OllamaClient", OllamaClient)
    except Exception:
        client_cls = OllamaClient
    return cast(
        OllamaClient,
        make_generation_client(
            cfg,
            request_timeout=resolved_request_timeout,
            connect_timeout=connect_timeout,
            stream_idle_timeout=stream_idle_timeout,
            ollama_client_cls=client_cls,
        ),
    )


def model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return cast(dict[str, Any], model.model_dump())
    return cast(dict[str, Any], model.dict())


def public_path_error(exc: Exception, *, default: str) -> str:
    """Return a message safe to expose in HTTP responses without leaking paths."""
    message = str(exc or "").strip()
    if isinstance(exc, PermissionError):
        if "outside allowed_source_roots" in message:
            return "Path is outside allowed_source_roots"
        if "allowed_source_roots is empty" in message:
            return "allowed_source_roots is empty"
        return default
    if isinstance(exc, FileNotFoundError):
        if not message:
            return default
        if "/" in message or "\\" in message:
            return default
        return Path(message).name
    return default


def require_api_key(
    request: "Request",
    x_api_key: str | None = Header(default=None) if Header is not None else None,
    authorization: str | None = Header(default=None) if Header is not None else None,
) -> None:
    """Standard FastAPI dependency: enforce the API key policy from config."""
    if HTTPException is None:  # pragma: no cover
        return
    cfg = _resolve_config()
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    supplied = x_api_key or bearer
    if cfg.api_key:
        if not supplied or not hmac.compare_digest(str(supplied), str(cfg.api_key)):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return

    client_host = (request.client.host if request.client else "") or ""
    if not cfg.require_api_key_localhost and (
        client_host in {"127.0.0.1", "::1", "localhost"} or client_host.startswith("127.")
    ):
        return
    raise HTTPException(
        status_code=401,
        detail="ZI_RAG api_key is required for non-local sidecar access",
    )


__all__ = [
    "register_app",
    "configure_runtime_state",
    "model_cache",
    "analysis_jobs",
    "analysis_jobs_lock",
    "get_config",
    "get_service",
    "refresh_service",
    "make_ollama_client",
    "model_to_dict",
    "public_path_error",
    "require_api_key",
    "set_get_config_provider",
]
