"""Application factory for the ZI_RAG sidecar.

The factory wires runtime state, route packages and OpenAPI metadata onto a
single ``FastAPI`` instance. ``server`` keeps the legacy module-level ``app``
attribute by calling :func:`create_app` once on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in bare envs
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment, misc]
    StaticFiles = None  # type: ignore[assignment, misc]

from .config import SidecarConfig
from .indexing.service import RagService
from .openapi_meta import apply_openapi_route_metadata
from .runtime import register_app

from . import __version__ as ZI_RAG_VERSION


_WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app(
    *,
    config: SidecarConfig | None = None,
    service: RagService | None = None,
) -> Any:
    """Create the FastAPI app and register routers and runtime state."""
    if FastAPI is None:  # pragma: no cover
        return None

    app = FastAPI(title="OpenWebUI Enhanced RAG Sidecar", version=ZI_RAG_VERSION)
    register_app(app)

    if config is not None or service is not None:
        from .runtime import configure_runtime_state

        configure_runtime_state(config=config, service=service)

    if StaticFiles is not None and _WEB_DIR.exists():
        app.mount("/ui/assets", StaticFiles(directory=str(_WEB_DIR)), name="ui-assets")

    from .routes import (
        admin_router,
        analyze_router,
        chat_attachments_router,
        compliance_router,
        documents_router,
        indexes_router,
        jobs_router,
    )

    for router in (
        admin_router,
        indexes_router,
        documents_router,
        chat_attachments_router,
        analyze_router,
        compliance_router,
        jobs_router,
    ):
        app.include_router(router)

    apply_openapi_route_metadata(app)
    return app


__all__ = ["create_app"]
