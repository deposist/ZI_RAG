"""Top-level entry point for the ZI_RAG sidecar.

Historically this module hosted the FastAPI app, schemas, runtime state, route
handlers and helpers. Decomposition (REVIEW_PLAN.md item 2.1) extracted those
pieces into focused modules:

* schemas live in :mod:`openwebui_zi_rag.schemas`
* runtime state and dependencies in :mod:`openwebui_zi_rag.runtime`
* routes in :mod:`openwebui_zi_rag.routes.*`
* health checks in :mod:`openwebui_zi_rag.services.health`
* analysis-job orchestration in :mod:`openwebui_zi_rag.services.jobs`

This module re-exports the public surface used by the existing tests and by the
``python -m openwebui_zi_rag`` entry point so the public API remains stable.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import uvicorn
    from fastapi import HTTPException
except Exception as exc:  # pragma: no cover - import-time guidance for bare Python envs
    uvicorn = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment, misc]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None

from . import app as _app_module
from . import runtime as _runtime
from .config import SidecarConfig, load_config, save_config, update_config
from .indexing import vector_store
from .indexing.extraction import extract_text, is_supported_file
from .indexing.registry import JobStatus, Registry
from .indexing.service import RagService
from .ollama_client import OllamaCancelled, OllamaClient, OllamaError
from .openapi_meta import OPENAPI_ROUTE_METADATA, apply_openapi_route_metadata
from .schemas import (
    AddPathRequest,
    AnalyzeRequest,
    ChatAttachmentFileMeta,
    ChatAttachmentsRequest,
    ComplianceRequest,
    ConfigUpdate,
    DeleteDocumentsRequest,
    IndexCreate,
    RebuildIndexRequest,
    ReindexDocumentsRequest,
    RetrieveRequest,
)
from .services import analysis as _analysis_service
from .services.analysis import (  # noqa: F401  - re-exported for tests and external code
    AnalysisCancelled,
    available_model_names as _available_model_names,
    compliance_remaining as _compliance_remaining,
    compliance_timeout_error as _compliance_timeout_error,
    generation_model_aliases as _generation_model_aliases,
    progress_event as _progress_event,
    progress_note_excerpt as _progress_note_excerpt,
    raise_if_analysis_cancelled as _raise_if_analysis_cancelled,
    resolve_generation_model as _resolve_generation_model,
    set_compliance_client_timeout as _set_compliance_client_timeout,
)
from .services.health import build_full_health_payload, build_public_health_payload
from .services.jobs import (  # noqa: F401  - re-exported for tests and external code
    ANALYSIS_JOB_TTL_SEC as _ANALYSIS_JOB_TTL_SEC,
    ANALYSIS_TERMINAL_STATUSES as _ANALYSIS_TERMINAL_STATUSES,
    analysis_cancel_requested as _analysis_cancel_requested,
    analysis_job_event_stream as _analysis_job_event_stream,
    analysis_job_snapshot as _analysis_job_snapshot,
    append_analysis_event as _append_analysis_event,
    cleanup_analysis_jobs as _cleanup_analysis_jobs,
    run_analysis_job as _run_analysis_job,
    sse_event as _sse_event,
)
from .services.prompting import (  # noqa: F401  - re-exported for tests and external code
    analysis_context as _analysis_context,
    batch_prompt as _batch_prompt,
    checked_source_details as _checked_source_details,
    clamp_int as _clamp_int,
    compliance_final_prompt as _compliance_final_prompt,
    compliance_section_prompt as _compliance_section_prompt,
    final_prompt as _final_prompt,
    format_batch_doc as _format_batch_doc,
    format_requirement_doc as _format_requirement_doc,
    format_source_line as _format_source_line,
    locators_from_text as _locators_from_text,
    matrix_markdown as _matrix_markdown,
    pack_analysis_batches as _pack_analysis_batches,
    parse_matrix_rows as _parse_matrix_rows,
    safe_upload_name as _safe_upload_name,
    source_details as _source_details,
    split_checked_sections as _split_checked_sections,
)
from .text_utils import (  # noqa: F401  - re-exported for tests and external code
    clean_quote_text as _clean_quote_text,
    compact_dialog_context as _compact_dialog_context,
    filter_analysis_docs as _filter_analysis_docs,
    query_term_hits as _query_term_hits,
    query_terms as _query_terms,
    score as _score,
    trim_text as _trim_text,
)


# ---------------------------------------------------------------------------
# Runtime helpers and dependencies (re-exported for tests and external code)
# ---------------------------------------------------------------------------
configure_runtime_state = _runtime.configure_runtime_state
make_ollama_client = _runtime.make_ollama_client
model_to_dict = _runtime.model_to_dict
require_api_key = _runtime.require_api_key
_public_path_error = _runtime.public_path_error
build_health_payload = build_full_health_payload

# Re-export the runtime accessors as module attributes (identity-preserving) so
# ``rag_server.app.dependency_overrides[rag_server.get_config]`` resolves the
# same callable that route handlers depend on, and so ``monkeypatch.setattr``
# tests still observe the runtime functions by attribute name.
get_config = _runtime.get_config
get_service = _runtime.get_service


# ``require_api_key`` reads the active config; tests rely on
# ``monkeypatch.setattr(rag_server, "get_config", lambda: cfg)`` taking effect.
# ``set_get_config_provider`` lets the runtime resolve the current attribute on
# this module so a monkeypatched callable is picked up automatically.
_runtime.set_get_config_provider(lambda: globals()["get_config"]())


def _model_cache() -> dict[str, tuple[float, dict[str, Any]]]:
    return _runtime.model_cache()


def _analysis_jobs() -> dict[str, dict[str, Any]]:
    return _runtime.analysis_jobs()


def _analysis_jobs_lock() -> Any:
    return _runtime.analysis_jobs_lock()


def _refresh_service(new_config: SidecarConfig) -> None:
    _runtime.refresh_service(new_config)


def _apply_openapi_route_metadata() -> None:
    apply_openapi_route_metadata(app)


# ---------------------------------------------------------------------------
# Analysis service shim: tests monkeypatch ``rag_server.OllamaClient`` and
# ``rag_server.extract_text``; route handlers go through this module so the
# patched references take effect.
# ---------------------------------------------------------------------------
def _sync_analysis_service_dependencies() -> None:
    """Copy monkeypatched facade attributes into the analysis service.

    Existing tests patch ``rag_server.OllamaClient``, ``rag_server.extract_text``
    and sometimes the public analysis functions. Routes intentionally call the
    facade first, then this shim updates the extracted service module so those
    test-time substitutions still affect the implementation.
    """
    setattr(_analysis_service, "OllamaClient", OllamaClient)
    _analysis_service.extract_text = extract_text
    _analysis_service.is_supported_file = is_supported_file
    _analysis_service.filter_analysis_docs = _filter_analysis_docs
    _analysis_service.make_ollama_client = make_ollama_client


def run_multi_pass_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_analysis_service_dependencies()
    return _analysis_service.run_multi_pass_analysis(*args, **kwargs)


def run_compliance_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_analysis_service_dependencies()
    return _analysis_service.run_compliance_analysis(*args, **kwargs)


# ---------------------------------------------------------------------------
# FastAPI app instance (created at import time for backward compatibility)
# ---------------------------------------------------------------------------
app = _app_module.create_app()


def main() -> None:
    if uvicorn is None:
        raise RuntimeError(
            "FastAPI/Uvicorn dependencies are not installed. "
            "Install openwebui_zi_rag_requirements.txt"
        ) from _IMPORT_ERROR
    uvicorn.run("openwebui_zi_rag.server:app", host="127.0.0.1", port=8766, reload=False)


__all__ = [
    "app",
    "main",
    "configure_runtime_state",
    "get_config",
    "get_service",
    "make_ollama_client",
    "model_to_dict",
    "require_api_key",
    "run_multi_pass_analysis",
    "run_compliance_analysis",
    "build_health_payload",
    "build_full_health_payload",
    "build_public_health_payload",
    "OPENAPI_ROUTE_METADATA",
    "AnalysisCancelled",
    "OllamaCancelled",
    "OllamaClient",
    "OllamaError",
    "HTTPException",
    "JobStatus",
    "Registry",
    "RagService",
    "SidecarConfig",
    "load_config",
    "save_config",
    "update_config",
    "extract_text",
    "is_supported_file",
    "vector_store",
    "time",
    "AddPathRequest",
    "AnalyzeRequest",
    "ChatAttachmentFileMeta",
    "ChatAttachmentsRequest",
    "ComplianceRequest",
    "ConfigUpdate",
    "DeleteDocumentsRequest",
    "IndexCreate",
    "RebuildIndexRequest",
    "ReindexDocumentsRequest",
    "RetrieveRequest",
]
