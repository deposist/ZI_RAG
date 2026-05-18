"""Health probe helpers used by ``/health``."""

from __future__ import annotations

from typing import Any

from ..config import SidecarConfig
from ..indexing import vector_store
from ..indexing.service import RagService
from ..runtime import make_ollama_client

from .. import __version__ as ZI_RAG_VERSION


def _health_error(exc: Exception) -> dict[str, Any]:
    return {"status": "error", "error": str(exc or "unknown error")}


def _sqlite_health(service: RagService) -> dict[str, Any]:
    try:
        with service.registry.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM indexes").fetchone()
        return {"status": "ok", "index_count": int(row["count"] if row else 0)}
    except Exception as exc:
        return _health_error(exc)


def _ollama_health(cfg: SidecarConfig) -> dict[str, Any]:
    try:
        timeout = max(1, min(int(cfg.request_timeout_sec or 5), 5))
        models = make_ollama_client(cfg, request_timeout=timeout).list_models()
        return {"status": "ok", "model_count": len(models)}
    except Exception as exc:
        return _health_error(exc)


def _faiss_health(cfg: SidecarConfig, service: RagService) -> dict[str, Any]:
    try:
        indexes = service.registry.list_indexes()
        index_count = len(indexes)
        for item in indexes:
            index_id = str(item.get("id") or "").strip()
            if not index_id:
                continue
            vector_path = cfg.indexes_path / index_id / "vectors.faiss"
            map_path = cfg.indexes_path / index_id / "vector_map.json"
            if not vector_path.exists() and not map_path.exists():
                continue
            if not vector_path.exists() or not map_path.exists():
                return {
                    "status": "error",
                    "index_id": index_id,
                    "index_count": index_count,
                    "error": "FAISS index files are incomplete",
                }
            index, chunk_ids = vector_store._cached_index(cfg.indexes_path, index_id)
            if index is None:
                return {
                    "status": "error",
                    "index_id": index_id,
                    "index_count": index_count,
                    "error": "FAISS index could not be loaded",
                }
            return {
                "status": "ok",
                "index_id": index_id,
                "index_count": index_count,
                "chunk_count": len(chunk_ids),
                "vector_count": int(getattr(index, "ntotal", 0)),
            }
        return {"status": "skipped", "index_count": index_count, "reason": "no FAISS index files"}
    except Exception as exc:
        return _health_error(exc)


def _embedding_model_dimension_health(cfg: SidecarConfig, service: RagService) -> dict[str, Any]:
    indexes: list[dict[str, Any]] = []
    warnings: list[str] = []
    current_model = str(cfg.embedding_model or "").strip()
    for item in service.registry.list_indexes():
        index_id = str(item.get("id") or "").strip()
        index_model = str(item.get("embedding_model") or "").strip()
        embedding_dim = int(item.get("embedding_dim") or 0)
        indexes.append(
            {
                "index_id": index_id,
                "name": item.get("name") or index_id,
                "embedding_model": index_model,
                "embedding_dim": embedding_dim,
            }
        )
        if current_model and index_model and index_model != current_model:
            warnings.append(
                f"Index {index_id} uses embedding_model={index_model}, current config embedding_model={current_model}"
            )
    return {
        "current_embedding_model": current_model,
        "indexes": indexes,
        "warnings": warnings,
    }


def _health_checks(cfg: SidecarConfig, service: RagService) -> dict[str, dict[str, Any]]:
    return {
        "sqlite": _sqlite_health(service),
        "ollama": _ollama_health(cfg),
        "faiss": _faiss_health(cfg, service),
    }


def _status_code_for_checks(checks: dict[str, dict[str, Any]]) -> tuple[int, list[str]]:
    unhealthy = [name for name, result in checks.items() if result.get("status") == "error"]
    return (503 if unhealthy else 200), unhealthy


def _public_check(name: str, result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "unknown")
    payload: dict[str, Any] = {"status": status}
    if name == "sqlite" and "index_count" in result:
        payload["index_count"] = result["index_count"]
    if name == "ollama" and "model_count" in result:
        payload["model_count"] = result["model_count"]
    if name == "faiss":
        payload["index_count"] = int(result.get("index_count") or 0)
    if status == "skipped" and result.get("reason"):
        payload["reason"] = result["reason"]
    if status == "error":
        payload["error"] = "check failed"
    return payload


def build_public_health_payload(cfg: SidecarConfig, service: RagService) -> tuple[int, dict[str, Any]]:
    checks = _health_checks(cfg, service)
    status_code, unhealthy = _status_code_for_checks(checks)
    payload: dict[str, Any] = {
        "status": "error" if unhealthy else "ok",
        "version": ZI_RAG_VERSION,
        "checks": {name: _public_check(name, result) for name, result in checks.items()},
    }
    return status_code, payload


def build_full_health_payload(cfg: SidecarConfig, service: RagService) -> tuple[int, dict[str, Any]]:
    checks = _health_checks(cfg, service)
    status_code, unhealthy = _status_code_for_checks(checks)
    payload: dict[str, Any] = {
        "status": "error" if unhealthy else "ok",
        "storage_dir": str(cfg.storage_path),
        "registry": str(cfg.registry_path),
        "version": ZI_RAG_VERSION,
        "checks": checks,
        "embedding_model_dimension": _embedding_model_dimension_health(cfg, service),
        "metrics": service.metrics_snapshot(),
    }
    if unhealthy:
        payload["unhealthy"] = unhealthy
    return status_code, payload


__all__ = ["build_full_health_payload", "build_public_health_payload"]
