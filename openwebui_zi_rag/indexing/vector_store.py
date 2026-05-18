from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


class VectorStoreError(RuntimeError):
    pass


_INDEX_CACHE_MAX_SIZE = 32
_INDEX_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_LOCKS: dict[str, threading.RLock] = {}
_INDEX_LOCKS_LOCK = threading.Lock()


def _imports() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except Exception as exc:
        raise VectorStoreError(
            "faiss and numpy are required. Install dependencies from openwebui_zi_rag_requirements.txt"
        ) from exc
    return faiss, np


def _normalize(matrix: Any) -> Any:
    _, np = _imports()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12)


def index_paths(base_dir: str | Path, index_id: str) -> tuple[Path, Path]:
    root = Path(base_dir) / index_id
    root.mkdir(parents=True, exist_ok=True)
    return root / "vectors.faiss", root / "vector_map.json"


def _cache_key(vector_path: Path) -> str:
    return str(vector_path.resolve())


def _index_lock(vector_path: Path) -> threading.RLock:
    key = _cache_key(vector_path)
    with _INDEX_LOCKS_LOCK:
        lock = _INDEX_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _INDEX_LOCKS[key] = lock
        return lock


@contextmanager
def acquire_index_lock(base_dir: str | Path, index_id: str) -> Iterator[None]:
    vector_path = Path(base_dir) / index_id / "vectors.faiss"
    with _index_lock(vector_path):
        yield


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _cache_set_locked(key: str, value: dict[str, Any]) -> None:
    _INDEX_CACHE[key] = value
    _INDEX_CACHE.move_to_end(key)
    while len(_INDEX_CACHE) > _INDEX_CACHE_MAX_SIZE:
        _INDEX_CACHE.popitem(last=False)


def _selected_index_type(index_type: str, chunk_count: int, hnsw_threshold_chunks: int) -> str:
    normalized = str(index_type or "auto").strip().lower()
    if normalized not in {"auto", "flat", "hnsw"}:
        raise VectorStoreError("index_type must be one of: auto, flat, hnsw")
    if normalized == "auto":
        return "hnsw" if int(chunk_count) > int(hnsw_threshold_chunks) else "flat"
    return normalized


def _build_faiss_index(
    faiss: Any,
    dimension: int,
    index_type: str,
    *,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
) -> Any:
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(int(dimension), int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(hnsw_ef_construction)
        index.hnsw.efSearch = int(hnsw_ef_search)
        return index
    return faiss.IndexFlatIP(int(dimension))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_faiss_atomic(faiss: Any, index: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        faiss.write_index(index, str(tmp_path))
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def invalidate_index_cache(base_dir: str | Path, index_id: str) -> None:
    vector_path = Path(base_dir) / index_id / "vectors.faiss"
    with _index_lock(vector_path):
        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE.pop(_cache_key(vector_path), None)


def clear_index_cache() -> None:
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def build_index(
    base_dir: str | Path,
    index_id: str,
    chunk_ids: list[str],
    embeddings: Iterable[Iterable[float]],
    *,
    index_type: str = "auto",
    hnsw_threshold_chunks: int = 50000,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
) -> int:
    faiss, np = _imports()
    vectors = np.array(list(embeddings), dtype=np.float32)
    vector_path, map_path = index_paths(base_dir, index_id)
    with _index_lock(vector_path):
        if vectors.size == 0 or not chunk_ids:
            if vector_path.exists():
                vector_path.unlink()
            _write_text_atomic(map_path, "[]")
            with _INDEX_CACHE_LOCK:
                _INDEX_CACHE.pop(_cache_key(vector_path), None)
            return 0

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(chunk_ids):
            raise VectorStoreError("Embedding count does not match chunk count")

        vectors = _normalize(vectors)
        selected_index_type = _selected_index_type(index_type, len(chunk_ids), hnsw_threshold_chunks)
        index = _build_faiss_index(
            faiss,
            int(vectors.shape[1]),
            selected_index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
        index.add(vectors)
        _write_faiss_atomic(faiss, index, vector_path)
        _write_text_atomic(map_path, json.dumps(chunk_ids, ensure_ascii=False, indent=2))
        with _INDEX_CACHE_LOCK:
            _cache_set_locked(
                _cache_key(vector_path),
                {
                    "index": index,
                    "chunk_ids": list(chunk_ids),
                    "vector_sig": _file_signature(vector_path),
                    "map_sig": _file_signature(map_path),
                },
            )
        return int(vectors.shape[1])


def _cached_index(base_dir: str | Path, index_id: str) -> tuple[Any | None, list[str]]:
    faiss, _ = _imports()
    vector_path, map_path = index_paths(base_dir, index_id)
    with _index_lock(vector_path):
        if not vector_path.exists() or not map_path.exists():
            return None, []

        vector_sig = _file_signature(vector_path)
        map_sig = _file_signature(map_path)
        key = _cache_key(vector_path)
        with _INDEX_CACHE_LOCK:
            cached = _INDEX_CACHE.get(key)
            if (
                cached
                and cached.get("vector_sig") == vector_sig
                and cached.get("map_sig") == map_sig
            ):
                _INDEX_CACHE.move_to_end(key)
                return cached["index"], list(cached["chunk_ids"])
            if cached:
                _INDEX_CACHE.pop(key, None)

        raw_chunk_ids = json.loads(map_path.read_text(encoding="utf-8"))
        chunk_ids = [str(chunk_id) for chunk_id in raw_chunk_ids] if isinstance(raw_chunk_ids, list) else []
        if not chunk_ids:
            with _INDEX_CACHE_LOCK:
                _INDEX_CACHE.pop(key, None)
            return None, []

        index = faiss.read_index(str(vector_path))
        with _INDEX_CACHE_LOCK:
            _cache_set_locked(
                key,
                {
                    "index": index,
                    "chunk_ids": list(chunk_ids),
                    "vector_sig": vector_sig,
                    "map_sig": map_sig,
                },
            )
        return index, chunk_ids


def search_index(
    base_dir: str | Path,
    index_id: str,
    query_embedding: Iterable[float],
    top_k: int,
) -> list[tuple[str, float]]:
    _, np = _imports()
    vector_path, _ = index_paths(base_dir, index_id)
    with _index_lock(vector_path):
        index, chunk_ids = _cached_index(base_dir, index_id)
        if index is None:
            return []
        if not chunk_ids:
            return []

        query = np.array(list(query_embedding), dtype=np.float32).reshape(1, -1)
        if query.shape[1] != index.d:
            raise VectorStoreError(
                f"Query embedding dimension {query.shape[1]} does not match index dimension {index.d}"
            )
        query = _normalize(query)
        limit = max(1, min(int(top_k), len(chunk_ids)))
        scores, ids = index.search(query, limit)
        results: list[tuple[str, float]] = []
        for raw_score, raw_idx in zip(scores[0], ids[0]):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(chunk_ids):
                continue
            score = max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))
            results.append((str(chunk_ids[idx]), score))
        return results
