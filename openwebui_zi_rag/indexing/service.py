from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import mimetypes
import os
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from ..config import DEFAULT_QUERY_SYNONYMS, SidecarConfig
from ..ollama_client import OllamaClient, make_embedding_client, make_generation_client, make_rerank_client
from .chunking import chunk_text
from .extraction import clear_ocr_gpu_cache, collect_supported_files, easyocr_reader_count, extract_text
from .registry import DocumentStatus, JobStatus, Registry, ensure_inside_allowed_roots, file_sha256, sanitize_id
from .vector_store import acquire_index_lock, build_index, invalidate_index_cache, search_index


LOCATOR_RE = re.compile(r"\[([^\]]*(?:абз\.|стр\.|пункт|таблица|лист|строка|тело письма)[^\]]*)\]", re.I)
INDEXING_JOB_KINDS = {
    "index_document",
    "index_documents",
    "reindex_document",
    "reindex_documents",
    "force_reindex_document",
    "force_reindex_documents",
}
OPENAI_COMPATIBLE_EMBEDDING_PROVIDERS = {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}
MMR_LAMBDA = 0.75
RRF_K = 60.0
QUERY_EXPANSION_SYSTEM_PROMPT = (
    "Ты расширяешь поисковые запросы для RAG по корпоративным документам. "
    "Верни только JSON-массив строк без пояснений. "
    "Добавь короткие альтернативные формулировки, синонимы и при необходимости один HyDE-фрагмент. "
    "Не отвечай на вопрос пользователя."
)


def safe_filename(filename: str) -> str:
    name = Path(filename or "upload").name.replace("\x00", "").strip()
    name = re.sub(r"[\r\n\t]", "_", name)
    if name in {"", ".", ".."}:
        name = "upload"
    return name[:180] or "upload"


def content_sha256(content: bytes) -> str:
    return hashlib.sha256(bytes(content or b"")).hexdigest()


def write_unique_upload(upload_root: Path, safe_name: str, content: bytes, *, attempts: int = 20) -> Path:
    base = Path(safe_name)
    stem = base.stem or "upload"
    suffix = base.suffix
    for _ in range(max(1, attempts)):
        token = secrets.token_hex(8)
        candidate = upload_root / f"{stem}_{token}{suffix}"
        fd = -1
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(content)
        except Exception:
            if fd >= 0:
                os.close(fd)
            candidate.unlink(missing_ok=True)
            raise
        return candidate
    raise FileExistsError(f"Could not allocate unique upload path for {safe_name}")


def chunk_locator(text: str, chunk_no: int | None = None) -> str:
    locators: list[str] = []
    for match in LOCATOR_RE.finditer(text or ""):
        locator = match.group(1).strip()
        if locator and locator not in locators:
            locators.append(locator)
        if len(locators) >= 3:
            break
    if locators:
        return " / ".join(locators)
    return f"chunk {chunk_no}" if chunk_no is not None else ""


def clean_quote_text(text: str) -> str:
    quote = re.sub(r"\[[^\]]+\]", " ", str(text or ""))
    quote = re.sub(r"(?:\s*\|\s*(?:-|–|—|v|V|x|X|✓|✔)?\s*){3,}", " ", quote)
    quote = re.sub(r"\s+\|(?=\s*[.;,]|$)", " ", quote)
    quote = re.sub(r"\|\s+", "| ", quote)
    quote = re.sub(r"\s+", " ", quote).strip()
    return quote.strip(" |")


def chunk_quote(text: str, max_chars: int = 420) -> str:
    quote = clean_quote_text(text)
    if len(quote) <= max_chars:
        return quote
    return quote[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _chunk_dedupe_key(item: dict[str, Any]) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip().lower()
    if text:
        return "text", text[:1200]
    source = str(item.get("source") or "").strip().lower()
    chunk_id = str(item.get("chunk_id") or item.get("chunk_no") or "")
    return source, chunk_id


def _normalize_embedding(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        return []
    return [float(value) / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + max(1, int(rank)))


def _lexical_score_from_fts_rank(fts_rank: int) -> float:
    normalized = 1.0 / (1.0 + max(0, int(fts_rank) - 1))
    return min(0.95, normalized)


def _query_synonym_matches(trigger: str, normalized_query: str) -> bool:
    tokens = [token for token in re.split(r"[\s,;+/|]+", trigger.lower().replace("ё", "е")) if token]
    return bool(tokens) and all(token in normalized_query for token in tokens)


def retrieval_query_variants(query: str, query_synonyms: dict[str, list[str]] | None = None) -> list[str]:
    original = re.sub(r"\s+", " ", str(query or "")).strip()
    if not original:
        return []

    variants: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", value or "").strip(" .;:,-")
        if cleaned and cleaned.lower() not in {item.lower() for item in variants}:
            variants.append(cleaned)

    add(original)
    focus = original
    cleanup_patterns = [
        r"\bнормативно[- ]методическ\w*\s+документ\w*\b",
        r"\bнмд\b",
        r"\bпроанализируй\b",
        r"\bпроанализировать\b",
        r"\bпроверь\b",
        r"\bпроверить\b",
        r"\bпроверка\b",
        r"\bполный\s+перечень\b",
        r"\bперечень\b",
        r"\bвсе\s+требования\b",
        r"\bкакие\s+пункты\b",
        r"\bпункты\b",
    ]
    for pattern in cleanup_patterns:
        focus = re.sub(pattern, " ", focus, flags=re.I)
    add(focus)

    normalized_query = f"{original} {focus}".lower().replace("ё", "е")
    synonyms = {
        **DEFAULT_QUERY_SYNONYMS,
        **(query_synonyms or {}),
    }
    for trigger, expansions in (synonyms or {}).items():
        if not _query_synonym_matches(str(trigger), normalized_query):
            continue
        if isinstance(expansions, str):
            add(expansions)
            continue
        for expansion in expansions or []:
            add(str(expansion))

    return variants[:8]


def llm_query_expansion_variants(raw: str, max_variants: int = 3) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    candidates: list[Any] = []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                loaded = json.loads(match.group(0))
            except json.JSONDecodeError:
                loaded = None
        else:
            loaded = None
    if isinstance(loaded, list):
        candidates.extend(loaded)
    elif isinstance(loaded, dict):
        for key in ("queries", "query_variants", "variants", "expansions", "hypothetical_document", "hyde"):
            value = loaded.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, str):
                candidates.append(value)
    if not candidates:
        candidates.extend(text.splitlines())

    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = re.sub(r"\s+", " ", candidate).strip(" \t\r\n\"'`;,.")
        cleaned = re.sub(r"^(?:[-*]+|\d+[\).:-])\s*", "", cleaned).strip()
        if not cleaned:
            continue
        cleaned = cleaned[:600].strip()
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        variants.append(cleaned)
        if len(variants) >= max(1, int(max_variants or 1)):
            break
    return variants


class JobCancelled(RuntimeError):
    pass


class IndexingDeadlineExceeded(TimeoutError):
    pass


class RagService:
    def __init__(
        self,
        config: SidecarConfig,
        *,
        registry: Registry | None = None,
        embedding_client: OllamaClient | None = None,
        rerank_client: Any | None = None,
        generation_client: Any | None = None,
    ):
        self.config = config
        self.config.ensure_dirs()
        self.registry = registry or Registry(config.registry_path)
        if registry is None:
            self.registry.cancel_stale_jobs()
        self.embedding_client = embedding_client or make_embedding_client(config)
        self.rerank_client = rerank_client if rerank_client is not None else make_rerank_client(config)
        timeout = float(getattr(config, "request_timeout_sec", 120) or 120)
        self.generation_client = generation_client if generation_client is not None else make_generation_client(
            config,
            request_timeout=timeout,
        )
        self._metrics: dict[str, dict[str, Any]] = {}
        self._metrics_lock = threading.Lock()
        self._rebuild_debounce_condition = threading.Condition()
        self._rebuild_debounce: dict[str, dict[str, Any]] = {}

    def create_index(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.registry.create_index(
            payload.get("name") or payload.get("id") or "Knowledge",
            index_id=payload.get("id"),
            description=payload.get("description") or "",
            embedding_model=payload.get("embedding_model") or self.config.embedding_model,
            chunk_size=int(payload.get("chunk_size") or self.config.chunk_size),
            chunk_overlap=int(payload.get("chunk_overlap") or self.config.chunk_overlap),
            index_type=payload.get("index_type") or self.config.index_type,
        )

    def save_upload(
        self,
        index_id: str,
        filename: str,
        content: bytes,
        mime_type: str = "",
        *,
        external_id: str = "",
        external_source: str = "",
        metadata: dict[str, Any] | None = None,
        file_hash: str = "",
    ) -> dict[str, Any]:
        upload_root = self.config.uploads_path / index_id
        upload_root.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(filename)
        content_bytes = bytes(content or b"")
        target = write_unique_upload(upload_root, safe_name, content_bytes)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        try:
            return self.registry.create_document(
                index_id,
                filename=safe_name,
                stored_path=str(target),
                mime_type=mime_type,
                file_hash=file_hash or content_sha256(content_bytes),
                external_id=external_id,
                external_source=external_source,
                metadata_json=metadata_json,
            )
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def chat_attachment_index_id(self, scope_id: str) -> str:
        raw_prefix = str(getattr(self.config, "chat_attachment_index_prefix", "owui_chat_") or "owui_chat_").strip()
        prefix = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", raw_prefix)
        prefix = re.sub(r"_+", "_", prefix)[:40] or "owui_chat_"
        scope = sanitize_id(scope_id)
        return sanitize_id(f"{prefix}{scope}")

    def ensure_chat_attachment_index(self, scope_id: str) -> dict[str, Any]:
        index_id = self.chat_attachment_index_id(scope_id)
        existing = self.registry.get_index(index_id)
        if existing:
            return existing
        return self.create_index(
            {
                "id": index_id,
                "name": f"OpenWebUI chat attachments: {scope_id}",
                "description": "Files attached to one OpenWebUI chat. Managed automatically by ZI_RAG.",
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "index_type": self.config.index_type,
            }
        )

    def upsert_chat_attachment(
        self,
        index_id: str,
        *,
        filename: str,
        content: bytes,
        mime_type: str = "",
        external_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        digest = content_sha256(content)
        source = "openwebui"
        existing = self.registry.get_active_document_by_external(index_id, source, external_id)
        if existing and str(existing.get("file_hash") or "") == digest:
            return {
                "status": "skipped",
                "reason": "already_indexed",
                "document": existing,
            }
        replaced = None
        if existing:
            replaced = self.registry.soft_delete_document(str(existing["id"]))
            if replaced:
                self.remove_storage_file(replaced)
        document = self.save_upload(
            index_id,
            filename,
            content,
            mime_type,
            external_id=external_id,
            external_source=source,
            metadata=metadata,
            file_hash=digest,
        )
        return {
            "status": "created" if not replaced else "replaced",
            "reason": "",
            "document": document,
            "replaced": replaced,
        }

    def index_chat_attachments(
        self,
        scope_id: str,
        files: list[dict[str, Any]],
        *,
        chat_id: str = "",
        user_id: str = "",
        message_id: str = "",
        deadline: float | None = None,
    ) -> dict[str, Any]:
        scope = str(scope_id or "").strip()
        if not scope:
            raise ValueError("chat attachment scope is empty")
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        index = self.ensure_chat_attachment_index(scope)
        index_id = str(index["id"])
        documents: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        selected_document_ids: list[str] = []

        for pos, item in enumerate(files, start=1):
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            filename = safe_filename(str(item.get("filename") or f"attachment-{pos}"))
            content = bytes(item.get("content") or b"")
            metadata = dict(item.get("metadata") or {})
            external_id = str(item.get("external_id") or metadata.get("id") or "").strip()
            if not external_id:
                external_id = f"{filename}:{content_sha256(content)}"
            metadata.update(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "user_id": user_id,
                    "original_filename": item.get("filename") or filename,
                }
            )
            try:
                result = self.upsert_chat_attachment(
                    index_id,
                    filename=filename,
                    content=content,
                    mime_type=str(item.get("content_type") or ""),
                    external_id=external_id,
                    metadata=metadata,
                )
            except IndexingDeadlineExceeded:
                raise
            except Exception as exc:
                failed.append(
                    {
                        "filename": filename,
                        "external_id": external_id,
                        "error": str(exc),
                    }
                )
                continue
            document = dict(result.get("document") or {})
            if result.get("status") == "skipped":
                skipped.append(
                    {
                        "document_id": document.get("id"),
                        "filename": filename,
                        "external_id": external_id,
                        "reason": result.get("reason") or "already_indexed",
                    }
                )
                documents.append(document)
                continue
            documents.append(document)
            if document.get("id"):
                selected_document_ids.append(str(document["id"]))
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")

        indexing: dict[str, Any] | None = None
        if selected_document_ids:
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            indexing = self.index_documents_now(index_id, selected_document_ids, deadline=deadline)
            failed.extend(indexing.get("failed") or [])
        return {
            "index": self.registry.get_index(index_id) or index,
            "index_id": index_id,
            "documents": documents,
            "indexed_document_ids": selected_document_ids,
            "skipped": skipped,
            "failed": failed,
            "indexing": indexing,
        }

    def add_path(
        self,
        index_id: str,
        path: str,
        *,
        recursive: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        root = ensure_inside_allowed_roots(path, self.config.allowed_source_roots)
        files = collect_supported_files(root, recursive=recursive)
        include = include or []
        exclude = exclude or []
        selected: list[Path] = []
        for item in files:
            rel = str(item.relative_to(root)) if root.is_dir() else item.name
            if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
                continue
            if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                continue
            selected.append(item)

        docs = []
        for file_path in selected:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            docs.append(
                self.registry.create_document(
                    index_id,
                    filename=file_path.name,
                    source_path=str(file_path),
                    mime_type=mime_type or "",
                    file_hash=file_sha256(file_path),
                )
            )
        return docs

    def _document_path(self, document: dict[str, Any]) -> Path:
        path = document.get("stored_path") or document.get("source_path")
        if not path:
            raise FileNotFoundError("Document has no stored_path/source_path")
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"File not found: {candidate.name}")
        return candidate

    def _raise_if_cancelled(self, job_id: str | None) -> None:
        if self.registry.job_cancel_requested(job_id):
            raise JobCancelled("Indexing canceled")

    def _raise_if_deadline_expired(self, deadline: float | None, message: str = "Indexing timed out") -> None:
        if deadline is not None and time.monotonic() >= float(deadline):
            raise IndexingDeadlineExceeded(message)

    def _job_message(self, job_id: str | None, message: str) -> None:
        if job_id:
            self.registry.update_job(job_id, JobStatus.RUNNING, message=message)

    def _set_documents_status(
        self,
        document_ids: list[str],
        status: str | DocumentStatus,
        *,
        error: str = "",
    ) -> None:
        for document_id in document_ids:
            self.registry.set_document_status(document_id, status, error=error)

    def _embedding_inputs(self, texts: list[str], *, query: bool = False) -> list[str]:
        prefix = self.config.embedding_query_prefix if query else self.config.embedding_document_prefix
        prefix = str(prefix or "")
        if not prefix:
            return texts
        return [f"{prefix}{text}" for text in texts]

    def _embedding_text_hash(self, text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

    def _embedding_provider_label(self) -> str:
        return str(getattr(self.config, "embedding_provider", "ollama") or "ollama")

    def _embedding_cache_dtype(self) -> str:
        dtype = str(getattr(self.config, "embedding_cache_dtype", "fp32") or "fp32").strip().lower()
        return dtype if dtype in {"fp32", "fp16"} else "fp32"

    def _parallel_embedding_batches_enabled(self) -> bool:
        return self._embedding_provider_label().strip().lower() in OPENAI_COMPATIBLE_EMBEDDING_PROVIDERS

    def _embedding_cache_key(self, model: str, *, query: bool = False) -> str:
        provider = self._embedding_provider_label().strip().lower()
        base_url = (
            str(getattr(self.config, "embedding_base_url", "") or "")
            if provider in {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}
            else str(getattr(self.config, "ollama_base_url", "") or "")
        )
        prefix = (
            str(getattr(self.config, "embedding_query_prefix", "") or "")
            if query
            else str(getattr(self.config, "embedding_document_prefix", "") or "")
        )
        payload = json.dumps(
            {
                "provider": provider,
                "base_url": base_url,
                "model": model,
                "prefix": prefix,
                "kind": "query" if query else "document",
                "cache_dtype": self._embedding_cache_dtype(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"{model}|cache:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    def _embedding_batch_message(
        self,
        *,
        batch_no: int,
        total_batches: int,
        model: str,
        cached_count: int,
        missing_count: int,
    ) -> str:
        return (
            f"Embeddings batch {batch_no}/{total_batches} через {self._embedding_provider_label()}: "
            f"{model} (cached {cached_count}, new {missing_count})"
        )

    def _embed_missing_batches(
        self,
        *,
        model: str,
        missing: list[tuple[str, str]],
        batch_size: int,
        cached_count: int,
        job_id: str | None,
        deadline: float | None = None,
        prepared_inputs: bool = False,
    ) -> list[tuple[list[str], list[list[float]]]]:
        batches: list[tuple[int, list[str], list[str]]] = []
        for start in range(0, len(missing), batch_size):
            batch_no = start // batch_size + 1
            batch_pairs = missing[start : start + batch_size]
            batch_ids = [item[0] for item in batch_pairs]
            batch_texts = [item[1] for item in batch_pairs]
            if not prepared_inputs:
                batch_texts = self._embedding_inputs(batch_texts, query=False)
            batches.append((batch_no, batch_ids, batch_texts))
        if not batches:
            return []
        total_batches = len(batches)
        if self._parallel_embedding_batches_enabled() and total_batches > 1:
            return self._embed_missing_batches_parallel(
                model=model,
                batches=batches,
                total_batches=total_batches,
                cached_count=cached_count,
                missing_count=len(missing),
                job_id=job_id,
                deadline=deadline,
            )

        results: list[tuple[list[str], list[list[float]]]] = []
        for batch_no, batch_ids, batch_texts in batches:
            self._raise_if_cancelled(job_id)
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            self._job_message(
                job_id,
                self._embedding_batch_message(
                    batch_no=batch_no,
                    total_batches=total_batches,
                    model=model,
                    cached_count=cached_count,
                    missing_count=len(missing),
                ),
            )
            batch_vectors = self.embedding_client.embed(model, batch_texts)
            if len(batch_vectors) != len(batch_ids):
                raise ValueError("Embedding endpoint returned a different number of vectors")
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            results.append((batch_ids, batch_vectors))
        return results

    def _embed_missing_batches_parallel(
        self,
        *,
        model: str,
        batches: list[tuple[int, list[str], list[str]]],
        total_batches: int,
        cached_count: int,
        missing_count: int,
        job_id: str | None,
        deadline: float | None = None,
    ) -> list[tuple[list[str], list[list[float]]]]:
        max_workers = min(4, max(2, total_batches))
        results: list[tuple[list[str], list[list[float]]] | None] = [None] * total_batches
        next_batch_index = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Any, int] = {}

            def submit_next() -> None:
                nonlocal next_batch_index
                batch_no, _batch_ids, batch_texts = batches[next_batch_index]
                self._raise_if_cancelled(job_id)
                self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
                self._job_message(
                    job_id,
                    self._embedding_batch_message(
                        batch_no=batch_no,
                        total_batches=total_batches,
                        model=model,
                        cached_count=cached_count,
                        missing_count=missing_count,
                    ),
                )
                pending[executor.submit(self.embedding_client.embed, model, batch_texts)] = next_batch_index
                next_batch_index += 1

            while next_batch_index < total_batches and len(pending) < max_workers:
                submit_next()

            while pending:
                self._raise_if_cancelled(job_id)
                try:
                    self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
                except IndexingDeadlineExceeded:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise
                done, _not_done = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    batch_index = pending.pop(future)
                    _batch_no, batch_ids, _batch_texts = batches[batch_index]
                    try:
                        batch_vectors = future.result()
                    except Exception:
                        for pending_future in pending:
                            pending_future.cancel()
                        raise
                    if len(batch_vectors) != len(batch_ids):
                        raise ValueError("Embedding endpoint returned a different number of vectors")
                    self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
                    results[batch_index] = (batch_ids, batch_vectors)
                    if next_batch_index < total_batches:
                        submit_next()

        return [item for item in results if item is not None]

    def _record_metric(self, name: str, seconds: float, extra: dict[str, Any] | None = None) -> None:
        elapsed = max(0.0, float(seconds))
        with self._metrics_lock:
            item = self._metrics.setdefault(
                name,
                {
                    "count": 0,
                    "total_sec": 0.0,
                    "last_sec": 0.0,
                    "max_sec": 0.0,
                    "last_extra": {},
                },
            )
            item["count"] += 1
            item["total_sec"] += elapsed
            item["last_sec"] = elapsed
            item["max_sec"] = max(float(item.get("max_sec") or 0.0), elapsed)
            item["avg_sec"] = item["total_sec"] / max(1, int(item["count"]))
            item["last_extra"] = dict(extra or {})

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            snapshot = {
                name: {
                    **data,
                    "last_extra": dict(data.get("last_extra") or {}),
                }
                for name, data in self._metrics.items()
            }
        return {
            name: {
                **data,
                "total_sec": round(float(data.get("total_sec") or 0.0), 6),
                "last_sec": round(float(data.get("last_sec") or 0.0), 6),
                "max_sec": round(float(data.get("max_sec") or 0.0), 6),
                "avg_sec": round(float(data.get("avg_sec") or 0.0), 6),
            }
            for name, data in snapshot.items()
        }

    def clear_ocr_gpu_cache(self, *, unload_readers: bool = True) -> dict[str, Any]:
        result = clear_ocr_gpu_cache(unload_readers=unload_readers)
        if result.get("torch_loaded") or result.get("readers_before"):
            self._record_metric(
                "ocr_gpu_cache_clear",
                float(result.get("elapsed_sec") or 0.0),
                {
                    "readers_before": result.get("readers_before", 0),
                    "readers_after": result.get("readers_after", 0),
                    "freed_reserved_mb": result.get("freed_reserved_mb", 0),
                    "freed_allocated_mb": result.get("freed_allocated_mb", 0),
                },
            )
        return result

    def _auto_clear_ocr_gpu_cache(self, job_id: str, job_kind: str) -> dict[str, Any]:
        if job_kind not in INDEXING_JOB_KINDS or easyocr_reader_count() <= 0:
            return {}

        active_jobs = self.registry.list_jobs(
            statuses=[JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED],
            limit=500,
        )
        other_indexing_jobs = [
            str(job.get("id") or "")
            for job in active_jobs
            if str(job.get("id") or "") != job_id and str(job.get("kind") or "") in INDEXING_JOB_KINDS
        ]
        if other_indexing_jobs:
            return {
                "skipped": True,
                "reason": "active_indexing_jobs",
                "active_job_ids": other_indexing_jobs[:10],
            }

        self._job_message(job_id, "Очистка OCR GPU cache после индексации")
        result = self.clear_ocr_gpu_cache(unload_readers=True)
        result["auto"] = True
        return result

    def _rebuild_index_debounced(self, index_id: str, *, job_id: str | None = None) -> dict[str, Any]:
        delay = max(0.0, float(getattr(self.config, "rebuild_debounce_sec", 0.0) or 0.0))
        if delay <= 0.0 or not job_id:
            return self.rebuild_index_now(index_id, job_id=job_id)

        self._job_message(job_id, f"Ожидание {delay:.1f}s для объединения быстрых перестроек")
        condition = self._rebuild_debounce_condition
        with condition:
            state = self._rebuild_debounce.setdefault(
                index_id,
                {
                    "generation": 0,
                    "completed_generation": 0,
                    "deadline": 0.0,
                    "running": False,
                    "result": None,
                    "error": None,
                },
            )
            state["generation"] += 1
            my_generation = int(state["generation"])
            state["deadline"] = time.monotonic() + delay
            state["error"] = None
            condition.notify_all()

            while True:
                self._raise_if_cancelled(job_id)
                if int(state.get("completed_generation") or 0) >= my_generation:
                    if state.get("error"):
                        raise state["error"]
                    return dict(state.get("result") or {})

                if state.get("running"):
                    condition.wait(timeout=0.2)
                    continue

                remaining = float(state.get("deadline") or 0.0) - time.monotonic()
                if remaining > 0.0:
                    condition.wait(timeout=min(remaining, 0.2))
                    continue

                rebuild_generation = int(state["generation"])
                state["running"] = True
                break

        try:
            result = self.rebuild_index_now(index_id, job_id=job_id)
            error: BaseException | None = None
        except BaseException as exc:
            result = None
            error = exc

        with condition:
            state = self._rebuild_debounce[index_id]
            state["running"] = False
            state["completed_generation"] = rebuild_generation
            state["result"] = result
            state["error"] = error
            condition.notify_all()

        if error:
            raise error
        return dict(result or {})

    def _extraction_message(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if getattr(self.config, "enable_ocr", False):
                engine = getattr(self.config, "ocr_engine", "easyocr") or "easyocr"
                device = "GPU" if getattr(self.config, "ocr_gpu", True) and engine.lower() != "tesseract" else "CPU"
                return f"Извлечение текста PDF; OCR fallback использует {engine} на {device}"
            return "Извлечение текстового слоя PDF через pdfplumber"
        return f"Извлечение текста из {suffix or 'document'}"

    def index_document_chunks_now(
        self,
        document_id: str,
        *,
        job_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        self._raise_if_cancelled(job_id)
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        document = self.registry.get_document(document_id)
        if not document or document.get("deleted_at"):
            raise FileNotFoundError(f"Document not found: {document_id}")
        index = self.registry.get_index(document["index_id"])
        if not index:
            raise FileNotFoundError(f"Index not found: {document['index_id']}")

        self.registry.set_document_status(document_id, DocumentStatus.EXTRACTING)
        timings: dict[str, float] = {}
        try:
            self._raise_if_cancelled(job_id)
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            document_path = self._document_path(document)
            self._job_message(job_id, self._extraction_message(document_path))
            started = time.perf_counter()
            text = extract_text(document_path, self.config)
            timings["extraction_sec"] = time.perf_counter() - started
            self._raise_if_cancelled(job_id)
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            self._job_message(
                job_id,
                f"Разбиение текста на chunks ({len(text)} символов, extraction {timings['extraction_sec']:.2f}s)",
            )
            started = time.perf_counter()
            chunks = chunk_text(
                text,
                int(index.get("chunk_size") or self.config.chunk_size),
                int(index.get("chunk_overlap") or self.config.chunk_overlap),
            )
            timings["chunking_sec"] = time.perf_counter() - started
            if not chunks:
                raise ValueError("Document text is empty after extraction")
            self._raise_if_cancelled(job_id)
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            self.registry.replace_document_chunks(index["id"], document_id, chunks)
            self.registry.set_document_status(
                document_id,
                DocumentStatus.VECTORIZING,
                text_chars=len(text),
                chunk_count=len(chunks),
            )
            self._job_message(job_id, f"Сохранено chunks: {len(chunks)}")
            return {
                "document_id": document_id,
                "index_id": index["id"],
                "filename": document.get("filename") or document_id,
                "chunks": len(chunks),
                "text_chars": len(text),
                "timings": {key: round(value, 6) for key, value in timings.items()},
            }
        except JobCancelled:
            self.registry.replace_document_chunks(index["id"], document_id, [])
            self.registry.set_document_status(document_id, DocumentStatus.CANCELED, error="Indexing canceled")
            raise
        except Exception as exc:
            self.registry.replace_document_chunks(index["id"], document_id, [])
            self.registry.set_document_status(document_id, DocumentStatus.FAILED, error=str(exc))
            self.registry.set_index_error(index["id"], str(exc))
            raise

    def index_document_now(self, document_id: str, *, job_id: str | None = None) -> dict[str, Any]:
        result = self.index_document_chunks_now(document_id, job_id=job_id)
        self._job_message(
            job_id,
            f"Сохранено chunks: {result['chunks']}; перестройка векторного индекса",
        )
        try:
            result["rebuild"] = self._rebuild_index_debounced(result["index_id"], job_id=job_id)
            self.registry.set_document_status(result["document_id"], DocumentStatus.INDEXED)
        except JobCancelled:
            self.registry.set_document_status(
                result["document_id"],
                DocumentStatus.CANCELED,
                error="Indexing canceled",
            )
            raise
        except Exception as exc:
            self.registry.set_document_status(result["document_id"], DocumentStatus.FAILED, error=str(exc))
            raise
        return result

    def index_documents_now(
        self,
        index_id: str,
        document_ids: list[str],
        *,
        job_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        self._raise_if_cancelled(job_id)
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        index = self.registry.get_index(index_id)
        if not index:
            raise FileNotFoundError(f"Index not found: {index_id}")

        ids = [str(item).strip() for item in document_ids if str(item).strip()]
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise ValueError("No documents selected")

        indexed: list[dict[str, Any]] = []
        vectorizing_ids: list[str] = []
        failed: list[dict[str, Any]] = []
        changed_chunks = False
        total = len(ids)
        try:
            for pos, document_id in enumerate(ids, start=1):
                self._raise_if_cancelled(job_id)
                self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
                document = self.registry.get_document(document_id)
                if not document or document.get("deleted_at") or document.get("index_id") != index_id:
                    failed.append(
                        {
                            "document_id": document_id,
                            "filename": document_id,
                            "error": "Document not found",
                        }
                    )
                    continue

                label = document.get("filename") or document_id
                self._job_message(job_id, f"Документ {pos}/{total}: {label}")
                changed_chunks = True
                try:
                    result = self.index_document_chunks_now(document_id, job_id=job_id, deadline=deadline)
                    self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
                    indexed.append(result)
                    vectorizing_ids.append(document_id)
                except IndexingDeadlineExceeded:
                    raise
                except JobCancelled:
                    raise
                except Exception as exc:
                    failed.append(
                        {
                            "document_id": document_id,
                            "filename": label,
                            "error": str(exc),
                        }
                    )

            self._raise_if_cancelled(job_id)
            self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
            rebuild: dict[str, Any] | None = None
            if changed_chunks:
                self._job_message(
                    job_id,
                    f"Финальная перестройка векторного индекса: документов {len(indexed)}/{total}",
                )
                self._set_documents_status(vectorizing_ids, DocumentStatus.VECTORIZING)
                try:
                    rebuild = self.rebuild_index_now(index_id, job_id=job_id, deadline=deadline)
                except Exception as exc:
                    self._set_documents_status(vectorizing_ids, DocumentStatus.FAILED, error=str(exc))
                    raise
                self._set_documents_status(vectorizing_ids, DocumentStatus.INDEXED)
            return {
                "index_id": index_id,
                "documents": total,
                "indexed": indexed,
                "failed": failed,
                "rebuild": rebuild,
            }
        except IndexingDeadlineExceeded as exc:
            self._set_documents_status(vectorizing_ids, DocumentStatus.FAILED, error=str(exc))
            raise
        except JobCancelled:
            if changed_chunks:
                try:
                    self.rebuild_index_now(index_id)
                    self._set_documents_status(vectorizing_ids, DocumentStatus.INDEXED)
                except Exception:
                    pass
            raise

    def rebuild_index_now(
        self,
        index_id: str,
        *,
        embedding_model: str | None = None,
        job_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        self._raise_if_cancelled(job_id)
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        index = self.registry.get_index(index_id)
        if not index:
            raise FileNotFoundError(f"Index not found: {index_id}")
        model = embedding_model or index.get("embedding_model") or self.config.embedding_model
        if not model:
            raise ValueError("Embedding model is not configured")
        started = time.perf_counter()
        chunks = self.registry.active_chunks(index_id)
        timings["db_chunks_sec"] = time.perf_counter() - started
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]
        embedding_inputs = self._embedding_inputs(texts, query=False)
        text_hash_by_chunk = {
            chunk_id: self._embedding_text_hash(text)
            for chunk_id, text in zip(chunk_ids, embedding_inputs)
        }
        input_by_hash: dict[str, str] = {}
        chunk_ids_by_hash: dict[str, list[str]] = {}
        for chunk_id, text in zip(chunk_ids, embedding_inputs):
            text_hash = text_hash_by_chunk[chunk_id]
            input_by_hash.setdefault(text_hash, text)
            chunk_ids_by_hash.setdefault(text_hash, []).append(chunk_id)
        cache_key = self._embedding_cache_key(model, query=False)
        started = time.perf_counter()
        embeddings_by_id = self.registry.get_chunk_embeddings(cache_key, chunk_ids)
        timings["embedding_cache_read_sec"] = time.perf_counter() - started
        seeded_text_embeddings = {
            text_hash_by_chunk[chunk_id]: vector
            for chunk_id, vector in embeddings_by_id.items()
            if chunk_id in text_hash_by_chunk
        }
        if seeded_text_embeddings:
            self.registry.save_text_embeddings(
                cache_key,
                seeded_text_embeddings.items(),
                dtype=self._embedding_cache_dtype(),
            )
        missing_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in embeddings_by_id]
        missing_hashes = list(dict.fromkeys(text_hash_by_chunk[chunk_id] for chunk_id in missing_chunk_ids))
        text_embeddings = self.registry.get_text_embeddings(cache_key, missing_hashes)
        text_embeddings.update(seeded_text_embeddings)
        chunk_cache_rows: list[tuple[str, list[float]]] = []
        for text_hash in missing_hashes:
            vector = text_embeddings.get(text_hash)
            if vector is None:
                continue
            for chunk_id in chunk_ids_by_hash.get(text_hash, []):
                if chunk_id not in embeddings_by_id:
                    embeddings_by_id[chunk_id] = vector
                    chunk_cache_rows.append((chunk_id, vector))
        if chunk_cache_rows:
            self.registry.save_chunk_embeddings(
                cache_key,
                chunk_cache_rows,
                dtype=self._embedding_cache_dtype(),
            )
        missing = [
            (text_hash, input_by_hash[text_hash])
            for text_hash in missing_hashes
            if text_hash not in text_embeddings
        ]
        new_embeddings_count = len(missing)
        batch_size = max(1, int(self.config.embedding_batch_size or 16))
        embed_started = time.perf_counter()
        for batch_hashes, batch_vectors in self._embed_missing_batches(
            model=model,
            missing=missing,
            batch_size=batch_size,
            cached_count=len(chunk_ids) - new_embeddings_count,
            job_id=job_id,
            deadline=deadline,
            prepared_inputs=True,
        ):
            self.registry.save_text_embeddings(
                cache_key,
                zip(batch_hashes, batch_vectors),
                dtype=self._embedding_cache_dtype(),
            )
            chunk_rows: list[tuple[str, list[float]]] = []
            for text_hash, vector in zip(batch_hashes, batch_vectors):
                for chunk_id in chunk_ids_by_hash.get(text_hash, []):
                    embeddings_by_id[chunk_id] = vector
                    chunk_rows.append((chunk_id, vector))
            self.registry.save_chunk_embeddings(
                cache_key,
                chunk_rows,
                dtype=self._embedding_cache_dtype(),
            )
        timings["embedding_sec"] = time.perf_counter() - embed_started
        embeddings = [embeddings_by_id[chunk_id] for chunk_id in chunk_ids]
        self._raise_if_cancelled(job_id)
        self._raise_if_deadline_expired(deadline, "Chat attachment indexing timed out")
        self._job_message(job_id, "Запись FAISS индекса")
        started = time.perf_counter()
        dimension = build_index(
            self.config.indexes_path,
            index_id,
            chunk_ids,
            embeddings,
            index_type=index.get("index_type") or self.config.index_type,
            hnsw_threshold_chunks=self.config.hnsw_threshold_chunks,
            hnsw_m=self.config.hnsw_m,
            hnsw_ef_construction=self.config.hnsw_ef_construction,
            hnsw_ef_search=self.config.hnsw_ef_search,
        )
        timings["faiss_write_sec"] = time.perf_counter() - started
        if dimension:
            self.registry.update_index_embedding(index_id, model, dimension)
        timings["total_sec"] = time.perf_counter() - total_started
        self._record_metric(
            "rebuild_index",
            timings["total_sec"],
            {
                "index_id": index_id,
                "chunks": len(chunk_ids),
                "cached_embeddings": len(chunk_ids) - new_embeddings_count,
                "new_embeddings": new_embeddings_count,
            },
        )
        self._job_message(
            job_id,
            f"Векторный индекс готов ({len(chunk_ids)} chunks, dim={dimension}, "
            f"cached={len(chunk_ids) - new_embeddings_count}, new={new_embeddings_count}, "
            f"{timings['total_sec']:.2f}s)",
        )
        return {
            "index_id": index_id,
            "chunks": len(chunk_ids),
            "embedding_dim": dimension,
            "cached_embeddings": len(chunk_ids) - new_embeddings_count,
            "new_embeddings": new_embeddings_count,
            "timings": {key: round(value, 6) for key, value in timings.items()},
        }

    def rebuild_index_documents_now(
        self,
        index_id: str,
        document_ids: list[str] | None = None,
        *,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        index = self.registry.get_index(index_id)
        if not index:
            raise FileNotFoundError(f"Index not found: {index_id}")
        selected = [str(item).strip() for item in (document_ids or []) if str(item).strip()]
        selected = list(dict.fromkeys(selected))
        if selected:
            documents = []
            for document_id in selected:
                document = self.registry.get_document(document_id)
                if document and not document.get("deleted_at") and document.get("index_id") == index_id:
                    documents.append(document)
        else:
            documents = self.registry.list_documents(index_id)
        vectorizing_ids = [
            document["id"]
            for document in documents
            if int(document.get("chunk_count") or 0) > 0
        ]
        if not vectorizing_ids:
            model = str(index.get("embedding_model") or self.config.embedding_model or "")
            dimension = build_index(
                self.config.indexes_path,
                index_id,
                [],
                [],
                index_type=index.get("index_type") or self.config.index_type,
                hnsw_threshold_chunks=self.config.hnsw_threshold_chunks,
                hnsw_m=self.config.hnsw_m,
                hnsw_ef_construction=self.config.hnsw_ef_construction,
                hnsw_ef_search=self.config.hnsw_ef_search,
            )
            self.registry.update_index_embedding(index_id, model, dimension)
            rebuild = {
                "index_id": index_id,
                "chunks": 0,
                "embedding_dim": dimension,
                "cached_embeddings": 0,
                "new_embeddings": 0,
                "timings": {},
            }
            return {
                "index_id": index_id,
                "documents": 0,
                "chunks": 0,
                "embedding_dim": dimension,
                "rebuild": rebuild,
            }

        self._set_documents_status(vectorizing_ids, DocumentStatus.VECTORIZING)
        try:
            rebuild = self.rebuild_index_now(index_id, job_id=job_id)
        except Exception as exc:
            self._set_documents_status(vectorizing_ids, DocumentStatus.FAILED, error=str(exc))
            raise
        self._set_documents_status(vectorizing_ids, DocumentStatus.INDEXED)
        return {
            "index_id": index_id,
            "documents": len(vectorizing_ids),
            "rebuild": rebuild,
        }

    def delete_document(self, document_id: str) -> dict[str, Any]:
        deleted = self.registry.soft_delete_document(document_id)
        if not deleted:
            raise FileNotFoundError(f"Document not found: {document_id}")
        try:
            self.rebuild_index_now(deleted["index_id"])
        except ValueError:
            build_index(self.config.indexes_path, deleted["index_id"], [], [])
        return deleted

    def delete_documents(self, index_id: str, document_ids: list[str]) -> dict[str, Any]:
        ids = [str(item).strip() for item in dict.fromkeys(document_ids) if str(item).strip()]
        if not ids:
            raise ValueError("No documents selected")
        deleted = self.registry.soft_delete_documents(index_id, ids)
        if not deleted:
            raise FileNotFoundError("No selected documents found")
        try:
            rebuild = self.rebuild_index_now(index_id)
        except ValueError:
            build_index(self.config.indexes_path, index_id, [], [])
            rebuild = {"index_id": index_id, "chunks": 0, "embedding_dim": 0}
        return {
            "index_id": index_id,
            "requested_count": len(ids),
            "deleted_count": len(deleted),
            "deleted": deleted,
            "rebuild": rebuild,
        }

    def delete_index(self, index_id: str) -> dict[str, Any]:
        with acquire_index_lock(self.config.indexes_path, index_id):
            self.registry.request_cancel_jobs(index_id=index_id)
            deleted = self.registry.delete_index(index_id)
            if not deleted:
                raise FileNotFoundError(f"Index not found: {index_id}")
            shutil.rmtree(self.config.indexes_path / index_id, ignore_errors=True)
            shutil.rmtree(self.config.uploads_path / index_id, ignore_errors=True)
            invalidate_index_cache(self.config.indexes_path, index_id)
            return deleted

    def _mmr_select(self, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(candidates) <= limit:
            return candidates[:limit]
        ids_by_cache_key: dict[str, list[str]] = {}
        for item in candidates:
            cache_key = str(item.get("_embedding_cache_key") or "")
            chunk_id = str(item.get("chunk_id") or "")
            if cache_key and chunk_id:
                ids_by_cache_key.setdefault(cache_key, []).append(chunk_id)
        if not ids_by_cache_key:
            return candidates[:limit]

        embeddings: dict[str, list[float]] = {}
        for cache_key, chunk_ids in ids_by_cache_key.items():
            for chunk_id, vector in self.registry.get_chunk_embeddings(cache_key, chunk_ids).items():
                normalized = _normalize_embedding(vector)
                if normalized:
                    embeddings[chunk_id] = normalized
        if not embeddings:
            return candidates[:limit]

        remaining = list(candidates)
        selected: list[dict[str, Any]] = []
        selected_vectors: list[list[float]] = []
        while remaining and len(selected) < limit:
            if not selected:
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: float(remaining[index].get("score") or 0.0),
                )
            else:
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: self._mmr_score(remaining[index], embeddings, selected_vectors),
                )
            item = remaining.pop(best_index)
            selected.append(item)
            selected_vector = embeddings.get(str(item.get("chunk_id") or ""))
            if selected_vector:
                selected_vectors.append(selected_vector)
        return selected

    def _mmr_score(
        self,
        item: dict[str, Any],
        embeddings: dict[str, list[float]],
        selected_vectors: list[list[float]],
    ) -> tuple[float, float]:
        relevance = float(item.get("score") or 0.0)
        vector = embeddings.get(str(item.get("chunk_id") or ""))
        if not vector or not selected_vectors:
            diversity_penalty = 0.0
        else:
            diversity_penalty = max(_cosine_similarity(vector, selected) for selected in selected_vectors)
        mmr = (MMR_LAMBDA * relevance) - ((1.0 - MMR_LAMBDA) * diversity_penalty)
        return mmr, relevance

    def _expand_query_variants(self, query: str, variants: list[str]) -> tuple[list[str], dict[str, Any]]:
        stats: dict[str, Any] = {
            "query_expansion_applied": False,
            "query_expansion_variants": 0,
        }
        if not bool(getattr(self.config, "query_expansion_enabled", False)):
            return variants, stats
        model = str(getattr(self.config, "query_expansion_model", "") or "").strip()
        if not model:
            stats["query_expansion_error"] = "query_expansion_model is not configured"
            return variants, stats

        max_variants = int(getattr(self.config, "query_expansion_max_variants", 3) or 3)
        max_tokens = int(getattr(self.config, "query_expansion_max_tokens", 256) or 256)
        messages = [
            {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Запрос: {query}\n"
                    f"Верни до {max_variants} поисковых вариантов для semantic/BM25 retrieval."
                ),
            },
        ]
        try:
            raw = self.generation_client.chat(
                model,
                messages,
                temperature=0.1,
                num_predict=max_tokens,
            )
        except Exception as exc:
            stats["query_expansion_error"] = str(exc)
            return variants, stats

        expanded = llm_query_expansion_variants(raw, max_variants=max_variants)
        combined = list(variants)
        seen = {item.lower() for item in combined}
        for item in expanded:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(item)
        added = len(combined) - len(variants)
        stats.update(
            {
                "query_expansion_applied": added > 0,
                "query_expansion_variants": added,
                "query_expansion_model": model,
            }
        )
        return combined, stats

    def _rerank_candidates(self, query: str, candidates: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        min_results = int(getattr(self.config, "rerank_min_results", 10) or 10)
        stats: dict[str, Any] = {
            "rerank_applied": False,
            "rerank_candidates": 0,
        }
        if len(candidates) <= min_results:
            return candidates, stats
        if not bool(getattr(self.config, "rerank_enabled", False)):
            return candidates, stats
        model = str(getattr(self.config, "rerank_model", "") or "").strip()
        if not model or self.rerank_client is None:
            return candidates, stats

        window_size = max(limit, int(getattr(self.config, "rerank_top_n", 50) or 50))
        window_size = max(limit, min(len(candidates), window_size))
        window = candidates[:window_size]
        documents = [str(item.get("text") or "") for item in window]
        try:
            scores = self.rerank_client.rerank(model, query, documents)
        except Exception as exc:
            stats["rerank_error"] = str(exc)
            return candidates, stats
        if len(scores) != len(window):
            stats["rerank_error"] = "rerank score count mismatch"
            return candidates, stats

        reranked = []
        for item, score in zip(window, scores):
            updated = dict(item)
            try:
                rerank_score = float(score)
            except (TypeError, ValueError):
                rerank_score = 0.0
            rerank_score = max(0.0, min(1.0, rerank_score))
            updated["retrieval_score"] = float(updated.get("score") or 0.0)
            updated["rerank_score"] = rerank_score
            updated["score"] = rerank_score
            reranked.append(updated)
        reranked.sort(
            key=lambda item: (
                float(item.get("rerank_score") or 0.0),
                float(item.get("hybrid_score") or 0.0),
                float(item.get("retrieval_score") or 0.0),
            ),
            reverse=True,
        )
        stats.update(
            {
                "rerank_applied": True,
                "rerank_candidates": len(reranked),
                "rerank_model": model,
            }
        )
        return [*reranked, *candidates[window_size:]], stats

    def retrieve(
        self,
        query: str,
        *,
        index_ids: list[str] | None = None,
        extra_index_ids: list[str] | None = None,
        embedding_model: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        if not query.strip():
            return {"query": query, "results": []}
        started = time.perf_counter()
        indexes_by_id: dict[str, dict[str, Any]] = {}
        if index_ids:
            selected_ids = index_ids
        elif self.config.default_index_ids:
            selected_ids = self.config.default_index_ids
        else:
            indexes = self.registry.list_indexes()
            indexes_by_id = {item["id"]: item for item in indexes}
            selected_ids = [item["id"] for item in indexes]
        selected_ids = [
            str(item).strip()
            for item in dict.fromkeys([*(selected_ids or []), *((extra_index_ids or []))])
            if str(item).strip()
        ]
        timings["index_select_sec"] = time.perf_counter() - started
        limit = max(1, int(top_k or self.config.top_k))
        threshold = float(score_threshold if score_threshold is not None else self.config.score_threshold)
        search_limit = max(limit, min(limit * 4, limit + 120))
        merged: list[dict[str, Any]] = []
        query_variants = retrieval_query_variants(query, self.config.query_synonyms)
        query_variants, query_expansion_stats = self._expand_query_variants(query, query_variants)
        query_embeddings: dict[str, list[list[float]]] = {}
        dense_hit_count = 0
        fts_hit_count = 0

        for index_id in selected_ids:
            started = time.perf_counter()
            index = indexes_by_id.get(index_id) or self.registry.get_index(index_id)
            timings["index_meta_sec"] = timings.get("index_meta_sec", 0.0) + (time.perf_counter() - started)
            if not index:
                continue
            model = embedding_model or index.get("embedding_model") or self.config.embedding_model
            if not model:
                continue
            configured_dim = int(index.get("embedding_dim") or 0)
            query_cache_key = self._embedding_cache_key(model, query=True)
            if query_cache_key not in query_embeddings:
                started = time.perf_counter()
                query_embeddings[query_cache_key] = self.embedding_client.embed(
                    model,
                    self._embedding_inputs(query_variants or [query], query=True),
                )
                timings["query_embedding_sec"] = timings.get("query_embedding_sec", 0.0) + (
                    time.perf_counter() - started
                )
            hits_by_id: dict[str, tuple[float, str]] = {}
            for variant, query_embedding in zip(query_variants or [query], query_embeddings[query_cache_key]):
                if configured_dim and len(query_embedding) != configured_dim:
                    raise ValueError(
                        f"Index {index_id} uses embedding_dim={configured_dim}, but {model} returned {len(query_embedding)}"
                    )
                started = time.perf_counter()
                hits = search_index(self.config.indexes_path, index_id, query_embedding, search_limit)
                timings["faiss_search_sec"] = timings.get("faiss_search_sec", 0.0) + (
                    time.perf_counter() - started
                )
                for chunk_id, score in hits:
                    current = hits_by_id.get(chunk_id)
                    if current is None or score > current[0]:
                        hits_by_id[chunk_id] = (score, variant)
            sorted_dense_hits = sorted(hits_by_id.items(), key=lambda item: item[1][0], reverse=True)[:search_limit]
            dense_hit_count += len(sorted_dense_hits)
            started = time.perf_counter()
            fts_hits_by_id: dict[str, tuple[int, float, str]] = {}
            for variant in query_variants or [query]:
                for rank, (chunk_id, bm25_rank) in enumerate(
                    self.registry.search_chunks_fts(index_id, variant, limit=search_limit),
                    start=1,
                ):
                    fts_current = fts_hits_by_id.get(chunk_id)
                    if fts_current is None or rank < fts_current[0]:
                        fts_hits_by_id[chunk_id] = (rank, bm25_rank, variant)
            fts_hit_count += len(fts_hits_by_id)
            timings["fts_search_sec"] = timings.get("fts_search_sec", 0.0) + (time.perf_counter() - started)
            fusion: dict[str, dict[str, Any]] = {}
            for rank, (chunk_id, (score, variant)) in enumerate(sorted_dense_hits, start=1):
                item = fusion.setdefault(
                    chunk_id,
                    {
                        "rrf": 0.0,
                        "dense_score": 0.0,
                        "query_variant": variant,
                        "sources": set(),
                    },
                )
                item["rrf"] = float(item["rrf"]) + _rrf_score(rank)
                item["dense_score"] = max(float(item.get("dense_score") or 0.0), float(score))
                item["query_variant"] = variant
                item["dense_rank"] = rank
                item["sources"].add("dense")
            for chunk_id, (rank, bm25_rank, variant) in sorted(fts_hits_by_id.items(), key=lambda item: item[1][0]):
                item = fusion.setdefault(
                    chunk_id,
                    {
                        "rrf": 0.0,
                        "dense_score": 0.0,
                        "query_variant": variant,
                        "sources": set(),
                    },
                )
                item["rrf"] = float(item["rrf"]) + _rrf_score(rank)
                item["fts_rank"] = rank
                item["bm25_rank"] = bm25_rank
                item["query_variant"] = variant
                item["sources"].add("bm25")
            sorted_hits = sorted(
                fusion.items(),
                key=lambda item: (
                    float(item[1].get("rrf") or 0.0),
                    float(item[1].get("dense_score") or 0.0),
                    -int(item[1].get("fts_rank") or 999999),
                ),
                reverse=True,
            )[:search_limit]
            started = time.perf_counter()
            chunks = self.registry.chunks_by_ids([chunk_id for chunk_id, _ in sorted_hits])
            timings["db_chunks_sec"] = timings.get("db_chunks_sec", 0.0) + (time.perf_counter() - started)
            fused = {chunk_id: info for chunk_id, info in sorted_hits}
            for chunk in chunks:
                info = fused.get(chunk["id"], {})
                dense_score = float(info.get("dense_score") or 0.0)
                lexical_score = 0.0
                if "bm25" in info.get("sources", set()):
                    lexical_score = _lexical_score_from_fts_rank(int(info.get("fts_rank") or 1))
                score = max(dense_score, lexical_score)
                if score < threshold:
                    continue
                merged.append(
                    {
                        "index_id": index_id,
                        "index_name": index.get("name") or index_id,
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["id"],
                        "chunk_no": chunk["chunk_no"],
                        "score": score,
                        "hybrid_score": float(info.get("rrf") or 0.0),
                        "retrieval_sources": sorted(info.get("sources", set())),
                        "query_variant": info.get("query_variant") or query,
                        "text": chunk["text"],
                        "locator": chunk_locator(chunk["text"], chunk.get("chunk_no")),
                        "quote": chunk_quote(chunk["text"]),
                        "source": chunk.get("filename") or chunk.get("source_path") or "unknown",
                        "source_path": chunk.get("source_path") or chunk.get("stored_path") or "",
                        "_embedding_cache_key": self._embedding_cache_key(model, query=False),
                    }
                )

        merged.sort(key=lambda item: (float(item.get("hybrid_score") or 0.0), item["score"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in merged:
            key = _chunk_dedupe_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        reranked, rerank_stats = self._rerank_candidates(query, deduped, limit)
        selected = self._mmr_select(reranked, limit)
        results = []
        for item in selected:
            result = dict(item)
            result.pop("_embedding_cache_key", None)
            results.append(result)
        timings["total_sec"] = time.perf_counter() - total_started
        self._record_metric(
            "retrieve",
            timings["total_sec"],
            {
                "indexes": len(selected_ids),
                "raw": len(merged),
                "deduped": len(deduped),
                "mmr_selected": len(selected),
                "dense_hits": dense_hit_count,
                "fts_hits": fts_hit_count,
                "rerank_applied": bool(rerank_stats.get("rerank_applied")),
                "query_expansion_applied": bool(query_expansion_stats.get("query_expansion_applied")),
                "top_k": limit,
                "query_variants": len(query_variants),
            },
        )
        return {
            "query": query,
            "results": results,
            "stats": {
                "timings": {key: round(value, 6) for key, value in timings.items()},
                "raw": len(merged),
                "deduped": len(deduped),
                "returned": len(results),
                "mmr_selected": len(selected),
                "dense_hits": dense_hit_count,
                "fts_hits": fts_hit_count,
                **query_expansion_stats,
                **rerank_stats,
                "query_embedding_calls": len(query_embeddings),
                "query_embedding_variants": len(query_variants),
            },
        }

    def run_job(self, job_id: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        job = self.registry.get_job(job_id)
        job_kind = str((job or {}).get("kind") or "")
        if job and job.get("status") == JobStatus.CANCEL_REQUESTED.value:
            if job.get("document_id"):
                self.registry.set_document_status(job["document_id"], DocumentStatus.CANCELED, error="Indexing canceled")
            self.registry.update_job(job_id, JobStatus.CANCELED, message="Job canceled", finished=True)
            return
        self.registry.update_job(job_id, JobStatus.RUNNING, message="Job started")
        try:
            kwargs.setdefault("job_id", job_id)
            result = func(*args, **kwargs)
            ocr_cleanup = self._auto_clear_ocr_gpu_cache(job_id, job_kind)
            if ocr_cleanup:
                result = dict(result or {})
                result["ocr_gpu_cache"] = ocr_cleanup
                if ocr_cleanup.get("skipped"):
                    message = "Job completed; OCR GPU cache cleanup skipped because another indexing job is active"
                else:
                    message = (
                        "Job completed; OCR GPU cache cleared "
                        f"({ocr_cleanup.get('freed_reserved_mb', 0)} MB reserved)"
                    )
            else:
                message = "Job completed"
            self.registry.update_job(
                job_id,
                JobStatus.COMPLETED,
                message=message,
                result_json=json.dumps(result, ensure_ascii=False),
                finished=True,
            )
        except JobCancelled as exc:
            ocr_cleanup = self._auto_clear_ocr_gpu_cache(job_id, job_kind)
            message = "Job canceled"
            if ocr_cleanup and not ocr_cleanup.get("skipped"):
                message += "; OCR GPU cache cleared"
            self.registry.update_job(
                job_id,
                JobStatus.CANCELED,
                message=message,
                error=str(exc),
                finished=True,
            )
        except Exception as exc:
            ocr_cleanup = self._auto_clear_ocr_gpu_cache(job_id, job_kind)
            message = "Job failed"
            if ocr_cleanup and not ocr_cleanup.get("skipped"):
                message += "; OCR GPU cache cleared"
            self.registry.update_job(
                job_id,
                JobStatus.FAILED,
                message=message,
                error=str(exc),
                finished=True,
            )

    def remove_storage_file(self, document: dict[str, Any]) -> None:
        stored_path = document.get("stored_path") or ""
        if stored_path:
            try:
                Path(stored_path).unlink(missing_ok=True)
            except Exception:
                pass
