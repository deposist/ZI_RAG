from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Literal

SCHEMA_MIGRATIONS = [
    (1, "initial_schema"),
    (2, "documents_external_metadata"),
    (3, "embedding_text_cache"),
    (4, "chunk_fts"),
    (5, "document_fts"),
]


def utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned[:80] or uuid.uuid4().hex


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class DocumentStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    VECTORIZING = "vectorizing"
    INDEXED = "indexed"
    FAILED = "failed"
    CANCELED = "canceled"
    DELETED = "deleted"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


StatusValue = str | DocumentStatus | JobStatus

ACTIVE_JOB_STATUSES = (
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.CANCEL_REQUESTED,
)


def _status_value(status: StatusValue) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status or "")


def _status_values(statuses: Iterable[StatusValue]) -> list[str]:
    values = []
    for status in statuses:
        value = _status_value(status)
        if value:
            values.append(value)
    return values


def _fts_query(value: str) -> str:
    tokens = []
    for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{2,}", str(value or "").lower()):
        token = token.replace("ё", "е")
        if token and token not in tokens:
            tokens.append(token)
    return " OR ".join(tokens[:32])


class _RegistryConnectionContext:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.registry._conn_lock.acquire()
        try:
            self.conn = self.registry._shared_connection()
            if self.registry._conn_depth == 0:
                self.registry._conn_rollback_required = False
            self.registry._conn_depth += 1
        except Exception:
            self.registry._conn_lock.release()
            raise
        return self.conn

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if self.conn is not None:
                if exc_type is not None:
                    self.registry._conn_rollback_required = True
                self.registry._conn_depth = max(0, self.registry._conn_depth - 1)
                if self.registry._conn_depth == 0:
                    try:
                        if self.registry._conn_rollback_required:
                            self.conn.rollback()
                        else:
                            self.conn.commit()
                    finally:
                        self.registry._conn_rollback_required = False
        finally:
            self.conn = None
            self.registry._conn_lock.release()
        return False


class Registry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.RLock()
        self._conn_depth = 0
        self._conn_rollback_required = False
        self.init()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _shared_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._open_connection()
        return self._conn

    def connect(self) -> _RegistryConnectionContext:
        return _RegistryConnectionContext(self)

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS indexes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL DEFAULT '',
                    embedding_dim INTEGER NOT NULL DEFAULT 0,
                    chunk_size INTEGER NOT NULL DEFAULT 1200,
                    chunk_overlap INTEGER NOT NULL DEFAULT 120,
                    index_type TEXT NOT NULL DEFAULT 'auto',
                    status TEXT NOT NULL DEFAULT 'ready',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    stored_path TEXT NOT NULL DEFAULT '',
                    mime_type TEXT NOT NULL DEFAULT '',
                    file_hash TEXT NOT NULL DEFAULT '',
                    external_id TEXT NOT NULL DEFAULT '',
                    external_source TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT NOT NULL DEFAULT '',
                    text_chars INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(index_id) REFERENCES indexes(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_no INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(index_id) REFERENCES indexes(id) ON DELETE CASCADE,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    index_id TEXT NOT NULL DEFAULT '',
                    document_id TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, model),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embedding_text_cache (
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model, text_hash)
                );

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    migration_id TEXT NOT NULL UNIQUE,
                    applied_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    index_id UNINDEXED,
                    document_id UNINDEXED,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                    document_id UNINDEXED,
                    index_id UNINDEXED,
                    filename,
                    source_path,
                    stored_path,
                    external_id,
                    error,
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE INDEX IF NOT EXISTS idx_documents_index_status
                    ON documents(index_id, status);
                CREATE INDEX IF NOT EXISTS idx_documents_index_deleted_created
                    ON documents(index_id, deleted_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_chunks_index_active
                    ON chunks(index_id, active);
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
                    ON chunk_embeddings(model);
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            migrations = {
                "external_id": "ALTER TABLE documents ADD COLUMN external_id TEXT NOT NULL DEFAULT ''",
                "external_source": "ALTER TABLE documents ADD COLUMN external_source TEXT NOT NULL DEFAULT ''",
                "metadata_json": "ALTER TABLE documents ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    conn.execute(statement)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_external_active
                    ON documents(index_id, external_source, external_id, deleted_at)
                """
            )
            now = utc_now()
            chunk_fts_migration_applied = False
            document_fts_migration_applied = False
            for version, migration_id in SCHEMA_MIGRATIONS:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO schema_version (version, migration_id, applied_at)
                    VALUES (?, ?, ?)
                    """,
                    (version, migration_id, now),
                )
                if version == 4 and cursor.rowcount == 1:
                    chunk_fts_migration_applied = True
                if version == 5 and cursor.rowcount == 1:
                    document_fts_migration_applied = True
            if chunk_fts_migration_applied or self._chunk_fts_empty_with_active_chunks(conn):
                self._rebuild_chunk_fts(conn)
            if document_fts_migration_applied or self._document_fts_empty_with_active_documents(conn):
                self._rebuild_document_fts(conn)

    def _chunk_fts_empty_with_active_chunks(self, conn: sqlite3.Connection) -> bool:
        fts_row = conn.execute("SELECT COUNT(*) AS count FROM chunk_fts").fetchone()
        fts_count = int(fts_row["count"] or 0) if fts_row is not None else 0
        if fts_count:
            return False
        active_row = conn.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.active = 1 AND d.deleted_at IS NULL
                LIMIT 1
            ) AS has_chunks
            """,
        ).fetchone()
        return bool(active_row is not None and active_row["has_chunks"])

    def _rebuild_chunk_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM chunk_fts")
        conn.execute(
            """
            INSERT INTO chunk_fts (chunk_id, index_id, document_id, text)
            SELECT c.id, c.index_id, c.document_id, c.text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.active = 1 AND d.deleted_at IS NULL
            """
        )

    def _document_fts_empty_with_active_documents(self, conn: sqlite3.Connection) -> bool:
        fts_row = conn.execute("SELECT COUNT(*) AS count FROM document_fts").fetchone()
        fts_count = int(fts_row["count"] or 0) if fts_row is not None else 0
        if fts_count:
            return False
        active_row = conn.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM documents
                WHERE deleted_at IS NULL
                LIMIT 1
            ) AS has_documents
            """,
        ).fetchone()
        return bool(active_row is not None and active_row["has_documents"])

    def _sync_document_fts(self, conn: sqlite3.Connection, document_id: str) -> None:
        conn.execute("DELETE FROM document_fts WHERE document_id = ?", (document_id,))
        row = conn.execute(
            """
            SELECT id, index_id, filename, source_path, stored_path, external_id, error
            FROM documents
            WHERE id = ? AND deleted_at IS NULL
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """
            INSERT INTO document_fts
                (document_id, index_id, filename, source_path, stored_path, external_id, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["index_id"],
                row["filename"],
                row["source_path"],
                row["stored_path"],
                row["external_id"],
                row["error"],
            ),
        )

    def _rebuild_document_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM document_fts")
        conn.execute(
            """
            INSERT INTO document_fts
                (document_id, index_id, filename, source_path, stored_path, external_id, error)
            SELECT id, index_id, filename, source_path, stored_path, external_id, error
            FROM documents
            WHERE deleted_at IS NULL
            """
        )

    def schema_versions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT version, migration_id, applied_at
                FROM schema_version
                ORDER BY version
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_index(
        self,
        name: str,
        *,
        index_id: str | None = None,
        description: str = "",
        embedding_model: str = "",
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
        index_type: str = "auto",
    ) -> dict[str, Any]:
        safe_id = sanitize_id(index_id or name)
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO indexes
                    (id, name, description, embedding_model, chunk_size, chunk_overlap,
                     index_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_id,
                    name.strip() or safe_id,
                    description or "",
                    embedding_model or "",
                    int(chunk_size),
                    int(chunk_overlap),
                    index_type or "auto",
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM indexes WHERE id = ?", (safe_id,)).fetchone()
        return row_to_dict(row) or {}

    def list_indexes(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT i.*,
                    COALESCE(d.document_count, 0) AS document_count,
                    COALESCE(c.chunk_count, 0) AS chunk_count
                FROM indexes i
                LEFT JOIN (
                    SELECT index_id, COUNT(*) AS document_count
                    FROM documents
                    WHERE deleted_at IS NULL
                    GROUP BY index_id
                ) d ON d.index_id = i.id
                LEFT JOIN (
                    SELECT index_id, COUNT(*) AS chunk_count
                    FROM chunks
                    WHERE active = 1
                    GROUP BY index_id
                ) c ON c.index_id = i.id
                ORDER BY i.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_index(self, index_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM indexes WHERE id = ?", (index_id,)).fetchone()
        return row_to_dict(row)

    def delete_index(self, index_id: str) -> dict[str, Any] | None:
        now = utc_now()
        active_statuses = _status_values(ACTIVE_JOB_STATUSES)
        active_placeholders = ",".join("?" for _ in active_statuses)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM indexes WHERE id = ?", (index_id,)).fetchone()
            if row is None:
                return None
            conn.execute(
                f"""
                UPDATE jobs
                SET status = ?,
                    message = 'Index deleted',
                    updated_at = ?
                WHERE index_id = ? AND status IN ({active_placeholders})
                """,
                [JobStatus.CANCEL_REQUESTED.value, now, index_id, *active_statuses],
            )
            conn.execute(
                f"DELETE FROM jobs WHERE index_id = ? AND status NOT IN ({active_placeholders})",
                [index_id, *active_statuses],
            )
            conn.execute("DELETE FROM document_fts WHERE index_id = ?", (index_id,))
            conn.execute("DELETE FROM chunk_fts WHERE index_id = ?", (index_id,))
            conn.execute("DELETE FROM indexes WHERE id = ?", (index_id,))
        return dict(row)

    def update_index_embedding(self, index_id: str, model: str, dimension: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE indexes
                SET embedding_model = ?, embedding_dim = ?, updated_at = ?, status = 'ready', error = ''
                WHERE id = ?
                """,
                (model, int(dimension), utc_now(), index_id),
            )

    def set_index_error(self, index_id: str, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE indexes SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (error, utc_now(), index_id),
            )

    def create_document(
        self,
        index_id: str,
        *,
        filename: str,
        source_path: str = "",
        stored_path: str = "",
        mime_type: str = "",
        file_hash: str = "",
        external_id: str = "",
        external_source: str = "",
        metadata_json: str = "{}",
    ) -> dict[str, Any]:
        doc_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, index_id, filename, source_path, stored_path, mime_type,
                     file_hash, external_id, external_source, metadata_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    index_id,
                    filename,
                    source_path,
                    stored_path,
                    mime_type,
                    file_hash,
                    external_id or "",
                    external_source or "",
                    metadata_json or "{}",
                    now,
                    now,
                ),
            )
            self._sync_document_fts(conn, doc_id)
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row_to_dict(row) or {}

    def list_documents(self, index_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE index_id = ? AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                (index_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_documents_page(
        self,
        index_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        query: str = "",
        status: str | DocumentStatus = "",
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit or 200), 500))
        offset = max(0, int(offset or 0))
        where = ["index_id = ?", "deleted_at IS NULL"]
        values: list[Any] = [index_id]
        status_value = _status_value(status).strip().lower()
        if status_value:
            if status_value == "unindexed":
                where.append("LOWER(status) != ?")
                values.append(DocumentStatus.INDEXED.value)
            else:
                where.append("LOWER(status) = ?")
                values.append(status_value)
        query_value = str(query or "").strip().lower()
        if query_value:
            fts_query = _fts_query(query_value)
            if fts_query:
                where.append(
                    """
                    id IN (
                        SELECT document_id
                        FROM document_fts
                        WHERE document_fts MATCH ?
                          AND index_id = ?
                    )
                    """
                )
                values.extend([fts_query, index_id])
            else:
                where.append(
                    """
                    (
                        LOWER(filename) LIKE ?
                        OR LOWER(source_path) LIKE ?
                        OR LOWER(stored_path) LIKE ?
                        OR LOWER(external_id) LIKE ?
                        OR LOWER(error) LIKE ?
                    )
                    """
                )
                pattern = f"%{query_value}%"
                values.extend([pattern, pattern, pattern, pattern, pattern])
        where_sql = " AND ".join(where)
        with self.connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM documents WHERE {where_sql}",
                    values,
                ).fetchone()[0]
                or 0
            )
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*values, limit, offset],
            ).fetchall()
            status_rows = conn.execute(
                """
                SELECT LOWER(status) AS status, COUNT(*) AS count
                FROM documents
                WHERE index_id = ? AND deleted_at IS NULL
                GROUP BY LOWER(status)
                """,
                (index_id,),
            ).fetchall()
        status_counts = {str(row["status"] or "unknown"): int(row["count"] or 0) for row in status_rows}
        return {
            "documents": [dict(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "status": status,
            "status_counts": status_counts,
        }

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row_to_dict(row)

    def get_active_document_by_external(
        self,
        index_id: str,
        external_source: str,
        external_id: str,
    ) -> dict[str, Any] | None:
        if not external_id:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE index_id = ?
                  AND external_source = ?
                  AND external_id = ?
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (index_id, external_source or "", external_id),
            ).fetchone()
        return row_to_dict(row)

    def set_document_status(
        self,
        document_id: str,
        status: str | DocumentStatus,
        *,
        error: str = "",
        text_chars: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        fields = ["status = ?", "error = ?", "updated_at = ?"]
        values: list[Any] = [_status_value(status), error, utc_now()]
        if text_chars is not None:
            fields.append("text_chars = ?")
            values.append(int(text_chars))
        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(int(chunk_count))
        values.append(document_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            self._sync_document_fts(conn, document_id)

    def replace_document_chunks(
        self,
        index_id: str,
        document_id: str,
        chunks: Iterable[str],
    ) -> list[str]:
        now = utc_now()
        texts = list(chunks)
        with self.connect() as conn:
            existing_rows = conn.execute(
                """
                SELECT id, chunk_no, text
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_no
                """,
                (document_id,),
            ).fetchall()
            existing_by_no = {int(row["chunk_no"]): row for row in existing_rows}
            output_ids: list[str] = []
            stale_embedding_ids: list[str] = []

            for chunk_no, text in enumerate(texts):
                existing = existing_by_no.get(chunk_no)
                if existing is None:
                    chunk_id = uuid.uuid4().hex
                    conn.execute(
                        """
                        INSERT INTO chunks (id, index_id, document_id, chunk_no, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (chunk_id, index_id, document_id, chunk_no, text, now),
                    )
                else:
                    chunk_id = str(existing["id"])
                    if str(existing["text"]) != text:
                        stale_embedding_ids.append(chunk_id)
                    conn.execute(
                        """
                        UPDATE chunks
                        SET index_id = ?, chunk_no = ?, text = ?, active = 1
                        WHERE id = ?
                        """,
                        (index_id, chunk_no, text, chunk_id),
                    )
                conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    """
                    INSERT INTO chunk_fts (chunk_id, index_id, document_id, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk_id, index_id, document_id, text),
                )
                output_ids.append(chunk_id)

            removed_ids = [
                str(row["id"])
                for row in existing_rows
                if int(row["chunk_no"]) >= len(texts)
            ]
            stale_embedding_ids.extend(removed_ids)
            if stale_embedding_ids:
                placeholders = ",".join("?" for _ in stale_embedding_ids)
                conn.execute(
                    f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                    stale_embedding_ids,
                )
            if removed_ids:
                placeholders = ",".join("?" for _ in removed_ids)
                conn.execute(
                    f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                    removed_ids,
                )
                conn.execute(
                    f"DELETE FROM chunks WHERE id IN ({placeholders})",
                    removed_ids,
                )
        return output_ids

    def _pack_embedding(self, embedding: Iterable[float], *, dtype: str = "fp32") -> tuple[int, bytes]:
        values = [float(item) for item in embedding]
        if not values:
            return 0, b""
        normalized_dtype = str(dtype or "fp32").strip().lower()
        if normalized_dtype == "fp16":
            return len(values), struct.pack(f"<{len(values)}e", *values)
        return len(values), struct.pack(f"<{len(values)}f", *values)

    def _unpack_embedding(self, blob: bytes, dim: int) -> list[float]:
        if not blob or dim <= 0:
            return []
        expected_fp32 = int(dim) * 4
        if len(blob) == expected_fp32:
            return list(struct.unpack(f"<{int(dim)}f", blob))
        expected_fp16 = int(dim) * 2
        if len(blob) == expected_fp16:
            return [float(item) for item in struct.unpack(f"<{int(dim)}e", blob)]
        return []

    def get_chunk_embeddings(self, model: str, chunk_ids: Iterable[str]) -> dict[str, list[float]]:
        ids = [str(item).strip() for item in dict.fromkeys(chunk_ids) if str(item).strip()]
        if not ids or not model:
            return {}
        output: dict[str, list[float]] = {}
        with self.connect() as conn:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT chunk_id, dim, embedding_blob
                    FROM chunk_embeddings
                    WHERE model = ?
                      AND chunk_id IN ({placeholders})
                    """,
                    [model, *batch],
                ).fetchall()
                for row in rows:
                    embedding = self._unpack_embedding(row["embedding_blob"], int(row["dim"] or 0))
                    if embedding:
                        output[str(row["chunk_id"])] = embedding
        return output

    def get_text_embeddings(self, model: str, text_hashes: Iterable[str]) -> dict[str, list[float]]:
        hashes = [str(item).strip() for item in dict.fromkeys(text_hashes) if str(item).strip()]
        if not hashes or not model:
            return {}
        output: dict[str, list[float]] = {}
        with self.connect() as conn:
            for start in range(0, len(hashes), 500):
                batch = hashes[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT text_hash, dim, embedding_blob
                    FROM embedding_text_cache
                    WHERE model = ?
                      AND text_hash IN ({placeholders})
                    """,
                    [model, *batch],
                ).fetchall()
                for row in rows:
                    embedding = self._unpack_embedding(row["embedding_blob"], int(row["dim"] or 0))
                    if embedding:
                        output[str(row["text_hash"])] = embedding
        return output

    def save_text_embeddings(
        self,
        model: str,
        embeddings: Iterable[tuple[str, Iterable[float]]],
        *,
        dtype: str = "fp32",
    ) -> None:
        if not model:
            return
        now = utc_now()
        rows = []
        for text_hash, embedding in embeddings:
            dim, blob = self._pack_embedding(embedding, dtype=dtype)
            if text_hash and dim and blob:
                rows.append((str(text_hash), model, dim, blob, now))
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_text_cache
                    (text_hash, model, dim, embedding_blob, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_chunk_embeddings(
        self,
        model: str,
        embeddings: Iterable[tuple[str, Iterable[float]]],
        *,
        dtype: str = "fp32",
    ) -> None:
        if not model:
            return
        now = utc_now()
        rows = []
        for chunk_id, embedding in embeddings:
            dim, blob = self._pack_embedding(embedding, dtype=dtype)
            if chunk_id and dim and blob:
                rows.append((str(chunk_id), model, dim, blob, now))
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                    (chunk_id, model, dim, embedding_blob, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def soft_delete_document(self, document_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE documents
                SET deleted_at = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, DocumentStatus.DELETED.value, now, document_id),
            )
            conn.execute("UPDATE chunks SET active = 0 WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_fts WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunk_fts WHERE document_id = ?", (document_id,))
            conn.execute(
                """
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                """,
                (document_id,),
            )
        return dict(row)

    def soft_delete_documents(self, index_id: str, document_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = [str(item).strip() for item in dict.fromkeys(document_ids) if str(item).strip()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        now = utc_now()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE index_id = ?
                  AND deleted_at IS NULL
                  AND id IN ({placeholders})
                ORDER BY created_at DESC
                """,
                [index_id, *ids],
            ).fetchall()
            deleted = [dict(row) for row in rows]
            if not deleted:
                return []
            deleted_ids = [row["id"] for row in deleted]
            deleted_placeholders = ",".join("?" for _ in deleted_ids)
            conn.execute(
                f"""
                UPDATE documents
                SET deleted_at = ?, status = ?, updated_at = ?
                WHERE index_id = ?
                  AND id IN ({deleted_placeholders})
                """,
                [now, DocumentStatus.DELETED.value, now, index_id, *deleted_ids],
            )
            conn.execute(
                f"""
                DELETE FROM document_fts
                WHERE index_id = ?
                  AND document_id IN ({deleted_placeholders})
                """,
                [index_id, *deleted_ids],
            )
            conn.execute(
                f"""
                UPDATE chunks
                SET active = 0
                WHERE index_id = ?
                  AND document_id IN ({deleted_placeholders})
                """,
                [index_id, *deleted_ids],
            )
            conn.execute(
                f"""
                DELETE FROM chunk_fts
                WHERE index_id = ?
                  AND document_id IN ({deleted_placeholders})
                """,
                [index_id, *deleted_ids],
            )
            conn.execute(
                f"""
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (
                    SELECT id FROM chunks
                    WHERE index_id = ?
                      AND document_id IN ({deleted_placeholders})
                )
                """,
                [index_id, *deleted_ids],
            )
        return deleted

    def active_chunks(self, index_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, d.filename, d.source_path, d.stored_path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.index_id = ? AND c.active = 1 AND d.deleted_at IS NULL
                ORDER BY d.filename COLLATE NOCASE, c.chunk_no
                """,
                (index_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_chunks_fts(self, index_id: str, query: str, *, limit: int = 100) -> list[tuple[str, float]]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.chunk_id, bm25(chunk_fts) AS rank
                FROM chunk_fts f
                JOIN chunks c ON c.id = f.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE chunk_fts MATCH ?
                  AND f.index_id = ?
                  AND c.active = 1
                  AND d.deleted_at IS NULL
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, index_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [(str(row["chunk_id"]), float(row["rank"] or 0.0)) for row in rows]

    def chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(chunk_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.filename, d.source_path, d.stored_path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders}) AND c.active = 1 AND d.deleted_at IS NULL
                """,
                ids,
            ).fetchall()
        by_id = {row["id"]: dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def create_job(
        self,
        kind: str,
        *,
        index_id: str = "",
        document_id: str = "",
        result_json: str = "{}",
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                    (id, kind, status, index_id, document_id, result_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    JobStatus.QUEUED.value,
                    index_id,
                    document_id,
                    result_json or "{}",
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row_to_dict(row) or {}

    def update_job(
        self,
        job_id: str,
        status: str | JobStatus,
        *,
        message: str = "",
        result_json: str | None = None,
        error: str = "",
        finished: bool = False,
    ) -> None:
        fields = ["status = ?", "message = ?", "error = ?", "updated_at = ?"]
        values: list[Any] = [_status_value(status), message, error, utc_now()]
        if result_json is not None:
            fields.append("result_json = ?")
            values.append(result_json)
        if finished:
            fields.append("finished_at = ?")
            values.append(utc_now())
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row_to_dict(row)

    def list_jobs(
        self,
        *,
        index_id: str = "",
        statuses: Iterable[str | JobStatus] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = []
        values: list[Any] = []
        if index_id:
            where.append("index_id = ?")
            values.append(index_id)
        status_values = _status_values(statuses or [])
        if status_values:
            placeholders = ",".join("?" for _ in status_values)
            where.append(f"status IN ({placeholders})")
            values.extend(status_values)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        values.append(max(1, min(int(limit), 500)))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def active_jobs_for_documents(self, index_id: str, document_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = [str(item) for item in dict.fromkeys(document_ids) if str(item)]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        active_statuses = _status_values(ACTIVE_JOB_STATUSES)
        active_placeholders = ",".join("?" for _ in active_statuses)
        id_set = set(ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE index_id = ?
                  AND document_id IN ({placeholders})
                  AND status IN ({active_placeholders})
                ORDER BY created_at DESC
                """,
                [index_id, *ids, *active_statuses],
            ).fetchall()
            bulk_rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE index_id = ?
                  AND document_id = ''
                  AND kind IN ('index_documents', 'reindex_documents', 'force_reindex_documents')
                  AND status IN ({active_placeholders})
                ORDER BY created_at DESC
                """,
                [index_id, *active_statuses],
            ).fetchall()
        jobs = [dict(row) for row in rows]
        seen = {job["id"] for job in jobs}
        for row in bulk_rows:
            job = dict(row)
            try:
                payload = json.loads(job.get("result_json") or "{}")
            except Exception:
                payload = {}
            job_ids = {str(item) for item in payload.get("document_ids", []) if str(item)}
            if job["id"] not in seen and id_set.intersection(job_ids):
                jobs.append(job)
                seen.add(job["id"])
        return jobs

    def request_cancel_jobs(
        self,
        *,
        index_id: str = "",
        document_id: str = "",
        job_id: str = "",
    ) -> list[dict[str, Any]]:
        active_statuses = _status_values(ACTIVE_JOB_STATUSES)
        active_placeholders = ",".join("?" for _ in active_statuses)
        where = [f"status IN ({active_placeholders})"]
        values: list[Any] = [*active_statuses]
        if job_id:
            where.append("id = ?")
            values.append(job_id)
        if index_id:
            where.append("index_id = ?")
            values.append(index_id)
        if document_id:
            where.append("document_id = ?")
            values.append(document_id)
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                f"""
                UPDATE jobs
                SET status = ?, message = 'Cancel requested', updated_at = ?
                WHERE {' AND '.join(where)}
                """,
                [JobStatus.CANCEL_REQUESTED.value, now, *values],
            )
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def job_cancel_requested(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        job = self.get_job(job_id)
        cancel_statuses = {JobStatus.CANCEL_REQUESTED.value, JobStatus.CANCELED.value}
        return bool(job and job.get("status") in cancel_statuses)

    def cancel_stale_jobs(self, message: str = "Sidecar restarted") -> int:
        now = utc_now()
        active_statuses = _status_values(ACTIVE_JOB_STATUSES)
        active_placeholders = ",".join("?" for _ in active_statuses)
        with self.connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE jobs
                SET status = ?,
                    message = ?,
                    error = COALESCE(NULLIF(error, ''), ?),
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ({active_placeholders})
                """,
                [JobStatus.CANCELED.value, message, message, now, now, *active_statuses],
            )
            return int(cursor.rowcount or 0)


def ensure_inside_allowed_roots(path: str | Path, roots: list[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not roots:
        raise PermissionError("allowed_source_roots is empty; add paths through upload or configure allowlist")
    allowed = [Path(root).expanduser().resolve() for root in roots if str(root).strip()]
    for root in allowed:
        try:
            if os.path.commonpath([str(candidate), str(root)]) == str(root):
                return candidate
        except ValueError:
            continue
    raise PermissionError(f"Path is outside allowed_source_roots: {candidate}")
