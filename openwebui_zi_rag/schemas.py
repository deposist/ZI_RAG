"""Pydantic request/response models for ZI_RAG sidecar HTTP API.

All schemas are kept in one module so route packages can import them without
pulling in the rest of ``server.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexCreate(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    embedding_model: str = ""
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    index_type: str = ""


class AddPathRequest(BaseModel):
    path: str
    recursive: bool = True
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    index_now: bool = True


class ReindexDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    force: bool = False


class DeleteDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class RebuildIndexRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str
    index_ids: list[str] = Field(default_factory=list)
    extra_index_ids: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    top_k: int | None = None
    score_threshold: float | None = None


class AnalyzeRequest(BaseModel):
    query: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    index_ids: list[str] = Field(default_factory=list)
    extra_index_ids: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    generation_model: str = ""
    mode: str = "answer"
    top_k: int | None = None
    score_threshold: float | None = None
    batch_chars: int | None = None
    max_batches: int | None = None


class ComplianceRequest(BaseModel):
    query: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    nmd_index_ids: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    generation_model: str = ""
    top_k: int | None = None
    score_threshold: float | None = None
    section_chars: int | None = None
    max_sections: int | None = None


class ChatAttachmentFileMeta(BaseModel):
    id: str = ""
    name: str = ""
    content_type: str = ""


class ChatAttachmentsRequest(BaseModel):
    chat_id: str = ""
    session_id: str = ""
    message_id: str = ""
    user_id: str = ""
    scope_id: str = ""
    files: list[ChatAttachmentFileMeta] = Field(default_factory=list)


class ConfigUpdate(BaseModel):
    storage_dir: str | None = None
    ollama_base_url: str | None = None
    api_key: str | None = None
    require_api_key_localhost: bool | None = None
    allowed_source_roots: list[str] | None = None
    default_index_ids: list[str] | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_query_prefix: str | None = None
    embedding_document_prefix: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_batch_size: int | None = None
    embedding_cache_dtype: str | None = None
    rebuild_debounce_sec: float | None = None
    top_k: int | None = None
    score_threshold: float | None = None
    rag_enabled: bool | None = None
    include_sources: bool | None = None
    retrieval_top_k: int | None = None
    adaptive_score_margin: float | None = None
    max_prompt_chunks: int | None = None
    min_query_term_hits: int | None = None
    rerank_enabled: bool | None = None
    rerank_model: str | None = None
    rerank_min_results: int | None = None
    rerank_top_n: int | None = None
    query_expansion_enabled: bool | None = None
    query_expansion_model: str | None = None
    query_expansion_max_variants: int | None = None
    query_expansion_max_tokens: int | None = None
    query_synonyms: dict[str, list[str]] | None = None
    max_context_chars: int | None = None
    context_batch_chars: int | None = None
    max_context_batches: int | None = None
    max_compact_sources: int | None = None
    context_template: str | None = None
    deep_analysis_enabled: bool | None = None
    deep_final_answer: bool | None = None
    deep_force_all: bool | None = None
    deep_trigger_phrases: list[str] | None = None
    deep_generation_provider: str | None = None
    deep_generation_base_url: str | None = None
    deep_generation_api_key: str | None = None
    deep_generation_model: str | None = None
    deep_top_k: int | None = None
    deep_max_batches: int | None = None
    deep_batch_chars: int | None = None
    deep_batch_max_tokens: int | None = None
    deep_final_max_tokens: int | None = None
    deep_timeout_sec: int | None = None
    compliance_enabled: bool | None = None
    compliance_auto_enabled: bool | None = None
    compliance_allow_user_index_override: bool | None = None
    compliance_index_ids: list[str] | None = None
    compliance_generation_model: str | None = None
    compliance_max_files: int | None = None
    compliance_max_file_mb: int | None = None
    compliance_section_chars: int | None = None
    compliance_max_sections: int | None = None
    compliance_requirement_top_k: int | None = None
    compliance_timeout_sec: int | None = None
    compliance_trigger_phrases: list[str] | None = None
    chat_attachments_enabled: bool | None = None
    chat_attachment_index_prefix: str | None = None
    chat_attachment_max_files: int | None = None
    chat_attachment_max_file_mb: int | None = None
    chat_attachment_timeout_sec: int | None = None
    index_type: str | None = None
    hnsw_threshold_chunks: int | None = None
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    enable_ocr: bool | None = None
    ocr_languages: str | None = None
    ocr_engine: str | None = None
    ocr_gpu: bool | None = None
    ocr_gpu_device: str | None = None
    ocr_model_storage_dir: str | None = None
    pdf_render_scale: float | None = None
    soffice_path: str | None = None
    max_upload_mb: int | None = None
    connect_timeout_sec: float | None = None
    request_timeout_sec: float | None = None
    stream_idle_timeout_sec: float | None = None


__all__ = [
    "IndexCreate",
    "AddPathRequest",
    "ReindexDocumentsRequest",
    "DeleteDocumentsRequest",
    "RebuildIndexRequest",
    "RetrieveRequest",
    "AnalyzeRequest",
    "ComplianceRequest",
    "ChatAttachmentFileMeta",
    "ChatAttachmentsRequest",
    "ConfigUpdate",
]
