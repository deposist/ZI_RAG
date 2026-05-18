from __future__ import annotations

import json
import math
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


DEFAULT_CONTEXT_TEMPLATE = (
    "Используй контекст RAG ниже как приоритетный источник. Контекст может "
    "быть разбит на пачки: просматривай все пачки последовательно и не "
    "игнорируй поздние пачки только из-за их номера. Не придумывай локаторы "
    "и цитаты. Если ответа в контексте нет, честно скажи, что в базе знаний "
    "ответ не найден.\n\n"
    "{knowledge}"
)


DEFAULT_DEEP_TRIGGER_PHRASES = [
    "проанализируй все",
    "проанализировать все",
    "проверь все",
    "проверь всё",
    "проверь все документы",
    "проверь весь пакет",
    "сравни",
    "сравнить",
    "полный перечень",
    "все требования",
    "все нарушения",
    "найди противоречия",
    "найти противоречия",
    "сделай отчет",
    "сделай отчёт",
    "подготовь отчет",
    "подготовь отчёт",
    "ничего не пропусти",
    "полный анализ",
    "по всем документам",
]


DEFAULT_COMPLIANCE_TRIGGER_PHRASES = [
    "проверь на соответствие",
    "проверка нмд",
    "соответствует ли",
    "найди нарушения",
    "найти нарушения",
    "сделай акт",
    "подготовь акт",
    "проведи проверку",
    "проверить документ",
    "матрица соответствия",
    "compliance",
]

DEFAULT_QUERY_SYNONYMS: dict[str, list[str]] = {
    "кспд тспд": [
        "КСПД ТСПД корпоративная сеть передачи данных технологическая сеть передачи данных",
        "КСПД ТСПД подключение оборудования доступ сегментация межсетевое экранирование",
        "КСПД ТСПД требования запрещено не допускается нарушение отдельное сетевое оборудование",
        "КСПД ТСПД использование одних и тех же компонентов отдельное оборудование запрещено",
    ],
    "корпоратив технолог сет": [
        "КСПД ТСПД корпоративная сеть передачи данных технологическая сеть передачи данных",
        "КСПД ТСПД подключение оборудования доступ сегментация межсетевое экранирование",
        "КСПД ТСПД требования запрещено не допускается нарушение отдельное сетевое оборудование",
        "КСПД ТСПД использование одних и тех же компонентов отдельное оборудование запрещено",
    ],
    "интернет тспд": [
        "ТСПД доступ к сети Интернет запрещен исключен сетевой доступ",
    ],
    "интернет технолог сет": [
        "ТСПД доступ к сети Интернет запрещен исключен сетевой доступ",
    ],
    "игр арм": [
        "работникам запрещается самостоятельно устанавливать ПО на АРМ",
        "АРМ стандартное предустановленное ПО разрешенное программное обеспечение",
        "допустимое использование информационных активов самостоятельная установка ПО",
        "использование только разрешенного ПО согласование с ДЗИиИТИ Службой ИБ",
    ],
    "установ игр": [
        "работникам запрещается самостоятельно устанавливать ПО на АРМ",
        "запрещенное программное обеспечение на рабочем месте АРМ",
        "допускается использование только разрешенного ПО",
        "нарушение установленного порядка и правил обращения с информационными активами",
    ],
    "установ по арм": [
        "работникам запрещается самостоятельно устанавливать ПО на АРМ",
        "АРМ предоставляются с предустановленным стандартным ПО",
        "допускается использование только разрешенного ПО",
        "ПО не включенное в перечень разрешенного согласовывается с ДЗИиИТИ Службой ИБ",
    ],
}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_list_value(value: str) -> list[str]:
    if "\n" in value:
        return [part.strip() for part in value.splitlines() if part.strip()]
    return _split_csv(value)


def _env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if lowered in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _default_query_synonyms() -> dict[str, list[str]]:
    return {key: list(values) for key, values in DEFAULT_QUERY_SYNONYMS.items()}


def _normalize_query_synonyms(value: Any) -> dict[str, list[str]]:
    if value is None:
        return _default_query_synonyms()
    if not isinstance(value, dict):
        raise ValueError("query_synonyms must be a JSON object")
    normalized: dict[str, list[str]] = {}
    for raw_key, raw_values in value.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if isinstance(raw_values, str):
            values = [raw_values]
        elif isinstance(raw_values, list):
            values = [str(item).strip() for item in raw_values if str(item).strip()]
        else:
            raise ValueError("query_synonyms values must be strings or string arrays")
        if values:
            normalized[key] = values
    return normalized


_CONFIG_PATH_CONTEXT: ContextVar[Path | None] = ContextVar("zi_rag_config_path", default=None)


StringList = Annotated[list[str], NoDecode]


class SidecarConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZI_RAG_",
        extra="ignore",
        env_ignore_empty=True,
        populate_by_name=True,
        validate_assignment=False,
    )

    storage_dir: str = "./openwebui_zi_rag_storage"
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "ZI_RAG_OLLAMA_BASE_URL"),
    )
    api_key: str = ""
    require_api_key_localhost: bool = False
    allowed_source_roots: StringList = Field(default_factory=list)
    default_index_ids: StringList = Field(default_factory=list)
    embedding_model: str = ""
    embedding_provider: str = "ollama"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    chunk_size: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_cache_dtype: str = "fp32"
    rebuild_debounce_sec: float = Field(default=2.0, ge=0)
    top_k: int = Field(default=8, ge=1)
    score_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    rag_enabled: bool = Field(default=True, validation_alias=AliasChoices("ZI_RAG_ENABLED", "ZI_RAG_RAG_ENABLED"))
    include_sources: bool = True
    retrieval_top_k: int = Field(default=70, ge=1)
    adaptive_score_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    max_prompt_chunks: int = Field(default=24, ge=1)
    min_query_term_hits: int = Field(default=1, ge=0)
    rerank_enabled: bool = False
    rerank_model: str = ""
    rerank_min_results: int = Field(default=10, ge=1)
    rerank_top_n: int = Field(default=50, ge=1)
    query_expansion_enabled: bool = False
    query_expansion_model: str = ""
    query_expansion_max_variants: int = Field(default=3, ge=1, le=8)
    query_expansion_max_tokens: int = Field(default=256, ge=16)
    query_synonyms: dict[str, list[str]] = Field(default_factory=_default_query_synonyms)
    max_context_chars: int = Field(default=32000, ge=1)
    context_batch_chars: int = Field(default=10000, ge=1)
    max_context_batches: int = Field(default=3, ge=1)
    max_compact_sources: int = Field(default=8, ge=0)
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    deep_analysis_enabled: bool = True
    deep_final_answer: bool = True
    deep_force_all: bool = False
    deep_trigger_phrases: StringList = Field(default_factory=lambda: list(DEFAULT_DEEP_TRIGGER_PHRASES))
    deep_generation_provider: str = "ollama"
    deep_generation_base_url: str = ""
    deep_generation_api_key: str = ""
    deep_generation_model: str = ""
    deep_top_k: int = Field(default=70, ge=1)
    deep_max_batches: int = Field(default=10, ge=1)
    deep_batch_chars: int = Field(default=10000, ge=1)
    deep_batch_max_tokens: int = Field(default=1024, ge=1)
    deep_final_max_tokens: int = Field(default=2048, ge=1)
    deep_timeout_sec: int = Field(default=900, ge=1)
    compliance_enabled: bool = True
    compliance_auto_enabled: bool = True
    compliance_allow_user_index_override: bool = True
    compliance_index_ids: StringList = Field(default_factory=list)
    compliance_generation_model: str = ""
    compliance_max_files: int = Field(default=10, ge=1)
    compliance_max_file_mb: int = Field(default=256, ge=1)
    compliance_section_chars: int = Field(default=8000, ge=1)
    compliance_max_sections: int = Field(default=80, ge=1)
    compliance_requirement_top_k: int = Field(default=24, ge=1)
    compliance_timeout_sec: int = Field(default=1200, ge=1)
    compliance_trigger_phrases: StringList = Field(default_factory=lambda: list(DEFAULT_COMPLIANCE_TRIGGER_PHRASES))
    chat_attachments_enabled: bool = True
    chat_attachment_index_prefix: str = "owui_chat_"
    chat_attachment_max_files: int = Field(default=10, ge=1)
    chat_attachment_max_file_mb: int = Field(default=256, ge=1)
    chat_attachment_timeout_sec: int = Field(default=900, ge=1)
    index_type: str = "auto"
    hnsw_threshold_chunks: int = Field(default=50000, ge=1)
    hnsw_m: int = Field(default=32, ge=1)
    hnsw_ef_construction: int = Field(default=200, ge=1)
    hnsw_ef_search: int = Field(default=128, ge=1)
    enable_ocr: bool = False
    ocr_languages: str = "rus+eng"
    ocr_engine: str = "easyocr"
    ocr_gpu: bool = True
    ocr_gpu_device: str = ""
    ocr_model_storage_dir: str = ""
    pdf_render_scale: float = Field(default=2.5, gt=0.0)
    soffice_path: str = "/usr/bin/soffice"
    max_upload_mb: int = Field(default=256, ge=1)
    connect_timeout_sec: float = Field(default=10.0, gt=0.0)
    request_timeout_sec: float = Field(default=120.0, gt=0.0)
    stream_idle_timeout_sec: float = Field(default=120.0, gt=0.0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_path = _CONFIG_PATH_CONTEXT.get()
        json_settings = JsonConfigSettingsSource(
            settings_cls,
            json_file=config_path,
            json_file_encoding="utf-8",
        )
        return init_settings, env_settings, json_settings, dotenv_settings, file_secret_settings

    @field_validator(
        "allowed_source_roots",
        "default_index_ids",
        "deep_trigger_phrases",
        "compliance_index_ids",
        "compliance_trigger_phrases",
        mode="before",
    )
    @classmethod
    def _parse_string_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _split_list_value(value)
        return value

    @field_validator(
        "rag_enabled",
        "require_api_key_localhost",
        "include_sources",
        "rerank_enabled",
        "query_expansion_enabled",
        "deep_analysis_enabled",
        "deep_final_answer",
        "deep_force_all",
        "compliance_enabled",
        "compliance_auto_enabled",
        "compliance_allow_user_index_override",
        "chat_attachments_enabled",
        "enable_ocr",
        "ocr_gpu",
        mode="before",
    )
    @classmethod
    def _parse_bool(cls, value: Any) -> Any:
        if isinstance(value, str):
            parsed = _env_bool(value)
            return value if parsed is None else parsed
        return value

    @field_validator("query_synonyms", mode="before")
    @classmethod
    def _parse_query_synonyms(cls, value: Any) -> dict[str, list[str]]:
        return _normalize_query_synonyms(value)

    @field_validator("embedding_cache_dtype", "index_type", "deep_generation_provider", mode="before")
    @classmethod
    def _normalize_lower_string(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> SidecarConfig:
        validate_config(self)
        return self

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir).expanduser().resolve()

    @property
    def registry_path(self) -> Path:
        return self.storage_path / "registry.sqlite"

    @property
    def uploads_path(self) -> Path:
        return self.storage_path / "uploads"

    @property
    def indexes_path(self) -> Path:
        return self.storage_path / "indexes"

    def ensure_dirs(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.indexes_path.mkdir(parents=True, exist_ok=True)


def default_config_path() -> Path:
    configured = os.getenv("ZI_RAG_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    storage = os.getenv("ZI_RAG_STORAGE_DIR", "./openwebui_zi_rag_storage")
    return Path(storage).expanduser().resolve() / "config.json"


def load_config(path: str | Path | None = None) -> SidecarConfig:
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    token = _CONFIG_PATH_CONTEXT.set(config_path)
    try:
        cfg = SidecarConfig()
    finally:
        _CONFIG_PATH_CONTEXT.reset(token)
    cfg.ensure_dirs()
    return cfg


def validate_config(config: SidecarConfig) -> None:
    if config.require_api_key_localhost and not str(config.api_key or "").strip():
        raise ValueError("api_key is required when require_api_key_localhost is enabled")
    config.embedding_cache_dtype = str(config.embedding_cache_dtype or "fp32").strip().lower()
    if config.embedding_cache_dtype not in {"fp32", "fp16"}:
        raise ValueError("embedding_cache_dtype must be one of: fp32, fp16")
    config.query_synonyms = _normalize_query_synonyms(config.query_synonyms)
    config.index_type = str(config.index_type or "auto").strip().lower()
    if config.index_type not in {"auto", "flat", "hnsw"}:
        raise ValueError("index_type must be one of: auto, flat, hnsw")
    config.deep_generation_provider = str(config.deep_generation_provider or "ollama").strip().lower()
    if config.deep_generation_provider not in {"ollama", "openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}:
        raise ValueError("deep_generation_provider must be one of: ollama, openai, openai-compatible")
    for key in ("hnsw_threshold_chunks", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"):
        try:
            int_value = int(getattr(config, key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a positive integer") from exc
        if int_value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        setattr(config, key, int_value)
    for key in ("connect_timeout_sec", "request_timeout_sec", "stream_idle_timeout_sec"):
        try:
            float_value = float(getattr(config, key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a positive number") from exc
        if not math.isfinite(float_value) or float_value <= 0:
            raise ValueError(f"{key} must be a positive number")
        setattr(config, key, float_value)
    try:
        config.pdf_render_scale = float(config.pdf_render_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("pdf_render_scale must be a positive number") from exc
    if not math.isfinite(config.pdf_render_scale) or config.pdf_render_scale <= 0:
        raise ValueError("pdf_render_scale must be a positive number")


def save_config(config: SidecarConfig, path: str | Path | None = None) -> None:
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, config_path)


def update_config(updates: dict[str, Any], path: str | Path | None = None) -> SidecarConfig:
    cfg = load_config(path)
    valid_fields = set(SidecarConfig.model_fields)
    for key, value in updates.items():
        if key in valid_fields:
            setattr(cfg, key, value)
    cfg = SidecarConfig(**cfg.model_dump())
    cfg.ensure_dirs()
    save_config(cfg, path)
    return cfg
