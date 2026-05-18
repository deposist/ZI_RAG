import asyncio
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
import types
from typing import Any
import zipfile

import pytest
from pydantic_settings import BaseSettings

import openwebui_zi_rag.server as rag_server
import openwebui_functions.zi_rag_filter as rag_filter_module
from openwebui_zi_rag import config as config_module
from openwebui_zi_rag import text_utils
from openwebui_zi_rag.config import SidecarConfig
from openwebui_zi_rag.indexing import extraction as extraction_module
from openwebui_zi_rag.indexing import service as service_module
from openwebui_zi_rag.indexing import vector_store
from openwebui_zi_rag.indexing.chunking import chunk_text
from openwebui_zi_rag.indexing.extraction import easyocr_languages, labeled_blocks, labeled_text, normalize_table_row
from openwebui_zi_rag.indexing.service import (
    chunk_locator,
    chunk_quote,
    llm_query_expansion_variants,
    retrieval_query_variants,
    safe_filename,
)
from openwebui_zi_rag.indexing.service import RagService
from openwebui_zi_rag.indexing.registry import DocumentStatus, JobStatus, Registry
from openwebui_zi_rag.indexing.vector_store import build_index
from openwebui_zi_rag.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaHTTPError,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    OpenAIRerankClient,
    make_embedding_client,
    make_generation_client,
    make_rerank_client,
)
from openwebui_functions.zi_rag_filter import Filter


class FakeEmbeddingClient:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def embed(self, model, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if self.dimension == 5:
                vectors.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif "alpha" in lower:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in lower:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return vectors


class CountingEmbeddingClient(FakeEmbeddingClient):
    def __init__(self, dimension=4):
        super().__init__(dimension=dimension)
        self.calls = 0
        self.texts = 0

    def embed(self, model, texts):
        items = list(texts)
        self.calls += 1
        self.texts += len(items)
        return super().embed(model, items)


class CapturingEmbeddingClient(FakeEmbeddingClient):
    def __init__(self):
        super().__init__()
        self.seen = []

    def embed(self, model, texts):
        self.seen.extend(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class StaticEmbeddingClient(FakeEmbeddingClient):
    def __init__(self, vector):
        super().__init__(dimension=len(vector))
        self.vector = [float(item) for item in vector]

    def embed(self, model, texts):
        return [list(self.vector) for _ in texts]


class SyntheticRankingEmbeddingClient(FakeEmbeddingClient):
    def _vector(self, text):
        lower = text.lower()
        if "alpha" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in lower:
            return [0.0, 0.0, 1.0, 0.0]
        if "delta" in lower:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.25, 0.25, 0.25, 0.25]

    def embed(self, model, texts):
        return [self._vector(text) for text in texts]


class KeywordRerankClient:
    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls: list[dict[str, Any]] = []

    def rerank(self, model, query, documents):
        items = list(documents)
        self.calls.append({"model": model, "query": query, "documents": items})
        return [0.99 if self.keyword in item else max(0.01, 0.5 - index * 0.01) for index, item in enumerate(items)]


class QueryExpansionClient:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, model, messages, temperature=0.1, num_predict=256):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "num_predict": num_predict,
            }
        )
        return self.response


class SlowTrackingEmbeddingClient:
    def __init__(self, delay=0.02):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls: list[list[str]] = []
        self.lock = threading.Lock()

    def embed(self, model, texts):
        items = list(texts)
        with self.lock:
            self.calls.append(items)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return [self._vector_for(text) for text in items]
        finally:
            with self.lock:
                self.active -= 1

    def _vector_for(self, text):
        lower = text.lower()
        if "alpha" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in lower:
            return [0.0, 0.0, 1.0, 0.0]
        if "delta" in lower:
            return [0.0, 0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0, 0.0]


def make_service(tmp_path, embedding_client=None):
    cfg = SidecarConfig(
        storage_dir=str(tmp_path),
        embedding_model="fake-embed",
        chunk_size=80,
        chunk_overlap=10,
        score_threshold=0.0,
    )
    return RagService(cfg, embedding_client=embedding_client or FakeEmbeddingClient())


def make_bm25_synthetic_service(tmp_path):
    chunks = [
        "alpha dense policy overview",
        "needleone exact compliance requirement",
        "alpha needleone hybrid policy requirement",
        "beta dense network control",
        "needletwo exact network evidence",
        "beta needletwo hybrid network evidence",
        "gamma dense operations guide",
        "needlethree exact safety requirement",
        "gamma needlethree hybrid safety requirement",
        "delta dense maintenance guide",
        "needlefour exact maintenance requirement",
        "delta needlefour hybrid maintenance requirement",
        "needlefive only lexical archive",
        "plain unrelated archive",
    ]
    cfg = SidecarConfig(
        storage_dir=str(tmp_path),
        embedding_model="fake-embed",
        score_threshold=0.0,
        query_synonyms={},
    )
    service = RagService(cfg, embedding_client=SyntheticRankingEmbeddingClient())
    index = service.create_index({"name": "BM25 Synthetic", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="synthetic.txt")
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.set_document_status(
        document["id"],
        DocumentStatus.INDEXED,
        text_chars=sum(len(chunk) for chunk in chunks),
        chunk_count=len(chunks),
    )
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(
        service.config.indexes_path,
        index["id"],
        chunk_ids,
        service.embedding_client.embed("fake-embed", chunks),
    )
    return service, index


def test_extraction_adds_document_locators():
    assert labeled_text("3.1 Use approved media", ["абз. 7"]).startswith("[пункт 3.1; абз. 7]")

    blocks = labeled_blocks("First paragraph\nSecond paragraph", ["стр. 2"])

    assert blocks == [
        "[стр. 2; абз. 1] First paragraph",
        "[стр. 2; абз. 2] Second paragraph",
    ]


def test_easyocr_language_aliases_match_tesseract_style():
    assert easyocr_languages("rus+eng") == ["ru", "en"]
    assert easyocr_languages("ru,en,eng") == ["ru", "en"]


def test_pdf_render_scale_defaults_and_accepts_configured_value(tmp_path, monkeypatch):
    assert SidecarConfig().pdf_render_scale == 2.5
    assert extraction_module.pdf_render_scale(None) == 2.5
    assert extraction_module.pdf_render_scale(types.SimpleNamespace(pdf_render_scale=3.25)) == 3.25
    assert extraction_module.pdf_render_scale(types.SimpleNamespace(pdf_render_scale=0)) == 2.5

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"storage_dir": str(tmp_path / "storage")}), encoding="utf-8")
    monkeypatch.setenv("ZI_RAG_PDF_RENDER_SCALE", "1.75")
    cfg = config_module.load_config(config_path)

    assert cfg.pdf_render_scale == 1.75


def test_hnsw_index_config_defaults_env_and_validation(tmp_path, monkeypatch):
    cfg = SidecarConfig()
    assert isinstance(cfg, BaseSettings)
    assert "кспд тспд" in cfg.query_synonyms
    assert cfg.embedding_cache_dtype == "fp32"
    assert cfg.connect_timeout_sec == 10.0
    assert cfg.request_timeout_sec == 120.0
    assert cfg.stream_idle_timeout_sec == 120.0
    assert cfg.rerank_enabled is False
    assert cfg.rerank_model == ""
    assert cfg.rerank_min_results == 10
    assert cfg.rerank_top_n == 50
    assert cfg.query_expansion_enabled is False
    assert cfg.query_expansion_model == ""
    assert cfg.query_expansion_max_variants == 3
    assert cfg.query_expansion_max_tokens == 256
    assert cfg.api_key == ""
    assert cfg.allowed_source_roots == []
    assert cfg.default_index_ids == []
    assert cfg.score_threshold == 0.5
    assert cfg.retrieval_top_k == 70
    assert cfg.adaptive_score_margin == 0.2
    assert cfg.deep_analysis_enabled is True
    assert cfg.deep_final_answer is True
    assert cfg.deep_force_all is False
    assert cfg.deep_generation_provider == "ollama"
    assert cfg.deep_generation_base_url == ""
    assert cfg.deep_generation_api_key == ""
    assert cfg.deep_generation_model == ""
    assert cfg.deep_top_k == 70
    assert cfg.embedding_model == ""
    assert cfg.embedding_base_url == ""
    assert cfg.embedding_api_key == ""
    assert "проверь всё" in cfg.deep_trigger_phrases
    assert cfg.index_type == "auto"
    assert cfg.hnsw_threshold_chunks == 50000
    assert cfg.hnsw_m == 32
    assert cfg.hnsw_ef_construction == 200
    assert cfg.hnsw_ef_search == 128

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"storage_dir": str(tmp_path / "storage")}), encoding="utf-8")
    monkeypatch.setenv("ZI_RAG_INDEX_TYPE", "hnsw")
    monkeypatch.setenv("ZI_RAG_HNSW_THRESHOLD_CHUNKS", "7")
    monkeypatch.setenv("ZI_RAG_HNSW_M", "8")
    monkeypatch.setenv("ZI_RAG_HNSW_EF_CONSTRUCTION", "64")
    monkeypatch.setenv("ZI_RAG_HNSW_EF_SEARCH", "32")
    monkeypatch.setenv("ZI_RAG_EMBEDDING_CACHE_DTYPE", "fp16")
    monkeypatch.setenv("ZI_RAG_CONNECT_TIMEOUT_SEC", "2.5")
    monkeypatch.setenv("ZI_RAG_REQUEST_TIMEOUT_SEC", "30")
    monkeypatch.setenv("ZI_RAG_STREAM_IDLE_TIMEOUT_SEC", "4.5")
    monkeypatch.setenv("ZI_RAG_RERANK_ENABLED", "1")
    monkeypatch.setenv("ZI_RAG_RERANK_MODEL", "reranker")
    monkeypatch.setenv("ZI_RAG_RERANK_MIN_RESULTS", "11")
    monkeypatch.setenv("ZI_RAG_RERANK_TOP_N", "33")
    monkeypatch.setenv("ZI_RAG_QUERY_EXPANSION_ENABLED", "1")
    monkeypatch.setenv("ZI_RAG_QUERY_EXPANSION_MODEL", "query-model")
    monkeypatch.setenv("ZI_RAG_QUERY_EXPANSION_MAX_VARIANTS", "4")
    monkeypatch.setenv("ZI_RAG_QUERY_EXPANSION_MAX_TOKENS", "128")
    monkeypatch.setenv("ZI_RAG_QUERY_SYNONYMS", json.dumps({"alpha": ["beta expansion"]}))
    monkeypatch.setenv("ZI_RAG_DEEP_GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("ZI_RAG_DEEP_GENERATION_BASE_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv("ZI_RAG_DEEP_GENERATION_API_KEY", "secret")

    loaded = config_module.load_config(config_path)

    assert loaded.query_synonyms == {"alpha": ["beta expansion"]}
    assert loaded.embedding_cache_dtype == "fp16"
    assert loaded.connect_timeout_sec == 2.5
    assert loaded.request_timeout_sec == 30.0
    assert loaded.stream_idle_timeout_sec == 4.5
    assert loaded.rerank_enabled is True
    assert loaded.rerank_model == "reranker"
    assert loaded.rerank_min_results == 11
    assert loaded.rerank_top_n == 33
    assert loaded.query_expansion_enabled is True
    assert loaded.query_expansion_model == "query-model"
    assert loaded.query_expansion_max_variants == 4
    assert loaded.query_expansion_max_tokens == 128
    assert loaded.deep_generation_provider == "openai"
    assert loaded.deep_generation_base_url == "http://127.0.0.1:8081/v1"
    assert loaded.deep_generation_api_key == "secret"
    assert loaded.index_type == "hnsw"
    assert loaded.hnsw_threshold_chunks == 7
    assert loaded.hnsw_m == 8
    assert loaded.hnsw_ef_construction == 64
    assert loaded.hnsw_ef_search == 32

    with pytest.raises(ValueError, match="index_type"):
        config_module.validate_config(SidecarConfig(index_type="ivf"))
    with pytest.raises(ValueError, match="hnsw_m"):
        config_module.validate_config(SidecarConfig(hnsw_m=0))
    with pytest.raises(ValueError, match="embedding_cache_dtype"):
        config_module.validate_config(SidecarConfig(embedding_cache_dtype="int8"))
    with pytest.raises(ValueError, match="connect_timeout_sec"):
        config_module.validate_config(SidecarConfig(connect_timeout_sec=0))
    with pytest.raises(ValueError, match="query_synonyms"):
        config_module.validate_config(SidecarConfig(query_synonyms=["bad"]))
    with pytest.raises(ValueError, match="deep_generation_provider"):
        config_module.validate_config(SidecarConfig(deep_generation_provider="bad"))


def test_sidecar_config_ignores_empty_env_values(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "storage_dir": str(tmp_path / "storage"),
                "embedding_provider": "openai",
                "embedding_base_url": "http://127.0.0.1:8082/v1",
                "embedding_model": "local-embeddings",
                "embedding_batch_size": 32,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZI_RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("ZI_RAG_EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("ZI_RAG_EMBEDDING_MODEL", "")
    monkeypatch.setenv("ZI_RAG_EMBEDDING_BATCH_SIZE", "")

    loaded = config_module.load_config(config_path)

    assert loaded.embedding_provider == "openai"
    assert loaded.embedding_base_url == "http://127.0.0.1:8082/v1"
    assert loaded.embedding_model == "local-embeddings"
    assert loaded.embedding_batch_size == 32


def test_sidecar_config_pydantic_settings_source_priority(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "storage_dir": str(tmp_path / "json-storage"),
                "index_type": "flat",
                "allowed_source_roots": ["/json-root"],
                "default_index_ids": ["json-index"],
                "unknown_field": "ignored",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZI_RAG_INDEX_TYPE", "hnsw")
    monkeypatch.setenv("ZI_RAG_ALLOWED_SOURCE_ROOTS", "/env-a,/env-b")
    monkeypatch.setenv("ZI_RAG_DEFAULT_INDEX_IDS", "env-index")

    loaded = config_module.load_config(config_path)

    assert isinstance(loaded, BaseSettings)
    assert loaded.storage_dir == str(tmp_path / "json-storage")
    assert loaded.index_type == "hnsw"
    assert loaded.allowed_source_roots == ["/env-a", "/env-b"]
    assert loaded.default_index_ids == ["env-index"]
    assert "unknown_field" not in loaded.model_dump()
    assert SidecarConfig(index_type="flat").index_type == "flat"


def test_extract_msg_keeps_string_headers_html_body_and_attachment_name(monkeypatch, tmp_path):
    class FakeAttachment:
        longFilename = "report.pdf"
        shortFilename = ""
        displayName = ""
        name = ""
        hidden = False

        def save(self, **_kwargs):
            return []

    class FakeMessage:
        subject = "Quarterly status"
        sender = "alice@example.test"
        to = "bob@example.test"
        cc = ""
        bcc = ""
        date = "2026-05-16"
        body = ""
        htmlBody = "<p>Hello</p><p>World</p>"
        attachments = [FakeAttachment()]

        def __init__(self, _path):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "extract_msg", types.SimpleNamespace(Message=FakeMessage))

    text = extraction_module.extract_msg(tmp_path / "mail.msg")

    assert "Subject: Quarterly status" in text
    assert "From: alice@example.test" in text
    assert "To: bob@example.test" in text
    assert "Date: 2026-05-16" in text
    assert "Hello" in text
    assert "World" in text
    assert "=== Attachment: report.pdf ===" in text


def test_clear_ocr_gpu_cache_unloads_cached_readers():
    extraction_module._EASYOCR_READERS.clear()
    extraction_module._EASYOCR_READERS[(("ru", "en"), True, "0", "/tmp/models")] = object()

    result = extraction_module.clear_ocr_gpu_cache(unload_readers=True)

    assert result["readers_before"] == 1
    assert result["readers_after"] == 0
    assert extraction_module.easyocr_reader_count() == 0


def test_chunk_citation_helpers():
    text = "[стр. 4; абз. 2] В документе сказано использовать утвержденные носители."

    assert chunk_locator(text, 3) == "стр. 4; абз. 2"
    assert chunk_quote(text) == "В документе сказано использовать утвержденные носители."


def test_quotes_remove_table_marker_noise():
    noisy = (
        "[пункт 2.1.12; строка 60] 2.1.12 | Исключен любой сетевой доступ из сегментов "
        "ТСПД в глобальную сеть Интернет | Инфраструктура ТСПД, АСУТП | - | - | V | V | - | - | - | "
        "2.1.13 | Запрещается интеграция компонентов АСУТП с доменами КСПД | - | V | V | - | - |"
    )

    quote = chunk_quote(noisy, max_chars=1000)
    filter_quote = Filter()._quote({"quote": noisy}, max_chars=1000)
    source_line = rag_server._format_source_line(1, {"source": "x.xlsx", "locator": "строка 60", "quote": noisy})

    for value in (quote, filter_quote, source_line):
        assert "| - |" not in value
        assert "| V |" not in value
        assert "Исключен любой сетевой доступ" in value
        assert "Запрещается интеграция" in value


def test_table_normalization_turns_markers_into_semantics():
    previous_rows = [
        ["№", "Требование", "Область", "КСПД", "ТСПД"],
    ]
    row = [
        "2.1.12",
        "Исключен любой сетевой доступ из сегментов ТСПД в глобальную сеть Интернет",
        "Инфраструктура ТСПД",
        "-",
        "V",
    ]

    normalized = normalize_table_row(row, previous_rows)

    assert "| - |" not in normalized
    assert "| V |" not in normalized
    assert "2.1.12" in normalized
    assert "Исключен любой сетевой доступ" in normalized
    assert "Область: Инфраструктура ТСПД" in normalized
    assert "Применимо: ТСПД." in normalized
    assert "Не применимо: КСПД." in normalized


def test_table_normalization_keeps_plain_rows_readable():
    row = ["№", "Требование", "КСПД", "ТСПД"]

    assert normalize_table_row(row, []) == "№ | Требование | КСПД | ТСПД"


def test_chunk_text_keeps_sentence_boundaries_and_locator_prefix():
    text = (
        "[абз. 1] Инцидент регистрируется в журнале учета. "
        "Ответственный сотрудник уведомляет владельца процесса и фиксирует время. "
        "Заключение содержит причины, последствия и корректирующие мероприятия."
    )

    chunks = chunk_text(text, chunk_size=95, chunk_overlap=25)

    assert len(chunks) > 1
    assert all(chunk.startswith("[абз. 1] ") for chunk in chunks)
    for chunk in chunks:
        body = re.sub(r"^\[[^\]]+\]\s*", "", chunk)
        assert body[0].isupper()
        assert body.endswith(".")
        assert not body.startswith(("цидент", "ается"))


def test_chunk_text_splits_long_unpunctuated_text_on_words():
    text = "[абз. 2] " + " ".join(["инцидент", "оформляется"] * 30)

    chunks = chunk_text(text, chunk_size=90, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        body = re.sub(r"^\[[^\]]+\]\s*", "", chunk)
        assert body.split()
        assert all(word in {"инцидент", "оформляется"} for word in body.split())
        assert not body.startswith(("цидент", "ается"))


def test_upload_filename_sanitizers_block_header_and_path_tricks():
    rag_filter = Filter()

    assert rag_filter._multipart_filename('../../evil\r\nX-Bad: 1.pdf') == 'evil__X-Bad: 1.pdf'
    assert rag_filter._multipart_filename('bad\x7fname.txt') == 'bad_name.txt'
    assert rag_filter._multipart_filename('bad\u0085name.txt') == 'bad_name.txt'
    assert rag_filter._multipart_filename('zero\u200bwidth.txt') == 'zero_width.txt'
    assert rag_filter._multipart_filename('отчёт ✅.txt') == 'отчёт ✅.txt'
    assert rag_filter._multipart_content_type('text/plain\r\nX-Bad: 1') == 'application/octet-stream'
    assert safe_filename("\x00\r\n") == "upload"


def test_openwebui_filter_json_export_matches_single_file():
    root = Path(__file__).parents[1]
    filter_content = (root / "openwebui_functions" / "zi_rag_filter.py").read_text(encoding="utf-8")
    exported = json.loads(
        (root / "openwebui_functions" / "zi_rag_filter.openwebui.json").read_text(encoding="utf-8")
    )

    assert exported[0]["content"].rstrip("\n") == filter_content.rstrip("\n")
    assert exported[0]["valves"] == Filter.Valves().model_dump()


def test_build_filter_script_syncs_json_export(tmp_path):
    root = Path(__file__).parents[1]
    script_path = root / "tools" / "build_filter.py"
    spec = importlib.util.spec_from_file_location("zi_rag_build_filter", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    filter_path = tmp_path / "zi_rag_filter.py"
    export_path = tmp_path / "zi_rag_filter.openwebui.json"
    filter_path.write_text("print('new filter')\n", encoding="utf-8")
    export_path.write_text(
        json.dumps([{"id": "zi_rag_filter", "content": "old", "updated_at": 1}]),
        encoding="utf-8",
    )

    assert module.sync_filter_json(filter_path, export_path, timestamp=123) is True
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported[0]["content"] == "print('new filter')\n"
    assert exported[0]["updated_at"] == 123
    assert module.sync_filter_json(filter_path, export_path, timestamp=456) is False
    assert json.loads(export_path.read_text(encoding="utf-8"))[0]["updated_at"] == 123


def test_bundle_build_script_lists_explicit_files_and_excludes_runtime_artifacts(tmp_path):
    root = Path(__file__).parents[1]
    script_path = root / "tools" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("zi_rag_build_bundle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    files = [path.relative_to(root).as_posix() for path in module.iter_bundle_files(root)]

    assert "OPENWEBUI_ZI_RAG.md" in files
    assert "openwebui_zi_rag/server.py" in files
    assert "openwebui_zi_rag/web/app.js" in files
    assert "openwebui_functions/zi_rag_filter.openwebui.json" in files
    assert "openwebui_zi_rag_bundle.zip" not in files
    assert not any(path.startswith("openwebui_zi_rag_storage/") for path in files)
    assert not any("__pycache__" in path for path in files)

    output = module.build_bundle(tmp_path / "bundle.zip", root)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "openwebui_zi_rag/server.py" in names
    assert "openwebui_zi_rag_bundle.zip" not in names


def test_openwebui_filter_request_state_ttl_cleans_orphaned_entries(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rag_filter_module.time, "monotonic", lambda: now[0])
    rag_filter = Filter()
    rag_filter._store_sources("old", [{"source": "old.txt"}])
    rag_filter._store_deep_answer("old", "old answer")

    now[0] += Filter.REQUEST_STATE_TTL_SEC + 1.0
    asyncio.run(rag_filter.inlet({"messages": []}))

    assert rag_filter._sources_by_key == {}
    assert rag_filter._deep_answers_by_key == {}


def test_openwebui_filter_request_key_avoids_parallel_user_collision(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_sidecar_admin_config", lambda: {})

    def fake_post(path, payload, *args):
        assert path == "/analyze"
        return {
            "answer": f"answer for {payload['query']}",
            "stats": {"batches": 1, "filtered": 1},
        }

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    user = {"id": "same-user"}
    body_alpha = {"model": "g4", "messages": [{"role": "user", "content": "/deep alpha"}]}
    body_beta = {"model": "g4", "messages": [{"role": "user", "content": "/deep beta"}]}

    asyncio.run(
        rag_filter.inlet(
            body_alpha,
            __user__=user,
            __user_valves__={"deep_analysis_enabled": False, "deep_final_answer": True},
        )
    )
    asyncio.run(
        rag_filter.inlet(
            body_beta,
            __user__=user,
            __user_valves__={"deep_analysis_enabled": False, "deep_final_answer": True},
        )
    )

    assert len(rag_filter._deep_answers_by_key) == 2

    body_beta["messages"].append({"role": "assistant", "content": "placeholder beta"})
    body_alpha["messages"].append({"role": "assistant", "content": "placeholder alpha"})
    outlet_beta = asyncio.run(rag_filter.outlet(body_beta, __user__=user))
    outlet_alpha = asyncio.run(rag_filter.outlet(body_alpha, __user__=user))

    assert outlet_beta["messages"][-1]["content"] == "answer for beta"
    assert outlet_alpha["messages"][-1]["content"] == "answer for alpha"


def test_ollama_chat_returns_message_content(monkeypatch):
    client = OllamaClient("http://ollama.test")

    def fake_request(path, payload=None):
        assert path == "/api/chat"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.1
        assert payload["options"]["num_predict"] == 512
        return {"message": {"content": " Deep answer "}}

    monkeypatch.setattr(client, "_json_request", fake_request)

    answer = client.chat(
        "test-model",
        [{"role": "user", "content": "question"}],
        num_predict=512,
    )

    assert answer == "Deep answer"


def test_ollama_chat_streaming_skips_malformed_ndjson(monkeypatch):
    client = OllamaClient("http://ollama.test")

    class FakeStreamingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            return iter(
                [
                    b"bad-json\n",
                    b'{"message":{"content":"Hello "}}\n',
                    b'{"response":"world"}\n',
                    b'{"done":true}\n',
                ]
            )

        def close(self):
            pass

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://ollama.test/api/chat"
        assert timeout == 120
        return FakeStreamingResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    answer = client.chat(
        "test-model",
        [{"role": "user", "content": "question"}],
        cancel_check=lambda: False,
    )

    assert answer == "Hello world"


def test_embedding_models_openai_urlopen_runs_in_thread(monkeypatch, tmp_path):
    from openwebui_zi_rag.routes import admin as admin_routes

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "embedding-a"}]}).encode("utf-8")

    urlopen_thread_ids: list[int] = []
    loop_thread_ids: list[int] = []

    def fake_urlopen(request, timeout):
        urlopen_thread_ids.append(threading.get_ident())
        assert request.full_url == "http://emb.test/v1/models"
        assert timeout == 1.0
        return FakeResponse()

    cfg = SidecarConfig(
        storage_dir=str(tmp_path),
        embedding_provider="openai",
        embedding_base_url="http://emb.test/v1",
        request_timeout_sec=1.0,
    )
    monkeypatch.setattr(admin_routes.urllib.request, "urlopen", fake_urlopen)
    admin_routes.model_cache().clear()

    async def run_route():
        loop_thread_ids.append(threading.get_ident())
        return await admin_routes.embedding_models(_=None, cfg=cfg)

    try:
        result = asyncio.run(run_route())
    finally:
        admin_routes.model_cache().clear()

    assert result == {"models": [{"name": "embedding-a"}]}
    assert urlopen_thread_ids
    assert loop_thread_ids
    assert urlopen_thread_ids[0] != loop_thread_ids[0]


def test_generation_model_resolution_requires_configured_available_model():
    class FakeClient:
        def list_models(self):
            return [{"name": "configured-model"}, {"model": "other-model"}]

    assert SidecarConfig().deep_generation_model == ""
    assert (
        rag_server._resolve_generation_model(["missing-model", "ollama:configured-model"], FakeClient())
        == "configured-model"
    )

    with pytest.raises(rag_server.HTTPException) as exc_info:
        rag_server._resolve_generation_model(["", ""], FakeClient())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Generation model is not configured. Select one of: configured-model, other-model"
    )


def test_generation_model_resolution_reports_empty_and_unavailable_model_lists():
    class EmptyClient:
        def list_models(self):
            return []

    class BrokenClient:
        def list_models(self):
            raise OllamaError("tags unavailable")

    with pytest.raises(rag_server.HTTPException) as empty_exc:
        rag_server._resolve_generation_model([""], EmptyClient())
    assert empty_exc.value.status_code == 409
    assert empty_exc.value.detail == "Generation model is not configured. Select one of: no models available"

    with pytest.raises(rag_server.HTTPException) as broken_exc:
        rag_server._resolve_generation_model(["configured-model"], BrokenClient())
    assert broken_exc.value.status_code == 502
    assert "Could not load Ollama models" in broken_exc.value.detail


def test_generation_model_ui_saves_empty_deep_model_without_fallback():
    app_js = (Path(__file__).parents[1] / "openwebui_zi_rag" / "web" / "app.js").read_text(encoding="utf-8")

    assert "qwen3.6:latest" not in app_js
    assert "deep_generation_model: els.deepGenerationModelInput.value.trim()," in app_js


def test_generation_model_ui_distinguishes_empty_list_and_ollama_error():
    app_js = (Path(__file__).parents[1] / "openwebui_zi_rag" / "web" / "app.js").read_text(encoding="utf-8")

    assert '"Модель не выбрана"' in app_js
    assert '"Список моделей пуст"' in app_js
    assert '"Ошибка /ollama/models"' in app_js
    assert 't("models.notSelected"' in app_js
    assert 't("models.empty"' in app_js
    assert 't("models.ollamaError"' in app_js


def test_hnsw_config_fields_are_exposed_in_admin_ui():
    root = Path(__file__).parents[1]
    index_html = (root / "openwebui_zi_rag" / "web" / "index.html").read_text(encoding="utf-8")
    app_js = (root / "openwebui_zi_rag" / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="embeddingCacheDtypeInput"' in index_html
    assert "embedding_cache_dtype" in app_js
    assert 'id="indexTypeInput"' in index_html
    assert 'id="hnswThresholdInput"' in index_html
    assert "hnsw_threshold_chunks" in app_js
    assert "hnsw_ef_search" in app_js
    assert 'id="queryExpansionEnabledInput"' in index_html
    assert 'id="queryExpansionModelInput"' in index_html
    assert "query_expansion_enabled" in app_js
    assert "query_expansion_model" in app_js


def test_admin_ui_i18n_messages_are_complete():
    root = Path(__file__).parents[1] / "openwebui_zi_rag" / "web"
    index_html = (root / "index.html").read_text(encoding="utf-8")
    app_js = (root / "app.js").read_text(encoding="utf-8")
    ru_messages = json.loads((root / "messages.ru.json").read_text(encoding="utf-8"))
    en_messages = json.loads((root / "messages.en.json").read_text(encoding="utf-8"))

    keys = set(re.findall(r'\bt\("([^"]+)"', app_js))
    keys.update(re.findall(r'data-i18n(?:-(?:title|placeholder))?="([^"]+)"', index_html))

    assert 'id="languageSelect"' in index_html
    assert "messages.${selected}.json" in app_js
    assert set(ru_messages) == set(en_messages)
    assert not sorted(keys - set(ru_messages))
    assert ru_messages["actions.refresh"] == "Обновить"
    assert en_messages["actions.refresh"] == "Refresh"


def test_require_api_key_localhost_strict_mode(monkeypatch):
    cfg = SidecarConfig()
    request = types.SimpleNamespace(client=types.SimpleNamespace(host="127.0.0.1"))
    monkeypatch.setattr(rag_server, "get_config", lambda: cfg)

    rag_server.require_api_key(request, x_api_key=None, authorization=None)

    cfg.require_api_key_localhost = True
    with pytest.raises(rag_server.HTTPException) as missing_key:
        rag_server.require_api_key(request, x_api_key=None, authorization=None)
    assert missing_key.value.status_code == 401

    cfg.api_key = "secret"
    with pytest.raises(rag_server.HTTPException) as wrong_key:
        rag_server.require_api_key(request, x_api_key="wrong", authorization=None)
    assert wrong_key.value.status_code == 401

    rag_server.require_api_key(request, x_api_key="secret", authorization=None)


def test_update_config_rejects_strict_localhost_without_api_key(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"storage_dir": str(tmp_path / "storage")}), encoding="utf-8")

    with pytest.raises(ValueError, match="api_key is required"):
        config_module.update_config(
            {"require_api_key_localhost": True, "api_key": ""},
            path=config_path,
        )

    cfg = config_module.update_config(
        {"require_api_key_localhost": True, "api_key": "secret"},
        path=config_path,
    )
    assert cfg.require_api_key_localhost is True
    assert cfg.api_key == "secret"


def test_health_checks_dependencies_and_reads_faiss(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    class FakeOllamaClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_models(self):
            return [{"name": "test-generation"}]

    service = make_service(tmp_path)
    index = service.create_index({"name": "Health", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])
    service.config.api_key = "secret"
    monkeypatch.setattr(rag_server, "OllamaClient", FakeOllamaClient)
    original_get_config = rag_server.get_config
    rag_server.app.dependency_overrides[original_get_config] = lambda: service.config
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    monkeypatch.setattr(rag_server, "get_config", lambda: service.config)
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        public_response = client.get("/health")
        full_unauthenticated_response = client.get("/health/full")
        full_response = client.get("/health/full", headers={"X-API-Key": "secret"})
    finally:
        rag_server.app.dependency_overrides.clear()

    public_payload = public_response.json()
    assert public_response.status_code == 200
    assert set(public_payload) == {"status", "version", "checks"}
    assert public_payload["status"] == "ok"
    assert public_payload["checks"]["sqlite"]["status"] == "ok"
    assert public_payload["checks"]["ollama"] == {"status": "ok", "model_count": 1}
    assert public_payload["checks"]["faiss"]["status"] == "ok"
    assert public_payload["checks"]["faiss"]["index_count"] >= 1
    assert "index_id" not in public_payload["checks"]["faiss"]
    public_json = json.dumps(public_payload, ensure_ascii=False)
    assert str(service.config.storage_path) not in public_json
    assert str(service.config.registry_path) not in public_json
    assert "storage_dir" not in public_payload
    assert "registry" not in public_payload
    assert "metrics" not in public_payload
    assert "embedding_model_dimension" not in public_payload

    assert full_unauthenticated_response.status_code == 401
    payload = full_response.json()
    assert full_response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["storage_dir"] == str(service.config.storage_path)
    assert payload["checks"]["sqlite"]["status"] == "ok"
    assert payload["checks"]["ollama"] == {"status": "ok", "model_count": 1}
    assert payload["checks"]["faiss"]["status"] == "ok"
    assert payload["checks"]["faiss"]["index_id"] == index["id"]
    assert payload["checks"]["faiss"]["chunk_count"] > 0
    dimensions = payload["embedding_model_dimension"]
    assert dimensions["current_embedding_model"] == "fake-embed"
    assert dimensions["indexes"][0]["embedding_model"] == "fake-embed"
    assert dimensions["indexes"][0]["embedding_dim"] == 4
    assert dimensions["warnings"] == []


def test_health_warns_about_embedding_model_dimension_mismatch(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    class FakeOllamaClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_models(self):
            return [{"name": "test-generation"}]

    service = make_service(tmp_path)
    index = service.create_index({"name": "Old Model", "embedding_model": "old-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])
    service.config.embedding_model = "new-embed"
    monkeypatch.setattr(rag_server, "OllamaClient", FakeOllamaClient)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: service.config
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.get("/health/full")
    finally:
        rag_server.app.dependency_overrides.clear()

    payload = response.json()
    dimensions = payload["embedding_model_dimension"]
    assert response.status_code == 200
    assert dimensions["current_embedding_model"] == "new-embed"
    assert dimensions["indexes"][0]["embedding_model"] == "old-embed"
    assert dimensions["indexes"][0]["embedding_dim"] == 4
    assert dimensions["warnings"] == [
        f"Index {index['id']} uses embedding_model=old-embed, current config embedding_model=new-embed"
    ]


def test_health_returns_503_when_ollama_ping_fails(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    class BrokenOllamaClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_models(self):
            raise rag_server.OllamaError("tags unavailable")

    service = make_service(tmp_path)
    monkeypatch.setattr(rag_server, "OllamaClient", BrokenOllamaClient)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: service.config
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.get("/health")
        full_response = client.get("/health/full")
    finally:
        rag_server.app.dependency_overrides.clear()

    payload = response.json()
    assert response.status_code == 503
    assert payload["status"] == "error"
    assert payload["checks"]["sqlite"]["status"] == "ok"
    assert payload["checks"]["faiss"]["status"] == "skipped"
    assert payload["checks"]["ollama"] == {"status": "error", "error": "check failed"}

    full_payload = full_response.json()
    assert full_response.status_code == 503
    assert full_payload["unhealthy"] == ["ollama"]
    assert "tags unavailable" in full_payload["checks"]["ollama"]["error"]


def test_api_smoke_health_config_and_retrieve(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    class FakeOllamaClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_models(self):
            return [{"name": "test-generation"}]

    service = make_service(tmp_path)
    index = service.create_index({"name": "Smoke", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])
    monkeypatch.setattr(rag_server, "OllamaClient", FakeOllamaClient)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: service.config
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        health_response = client.get("/health")
        config_response = client.get("/config")
        retrieve_response = client.post(
            "/retrieve",
            json={"query": "alpha", "index_ids": [index["id"]], "top_k": 3, "score_threshold": 0.0},
        )
    finally:
        rag_server.app.dependency_overrides.clear()

    assert health_response.status_code == 200
    assert health_response.json()["checks"]["faiss"]["status"] == "ok"
    assert config_response.status_code == 200
    assert config_response.json()["storage_dir"] == str(tmp_path)
    assert retrieve_response.status_code == 200
    results = retrieve_response.json()["results"]
    assert results
    assert results[0]["source"] == "alpha.txt"


def test_server_runtime_state_uses_app_state_and_refreshes_cache(tmp_path):
    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    state = rag_server.app.state
    attrs = [
        "zi_rag_initialized",
        "zi_rag_config",
        "zi_rag_service",
        "zi_rag_model_cache",
        "zi_rag_analysis_jobs",
        "zi_rag_analysis_jobs_lock",
    ]
    missing = object()
    snapshot = {name: getattr(state, name, missing) for name in attrs}
    original_service = snapshot["zi_rag_service"]
    try:
        service = make_service(tmp_path)
        rag_server.configure_runtime_state(service=service)

        assert rag_server.get_config() is service.config
        assert rag_server.get_service() is service
        assert state.zi_rag_config is service.config
        assert state.zi_rag_service is service

        state.zi_rag_model_cache["models"] = (time.monotonic(), {"models": []})
        state.zi_rag_analysis_jobs["stale"] = {
            "id": "stale",
            "status": "completed",
            "updated_at": time.time() - rag_server._ANALYSIS_JOB_TTL_SEC - 1,
        }
        rag_server._cleanup_analysis_jobs()
        assert state.zi_rag_analysis_jobs == {}

        refreshed_config = SidecarConfig(storage_dir=str(tmp_path / "refresh"))
        rag_server._refresh_service(refreshed_config)

        assert rag_server.get_config() is refreshed_config
        assert rag_server.get_service().config is refreshed_config
        assert state.zi_rag_model_cache == {}
    finally:
        current_service = getattr(state, "zi_rag_service", None)
        if current_service is not original_service and current_service is not None:
            try:
                current_service.registry.close()
            except Exception:
                pass
        for name, value in snapshot.items():
            if value is missing:
                try:
                    delattr(state, name)
                except AttributeError:
                    pass
            else:
                setattr(state, name, value)


def test_analyze_job_sse_streams_snapshot_and_done(tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    state = rag_server.app.state
    attrs = [
        "zi_rag_initialized",
        "zi_rag_config",
        "zi_rag_service",
        "zi_rag_model_cache",
        "zi_rag_analysis_jobs",
        "zi_rag_analysis_jobs_lock",
    ]
    missing = object()
    snapshot = {name: getattr(state, name, missing) for name in attrs}
    original_service = snapshot["zi_rag_service"]
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    try:
        service = make_service(tmp_path)
        rag_server.configure_runtime_state(service=service)
        now = time.time()
        with rag_server._analysis_jobs_lock():
            rag_server._analysis_jobs()["sse-1"] = {
                "id": "sse-1",
                "status": "completed",
                "message": "готово",
                "events": [{"stage": "done", "message": "готово", "done": True}],
                "result": {"answer": "ok"},
                "error": "",
                "created_at": now,
                "updated_at": now,
                "finished_at": now,
            }

        client = TestClient(rag_server.app, raise_server_exceptions=False)
        with client.stream("GET", "/analyze/jobs/sse-1/events") as response:
            body = "".join(response.iter_text())
    finally:
        rag_server.app.dependency_overrides.clear()
        current_service = getattr(state, "zi_rag_service", None)
        if current_service is not original_service and current_service is not None:
            try:
                current_service.registry.close()
            except Exception:
                pass
        for name, value in snapshot.items():
            if value is missing:
                try:
                    delattr(state, name)
                except AttributeError:
                    pass
            else:
                setattr(state, name, value)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: analysis" in body
    assert "event: done" in body
    assert '"status": "completed"' in body
    assert '"answer": "ok"' in body


def test_analyze_jobs_return_410_after_runtime_restart(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from openwebui_zi_rag.routes import analyze as analyze_routes

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    state = rag_server.app.state
    attrs = [
        "zi_rag_initialized",
        "zi_rag_config",
        "zi_rag_service",
        "zi_rag_model_cache",
        "zi_rag_analysis_jobs",
        "zi_rag_analysis_jobs_lock",
    ]
    missing = object()
    snapshot = {name: getattr(state, name, missing) for name in attrs}
    original_service = snapshot["zi_rag_service"]
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    monkeypatch.setattr(analyze_routes, "run_analysis_job", lambda *_args, **_kwargs: None)
    try:
        service = make_service(tmp_path)
        rag_server.configure_runtime_state(service=service)
        client = TestClient(rag_server.app, raise_server_exceptions=False)

        created = client.post("/analyze/jobs", json={"query": "lost job"})
        assert created.status_code == 200
        job_id = str(created.json()["id"])
        with rag_server._analysis_jobs_lock():
            assert job_id in rag_server._analysis_jobs()

        restarted_service = make_service(tmp_path / "restart")
        rag_server.configure_runtime_state(service=restarted_service)
        with rag_server._analysis_jobs_lock():
            assert rag_server._analysis_jobs() == {}

        read_response = client.get(f"/analyze/jobs/{job_id}")
        cancel_response = client.post(f"/analyze/jobs/{job_id}/cancel")
        with client.stream("GET", f"/analyze/jobs/{job_id}/events") as stream_response:
            stream_body = "".join(stream_response.iter_text())
    finally:
        rag_server.app.dependency_overrides.clear()
        current_service = getattr(state, "zi_rag_service", None)
        if current_service is not original_service and current_service is not None:
            try:
                current_service.registry.close()
            except Exception:
                pass
        for name, value in snapshot.items():
            if value is missing:
                try:
                    delattr(state, name)
                except AttributeError:
                    pass
            else:
                setattr(state, name, value)

    detail = "Sidecar restarted, multi-pass job is gone. Retry from filter."
    assert read_response.status_code == 410
    assert read_response.json()["detail"] == detail
    assert cancel_response.status_code == 410
    assert cancel_response.json()["detail"] == detail
    assert stream_response.status_code == 410
    assert stream_response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in stream_body
    assert detail in stream_body


def test_openapi_routes_have_tags_and_summaries():
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    client = TestClient(rag_server.app, raise_server_exceptions=False)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    for method, path in rag_server.OPENAPI_ROUTE_METADATA:
        operation = paths[path][method.lower()]
        assert operation.get("tags")
        assert operation.get("summary")
    assert paths["/health"]["get"]["tags"] == ["Health"]
    assert paths["/retrieve"]["post"]["summary"] == "Выполнить RAG retrieval"


def test_path_errors_do_not_leak_absolute_paths(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    service = make_service(tmp_path)
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    service.config.allowed_source_roots = [str(allowed_root)]
    index = service.create_index({"name": "Masked Paths", "embedding_model": "fake-embed"})

    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.post(
            f"/indexes/{index['id']}/documents/add-path",
            json={"path": str(outside_root), "recursive": False, "index_now": False},
        )
    finally:
        rag_server.app.dependency_overrides.clear()

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail == "Path is outside allowed_source_roots"
    assert str(outside_root) not in detail

    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    stored_path = Path(document["stored_path"])
    stored_path.unlink()
    with pytest.raises(FileNotFoundError) as exc_info:
        service.index_document_now(document["id"])
    message = str(exc_info.value)
    assert "File not found:" in message
    assert str(tmp_path) not in message


def test_sync_analyze_returns_499_on_ollama_cancelled(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    def raise_cancelled(*_args, **_kwargs):
        raise rag_server.OllamaCancelled("stop")

    monkeypatch.setattr(rag_server, "run_multi_pass_analysis", raise_cancelled)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: SidecarConfig(storage_dir=str(tmp_path))
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: object()
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.post("/analyze", json={"query": "stop"})
    finally:
        rag_server.app.dependency_overrides.clear()

    assert response.status_code == 499
    assert response.json()["detail"] == "Analysis canceled"


def test_sync_compliance_analyze_returns_499_on_ollama_cancelled(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    def raise_cancelled(*_args, **_kwargs):
        raise rag_server.OllamaCancelled("stop")

    monkeypatch.setattr(rag_server, "run_compliance_analysis", raise_cancelled)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: SidecarConfig(storage_dir=str(tmp_path))
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: object()
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.post(
            "/compliance/analyze",
            data={"payload": json.dumps({"query": "check"})},
            files=[("files", ("policy.txt", b"hello", "text/plain"))],
        )
    finally:
        rag_server.app.dependency_overrides.clear()

    assert response.status_code == 499
    assert response.json()["detail"] == "Analysis canceled"


def test_openai_embedding_client_reads_ordered_embeddings(monkeypatch):
    client = OpenAIEmbeddingClient("http://emb.test/v1", api_key="secret")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return (
                b'{"data":['
                b'{"index":1,"embedding":[0,1]},'
                b'{"index":0,"embedding":[1,0]}'
                b']}'
            )

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://emb.test/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer secret"
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.embed("giga", ["q", "d"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_rerank_client_reads_indexed_scores(monkeypatch):
    client = OpenAIRerankClient("http://emb.test/v1", api_key="secret")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return (
                b'{"results":['
                b'{"index":1,"relevance_score":0.2},'
                b'{"index":0,"score":0.9}'
                b']}'
            )

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://emb.test/v1/rerank"
        assert request.headers["Authorization"] == "Bearer secret"
        assert json.loads(request.data.decode("utf-8")) == {
            "model": "reranker",
            "query": "q",
            "documents": ["a", "b"],
        }
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.rerank("reranker", "q", ["a", "b"]) == [0.9, 0.2]


def test_make_rerank_client_uses_embedding_endpoint_when_enabled():
    cfg = SidecarConfig(
        rerank_enabled=True,
        rerank_model="reranker",
        embedding_base_url="http://rerank.test/v1",
        embedding_api_key="secret",
        request_timeout_sec=7,
    )

    client = make_rerank_client(cfg)

    assert isinstance(client, OpenAIRerankClient)
    assert client.base_url == "http://rerank.test/v1"
    assert client.api_key == "secret"
    assert client.timeout == 7
    assert make_rerank_client(SidecarConfig(rerank_enabled=False, rerank_model="reranker")) is None


def test_ollama_client_uses_separate_connect_request_and_stream_timeouts(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.timeouts = []

        def settimeout(self, timeout):
            self.timeouts.append(timeout)

    class FakeRaw:
        def __init__(self, sock):
            self._sock = sock

    class FakeFp:
        def __init__(self, sock):
            self.raw = FakeRaw(sock)

    class FakeJsonResponse:
        def __init__(self, sock):
            self.fp = FakeFp(sock)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"models":[]}'

    class FakeStreamResponse:
        def __init__(self, sock):
            self.fp = FakeFp(sock)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            return iter([b'{"message":{"content":"ok"},"done":true}\n'])

    sockets = [FakeSocket(), FakeSocket()]
    urlopen_timeouts = []

    def fake_urlopen(request, timeout):
        urlopen_timeouts.append(timeout)
        if request.full_url.endswith("/api/tags"):
            return FakeJsonResponse(sockets[0])
        return FakeStreamResponse(sockets[1])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(
        "http://ollama.test",
        timeout=99,
        connect_timeout=1.5,
        request_timeout=2.5,
        stream_idle_timeout=3.5,
    )

    assert client.list_models() == []
    assert client.chat("model", [{"role": "user", "content": "q"}], cancel_check=lambda: False) == "ok"
    assert urlopen_timeouts == [1.5, 1.5]
    assert sockets[0].timeouts == [2.5]
    assert sockets[1].timeouts == [3.5]


def test_openai_chat_client_lists_models_and_chats(monkeypatch):
    calls: list[tuple[str, dict[str, Any] | None, dict[str, str]]] = []
    timeouts: list[float] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        timeouts.append(timeout)
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        headers = dict(request.header_items())
        calls.append((request.full_url, body, headers))
        if request.full_url.endswith("/models"):
            return FakeResponse({"data": [{"id": "local-llama"}]})
        return FakeResponse({"choices": [{"message": {"content": " answer "}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIChatClient(
        "http://openai.test/v1",
        api_key="key",
        timeout=99,
        connect_timeout=1.5,
        request_timeout=7,
    )

    assert client.list_models() == [{"id": "local-llama", "name": "local-llama", "model": "local-llama"}]
    assert client.chat("local-llama", [{"role": "user", "content": "q"}], num_predict=12) == "answer"
    assert calls[0][0] == "http://openai.test/v1/models"
    assert calls[1][0] == "http://openai.test/v1/chat/completions"
    assert calls[1][1]["max_tokens"] == 12
    assert calls[1][2]["Authorization"] == "Bearer key"
    assert timeouts == [7, 7]


def test_make_generation_client_uses_openai_compatible_deep_endpoint():
    cfg = SidecarConfig(
        deep_generation_provider="openai",
        deep_generation_base_url="http://127.0.0.1:8081/v1",
        deep_generation_api_key="key",
    )

    client = make_generation_client(cfg, request_timeout=5)

    assert isinstance(client, OpenAIChatClient)
    assert client.base_url == "http://127.0.0.1:8081/v1"
    assert client.api_key == "key"
    assert client.request_timeout == 5


def test_make_embedding_client_passes_ollama_timeout_fields():
    cfg = types.SimpleNamespace(
        embedding_provider="ollama",
        ollama_base_url="http://ollama.test",
        request_timeout_sec=30,
        connect_timeout_sec=2.5,
        stream_idle_timeout_sec=4.5,
    )

    client = make_embedding_client(cfg)

    assert isinstance(client, OllamaClient)
    assert client.request_timeout == 30
    assert client.connect_timeout == 2.5
    assert client.stream_idle_timeout == 4.5


def test_ollama_embed_falls_back_to_legacy_only_for_404(monkeypatch):
    client = OllamaClient("http://ollama.test")
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_request(path, payload=None):
        calls.append((path, payload))
        if path == "/api/embed":
            raise OllamaHTTPError(404, "not found")
        assert path == "/api/embeddings"
        return {"embedding": [len(calls), 0.5]}

    monkeypatch.setattr(client, "_json_request", fake_request)

    assert client.embed("fake-embed", ["one", "two"]) == [[2.0, 0.5], [3.0, 0.5]]
    assert [path for path, _payload in calls] == ["/api/embed", "/api/embeddings", "/api/embeddings"]


def test_ollama_embed_does_not_fallback_for_500(monkeypatch):
    client = OllamaClient("http://ollama.test")
    calls: list[str] = []

    def fake_request(path, payload=None):
        calls.append(path)
        raise OllamaHTTPError(500, "oom")

    monkeypatch.setattr(client, "_json_request", fake_request)

    with pytest.raises(OllamaHTTPError) as exc_info:
        client.embed("fake-embed", ["one"])

    assert exc_info.value.status_code == 500
    assert calls == ["/api/embed"]


def test_registry_embedding_cache_dtype_fp16_recovers_with_tolerance(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("Cache dtype")
    document = registry.create_document(index["id"], filename="cache.txt")
    chunk_id = registry.replace_document_chunks(index["id"], document["id"], ["cache text"])[0]
    vector = [0.123456, -0.25, 1.0, 0.0]

    registry.save_chunk_embeddings("model-fp16", [(chunk_id, vector)], dtype="fp16")
    registry.save_chunk_embeddings("model-fp32", [(chunk_id, vector)], dtype="fp32")

    with registry.connect() as conn:
        fp16_row = conn.execute(
            "SELECT dim, embedding_blob FROM chunk_embeddings WHERE model = ?",
            ("model-fp16",),
        ).fetchone()
        fp32_row = conn.execute(
            "SELECT dim, embedding_blob FROM chunk_embeddings WHERE model = ?",
            ("model-fp32",),
        ).fetchone()

    assert fp16_row["dim"] == len(vector)
    assert len(fp16_row["embedding_blob"]) == len(vector) * 2
    assert len(fp32_row["embedding_blob"]) == len(vector) * 4
    assert registry.get_chunk_embeddings("model-fp16", [chunk_id])[chunk_id] == pytest.approx(
        vector,
        abs=1e-3,
    )
    assert registry.get_chunk_embeddings("model-fp32", [chunk_id])[chunk_id] == pytest.approx(vector)


def test_registry_records_schema_versions(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")

    versions = registry.schema_versions()

    assert [item["version"] for item in versions] == [1, 2, 3, 4, 5]
    assert [item["migration_id"] for item in versions] == [
        "initial_schema",
        "documents_external_metadata",
        "embedding_text_cache",
        "chunk_fts",
        "document_fts",
    ]
    with registry.connect() as conn:
        row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
    assert int(row["version"]) == 5


def test_registry_init_skips_existing_chunk_fts_rebuild(tmp_path):
    db_path = tmp_path / "registry.sqlite"
    registry = Registry(db_path)
    index = registry.create_index("FTS init skip")
    document = registry.create_document(index["id"], filename="fts.txt")
    chunk_id = registry.replace_document_chunks(index["id"], document["id"], ["needle text"])[0]
    with registry.connect() as conn:
        conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            """
            INSERT INTO chunk_fts (chunk_id, index_id, document_id, text)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, index["id"], document["id"], "staleonly text"),
        )
    registry.close()

    reopened = Registry(db_path)
    try:
        assert reopened.search_chunks_fts(index["id"], "needle") == []
        assert [item[0] for item in reopened.search_chunks_fts(index["id"], "staleonly")] == [chunk_id]
    finally:
        reopened.close()


def test_registry_init_restores_empty_chunk_fts_with_active_chunks(tmp_path):
    db_path = tmp_path / "registry.sqlite"
    registry = Registry(db_path)
    index = registry.create_index("FTS init restore")
    document = registry.create_document(index["id"], filename="fts.txt")
    chunk_id = registry.replace_document_chunks(index["id"], document["id"], ["recoverneedle text"])[0]
    with registry.connect() as conn:
        conn.execute("DELETE FROM chunk_fts")
    registry.close()

    reopened = Registry(db_path)
    try:
        assert [item[0] for item in reopened.search_chunks_fts(index["id"], "recoverneedle")] == [chunk_id]
    finally:
        reopened.close()


def test_replace_document_chunks_preserves_unchanged_embeddings_and_clears_stale(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("Chunk upsert")
    document = registry.create_document(index["id"], filename="chunks.txt")
    first_ids = registry.replace_document_chunks(index["id"], document["id"], ["same", "old", "removed"])
    registry.save_chunk_embeddings(
        "model",
        [
            (first_ids[0], [1.0, 0.0]),
            (first_ids[1], [0.0, 1.0]),
            (first_ids[2], [0.5, 0.5]),
        ],
    )

    second_ids = registry.replace_document_chunks(index["id"], document["id"], ["same", "new"])

    assert second_ids == first_ids[:2]
    cached = registry.get_chunk_embeddings("model", first_ids)
    assert cached[first_ids[0]] == [1.0, 0.0]
    assert first_ids[1] not in cached
    assert first_ids[2] not in cached
    active = registry.active_chunks(index["id"])
    assert [item["id"] for item in active] == second_ids
    assert [item["text"] for item in active] == ["same", "new"]


def test_embedding_prefixes_are_applied_for_documents_and_queries(tmp_path):
    client = CapturingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    service.config.embedding_document_prefix = "DOC: "
    service.config.embedding_query_prefix = "QUERY: "
    index = service.create_index({"name": "Prefixed", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")

    service.index_document_now(document["id"])
    service.retrieve("alpha", index_ids=[index["id"]], top_k=1)

    assert any(text.startswith("DOC: ") for text in client.seen)
    assert "QUERY: alpha" in client.seen


def test_index_document_now_marks_document_failed_when_rebuild_fails(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Rebuild Failure", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")

    def fail_rebuild(index_id, *, job_id=None):
        raise RuntimeError(f"rebuild boom: {index_id}:{job_id or ''}")

    monkeypatch.setattr(service, "_rebuild_index_debounced", fail_rebuild)

    with pytest.raises(RuntimeError, match="rebuild boom"):
        service.index_document_now(document["id"])

    stored = service.registry.get_document(document["id"])
    assert stored["status"] == DocumentStatus.FAILED.value
    assert "rebuild boom" in stored["error"]

    job_document = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    job = service.registry.create_job("index_document", index_id=index["id"], document_id=job_document["id"])

    service.run_job(job["id"], service.index_document_now, job_document["id"])

    stored_job = service.registry.get_job(job["id"])
    stored_job_document = service.registry.get_document(job_document["id"])
    assert stored_job["status"] == JobStatus.FAILED.value
    assert "rebuild boom" in stored_job["error"]
    assert stored_job_document["status"] == DocumentStatus.FAILED.value
    assert "rebuild boom" in stored_job_document["error"]


def test_save_upload_concurrent_same_filename_uses_unique_storage_paths(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Concurrent Uploads", "embedding_model": "fake-embed"})
    total = 12
    barrier = threading.Barrier(total)
    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            document = service.save_upload(
                index["id"],
                "same.txt",
                f"body-{worker_id}".encode("utf-8"),
                "text/plain",
            )
            with lock:
                results.append(document)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(total)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert len(results) == total
    assert all(document["filename"] == "same.txt" for document in results)
    stored_paths = [Path(document["stored_path"]) for document in results]
    assert len({str(path) for path in stored_paths}) == total
    assert all(path.exists() for path in stored_paths)
    assert all(path.name != "same.txt" for path in stored_paths)
    assert {path.read_bytes() for path in stored_paths} == {
        f"body-{worker_id}".encode("utf-8") for worker_id in range(total)
    }


def test_deleted_document_is_excluded_from_retrieval(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Test", "embedding_model": "fake-embed"})
    document = service.save_upload(
        index["id"],
        "alpha.txt",
        b"alpha document with searchable text",
        "text/plain",
    )

    service.index_document_now(document["id"])
    before = service.retrieve("alpha", index_ids=[index["id"]], top_k=5)
    assert before["results"]
    assert before["results"][0]["source"] == "alpha.txt"

    service.delete_document(document["id"])
    after = service.retrieve("alpha", index_ids=[index["id"]], top_k=5)
    assert after["results"] == []


def test_registry_migrates_external_document_columns(tmp_path):
    db_path = tmp_path / "registry.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                index_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL DEFAULT '',
                stored_path TEXT NOT NULL DEFAULT '',
                mime_type TEXT NOT NULL DEFAULT '',
                file_hash TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT NOT NULL DEFAULT '',
                text_chars INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            """
        )

    registry = Registry(db_path)
    with registry.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}

    assert {"external_id", "external_source", "metadata_json"}.issubset(columns)


def test_bulk_delete_documents_rebuilds_once_and_keeps_remaining_docs(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Bulk Delete", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    gamma_doc = service.save_upload(index["id"], "gamma.txt", b"gamma text", "text/plain")
    service.index_documents_now(index["id"], [alpha_doc["id"], beta_doc["id"], gamma_doc["id"]])
    rebuilds = 0
    original_rebuild = service.rebuild_index_now

    def counting_rebuild(*args, **kwargs):
        nonlocal rebuilds
        rebuilds += 1
        return original_rebuild(*args, **kwargs)

    service.rebuild_index_now = counting_rebuild

    result = service.delete_documents(index["id"], [alpha_doc["id"], beta_doc["id"]])

    assert rebuilds == 1
    assert result["deleted_count"] == 2
    assert service.registry.get_document(alpha_doc["id"])["deleted_at"]
    assert service.registry.get_document(beta_doc["id"])["deleted_at"]
    assert service.registry.get_document(gamma_doc["id"])["deleted_at"] is None
    alpha_sources = {
        item["source"]
        for item in service.retrieve("alpha", index_ids=[index["id"]], top_k=5, score_threshold=0.1)["results"]
    }
    assert "alpha.txt" not in alpha_sources
    assert "beta.txt" not in alpha_sources
    gamma_results = service.retrieve("gamma", index_ids=[index["id"]], top_k=5, score_threshold=0.1)["results"]
    assert gamma_results
    assert gamma_results[0]["source"] == "gamma.txt"


def test_rebuild_index_documents_now_allows_empty_index(tmp_path):
    service = make_service(tmp_path)
    service.config.embedding_model = ""
    index = service.create_index({"name": "Empty Rebuild", "embedding_model": ""})
    vector_path, map_path = vector_store.index_paths(service.config.indexes_path, index["id"])
    vector_path.write_bytes(b"stale")
    map_path.write_text(json.dumps(["stale-chunk"]), encoding="utf-8")

    result = service.rebuild_index_documents_now(index["id"])

    assert result["documents"] == 0
    assert result["chunks"] == 0
    assert result["embedding_dim"] == 0
    assert result["rebuild"]["chunks"] == 0
    assert vector_path.exists() is False
    assert json.loads(map_path.read_text(encoding="utf-8")) == []
    assert service.registry.get_index(index["id"])["embedding_dim"] == 0


def test_rebuild_empty_index_after_rest_bulk_delete(tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    service = make_service(tmp_path)
    index = service.create_index({"name": "REST Empty Rebuild", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_documents_now(index["id"], [alpha_doc["id"], beta_doc["id"]])
    vector_path, map_path = vector_store.index_paths(service.config.indexes_path, index["id"])
    assert vector_path.exists()

    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        delete_response = client.post(
            f"/indexes/{index['id']}/documents/delete",
            json={"document_ids": [alpha_doc["id"], beta_doc["id"]]},
        )
        rebuild_response = client.post(f"/indexes/{index['id']}/rebuild", json={})
    finally:
        rag_server.app.dependency_overrides.clear()

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_count"] == 2
    assert rebuild_response.status_code == 200
    job_id = rebuild_response.json()["job"]["id"]
    job = service.registry.get_job(job_id)
    payload = json.loads(job["result_json"])
    assert job["status"] == JobStatus.COMPLETED.value
    assert payload["chunks"] == 0
    assert payload["embedding_dim"] == 0
    assert payload["rebuild"]["chunks"] == 0
    assert vector_path.exists() is False
    assert json.loads(map_path.read_text(encoding="utf-8")) == []


def test_retrieve_deduplicates_identical_chunks(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Test", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    chunks = ["alpha duplicate", "alpha duplicate", "alpha other"]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    build_index(
        service.config.indexes_path,
        index["id"],
        chunk_ids,
        service.embedding_client.embed("fake-embed", chunks),
    )

    result = service.retrieve("alpha", index_ids=[index["id"]], top_k=5, score_threshold=0.0)

    texts = [item["text"] for item in result["results"]]
    assert texts.count("alpha duplicate") == 1


def test_retrieve_mmr_diversifies_top_k_with_chunk_embeddings(tmp_path):
    service = make_service(tmp_path, embedding_client=StaticEmbeddingClient([0.8, 0.6, 0.0, 0.0]))
    index = service.create_index({"name": "MMR", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="mmr.txt")
    chunks = [
        "alpha primary requirement",
        "alpha near duplicate requirement",
        "beta diverse access requirement",
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    cache_key = service._embedding_cache_key("fake-embed", query=False)
    service.registry.save_chunk_embeddings(cache_key, zip(chunk_ids, vectors))
    service.registry.set_document_status(
        document["id"],
        DocumentStatus.INDEXED,
        text_chars=sum(len(chunk) for chunk in chunks),
        chunk_count=len(chunks),
    )
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(service.config.indexes_path, index["id"], chunk_ids, vectors)

    result = service.retrieve("alpha access", index_ids=[index["id"]], top_k=2, score_threshold=0.0)
    texts = [item["text"] for item in result["results"]]

    assert "beta diverse access requirement" in texts
    assert sum(1 for text in texts if text.startswith("alpha ")) == 1
    assert all("_embedding_cache_key" not in item for item in result["results"])
    assert result["stats"]["mmr_selected"] == 2


def test_retrieve_hybrid_bm25_rrf_promotes_exact_term(tmp_path):
    service = make_service(tmp_path, embedding_client=StaticEmbeddingClient([1.0, 0.0, 0.0, 0.0]))
    index = service.create_index({"name": "Hybrid", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="hybrid.txt")
    chunks = [
        "general alpha policy overview",
        "needleterm exact access control requirement",
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(service.config.indexes_path, index["id"], chunk_ids, vectors)

    result = service.retrieve("needleterm", index_ids=[index["id"]], top_k=1, score_threshold=0.0)

    assert result["results"][0]["text"] == chunks[1]
    assert "bm25" in result["results"][0]["retrieval_sources"]
    assert result["stats"]["fts_hits"] >= 1


def test_bm25_lexical_score_normalizes_by_rank():
    assert service_module._lexical_score_from_fts_rank(1) == pytest.approx(0.95)
    assert service_module._lexical_score_from_fts_rank(2) == pytest.approx(0.5)
    assert service_module._lexical_score_from_fts_rank(5) == pytest.approx(0.2)
    assert service_module._lexical_score_from_fts_rank(10) == pytest.approx(0.1)


def test_retrieve_bm25_only_scores_follow_rank_without_072_plateau(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "BM25 Rank", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="bm25.txt")
    chunks = [f"needle rank {rank}" for rank in range(1, 11)]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.set_document_status(
        document["id"],
        DocumentStatus.INDEXED,
        text_chars=sum(len(chunk) for chunk in chunks),
        chunk_count=len(chunks),
    )

    def fake_fts_search(index_id, query, *, limit=100):
        assert index_id == index["id"]
        assert query == "needle"
        assert limit >= 10
        return [(chunk_id, -float(rank)) for rank, chunk_id in enumerate(chunk_ids, start=1)]

    monkeypatch.setattr(service.registry, "search_chunks_fts", fake_fts_search)

    result = service.retrieve("needle", index_ids=[index["id"]], top_k=10, score_threshold=0.0)
    scores = [item["score"] for item in result["results"]]

    assert scores[:4] == pytest.approx([0.95, 0.5, 1 / 3, 0.25])
    assert scores[4] == pytest.approx(0.2)
    assert scores[4] < 0.72


def test_retrieve_score_uses_max_dense_and_bm25_lexical_cap(tmp_path):
    service = make_service(tmp_path, embedding_client=StaticEmbeddingClient([1.0, 0.0, 0.0, 0.0]))
    index = service.create_index({"name": "BM25 Dense", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="bm25-dense.txt")
    chunks = ["needle dense lexical hybrid"]
    vectors = [[0.7, 0.714142842854285, 0.0, 0.0]]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.set_document_status(
        document["id"],
        DocumentStatus.INDEXED,
        text_chars=len(chunks[0]),
        chunk_count=len(chunks),
    )
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(service.config.indexes_path, index["id"], chunk_ids, vectors)

    result = service.retrieve("needle", index_ids=[index["id"]], top_k=1, score_threshold=0.0)

    assert result["results"][0]["score"] == pytest.approx(0.95)
    assert "bm25" in result["results"][0]["retrieval_sources"]
    assert "dense" in result["results"][0]["retrieval_sources"]


def test_bm25_synthetic_ranking_matches_baseline(tmp_path):
    baseline_path = Path(__file__).parent / "data" / "bm25_ranking_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    service, index = make_bm25_synthetic_service(tmp_path)

    for case in baseline["queries"]:
        result = service.retrieve(
            case["query"],
            index_ids=[index["id"]],
            top_k=int(baseline["top_k"]),
            score_threshold=0.0,
        )
        assert [item["text"] for item in result["results"]] == case["top_texts"]


def test_retrieve_llm_query_expansion_adds_search_variants(tmp_path):
    generation_client = QueryExpansionClient('["beta policy"]')
    cfg = SidecarConfig(
        storage_dir=str(tmp_path),
        embedding_model="fake-embed",
        score_threshold=0.5,
        query_synonyms={},
        query_expansion_enabled=True,
        query_expansion_model="qwen3.6:latest",
        query_expansion_max_variants=2,
        query_expansion_max_tokens=64,
    )
    service = RagService(
        cfg,
        embedding_client=FakeEmbeddingClient(),
        generation_client=generation_client,
    )
    index = service.create_index({"name": "Expansion", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="expansion.txt")
    chunks = ["beta policy requirement"]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(
        service.config.indexes_path,
        index["id"],
        chunk_ids,
        service.embedding_client.embed("fake-embed", chunks),
    )

    result = service.retrieve("alpha question", index_ids=[index["id"]], top_k=1, score_threshold=0.5)

    assert result["results"][0]["text"] == "beta policy requirement"
    assert result["results"][0]["query_variant"] == "beta policy"
    assert result["stats"]["query_expansion_applied"] is True
    assert result["stats"]["query_expansion_variants"] == 1
    assert result["stats"]["query_embedding_variants"] == 2
    assert generation_client.calls[0]["model"] == "qwen3.6:latest"
    assert generation_client.calls[0]["num_predict"] == 64


def test_retrieve_cross_encoder_reranks_candidate_pool(tmp_path):
    rerank_client = KeywordRerankClient("winner")
    cfg = SidecarConfig(
        storage_dir=str(tmp_path),
        embedding_model="fake-embed",
        score_threshold=0.0,
        rerank_enabled=True,
        rerank_model="reranker",
        rerank_min_results=10,
        rerank_top_n=20,
    )
    service = RagService(
        cfg,
        embedding_client=StaticEmbeddingClient([1.0, 0.0, 0.0, 0.0]),
        rerank_client=rerank_client,
    )
    index = service.create_index({"name": "Rerank", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="rerank.txt")
    chunks = [f"ordinary chunk {index}" for index in range(11)] + ["winner chunk"]
    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in chunks]
    chunk_ids = service.registry.replace_document_chunks(index["id"], document["id"], chunks)
    service.registry.update_index_embedding(index["id"], "fake-embed", 4)
    build_index(service.config.indexes_path, index["id"], chunk_ids, vectors)

    result = service.retrieve("find winner", index_ids=[index["id"]], top_k=3, score_threshold=0.0)

    assert result["results"][0]["text"] == "winner chunk"
    assert result["results"][0]["rerank_score"] == 0.99
    assert "retrieval_score" in result["results"][0]
    assert result["stats"]["rerank_applied"] is True
    assert result["stats"]["rerank_candidates"] == 12
    assert rerank_client.calls[0]["model"] == "reranker"
    assert rerank_client.calls[0]["query"] == "find winner"


def test_registry_chunk_fts_updates_with_chunk_replacement_and_delete(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("FTS")
    document = registry.create_document(index["id"], filename="fts.txt")

    registry.replace_document_chunks(index["id"], document["id"], ["oldneedle text"])
    assert registry.search_chunks_fts(index["id"], "oldneedle")

    registry.replace_document_chunks(index["id"], document["id"], ["newneedle text"])
    assert registry.search_chunks_fts(index["id"], "oldneedle") == []
    assert registry.search_chunks_fts(index["id"], "newneedle")

    registry.soft_delete_document(document["id"])
    assert registry.search_chunks_fts(index["id"], "newneedle") == []


def test_embedding_dimension_mismatch_blocks_retrieval(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Test", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])

    service.embedding_client = FakeEmbeddingClient(dimension=5)

    with pytest.raises(ValueError, match="embedding_dim"):
        service.retrieve("alpha", index_ids=[index["id"]], embedding_model="other-embed")


def test_retrieve_uses_sidecar_default_indexes(tmp_path):
    service = make_service(tmp_path)
    alpha_index = service.create_index({"name": "Alpha", "embedding_model": "fake-embed"})
    beta_index = service.create_index({"name": "Beta", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(alpha_index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(beta_index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_document_now(alpha_doc["id"])
    service.index_document_now(beta_doc["id"])

    service.config.default_index_ids = [beta_index["id"]]

    result = service.retrieve("alpha", top_k=5)

    assert result["results"]
    assert {item["index_id"] for item in result["results"]} == {beta_index["id"]}


def test_retrieve_adds_extra_indexes_without_replacing_default(tmp_path):
    service = make_service(tmp_path)
    alpha_index = service.create_index({"name": "Alpha", "embedding_model": "fake-embed"})
    beta_index = service.create_index({"name": "Beta", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(alpha_index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(beta_index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_document_now(alpha_doc["id"])
    service.index_document_now(beta_doc["id"])
    service.config.default_index_ids = [beta_index["id"]]

    result = service.retrieve("alpha", top_k=5, extra_index_ids=[alpha_index["id"]])

    assert {item["index_id"] for item in result["results"]} == {alpha_index["id"], beta_index["id"]}


def test_chat_attachment_indexing_dedupes_and_replaces(tmp_path):
    service = make_service(tmp_path)

    first = service.index_chat_attachments(
        "chat-1",
        [
            {
                "filename": "alpha.txt",
                "content_type": "text/plain",
                "content": b"alpha text",
                "external_id": "file-1",
                "metadata": {"id": "file-1"},
            }
        ],
        chat_id="chat-1",
        user_id="u1",
        message_id="m1",
    )

    assert first["index_id"] == "owui_chat_chat-1"
    assert len(first["indexed_document_ids"]) == 1

    second = service.index_chat_attachments(
        "chat-1",
        [
            {
                "filename": "alpha.txt",
                "content_type": "text/plain",
                "content": b"alpha text",
                "external_id": "file-1",
                "metadata": {"id": "file-1"},
            }
        ],
        chat_id="chat-1",
    )

    assert second["indexed_document_ids"] == []
    assert second["skipped"][0]["reason"] == "already_indexed"

    third = service.index_chat_attachments(
        "chat-1",
        [
            {
                "filename": "alpha.txt",
                "content_type": "text/plain",
                "content": b"beta text",
                "external_id": "file-1",
                "metadata": {"id": "file-1"},
            }
        ],
        chat_id="chat-1",
    )

    docs = service.registry.list_documents(first["index_id"])
    assert len(docs) == 1
    assert docs[0]["id"] != first["indexed_document_ids"][0]
    assert third["indexed_document_ids"] == [docs[0]["id"]]
    result = service.retrieve("beta", extra_index_ids=[first["index_id"]], top_k=5, score_threshold=0.0)
    assert result["results"][0]["source"].startswith("alpha")


def test_chat_attachment_indexing_enforces_wall_clock_deadline(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    now = [0.0]
    monkeypatch.setattr(service_module.time, "monotonic", lambda: now[0])
    original_upsert = service.upsert_chat_attachment
    upserted: list[str] = []

    def slow_upsert(index_id, *, filename, content, mime_type="", external_id="", metadata=None):
        upserted.append(filename)
        result = original_upsert(
            index_id,
            filename=filename,
            content=content,
            mime_type=mime_type,
            external_id=external_id,
            metadata=metadata,
        )
        now[0] += 2.0
        return result

    monkeypatch.setattr(service, "upsert_chat_attachment", slow_upsert)

    with pytest.raises(service_module.IndexingDeadlineExceeded, match="Chat attachment indexing timed out"):
        service.index_chat_attachments(
            "chat-deadline",
            [
                {
                    "filename": "alpha.txt",
                    "content_type": "text/plain",
                    "content": b"alpha text",
                    "external_id": "file-1",
                    "metadata": {"id": "file-1"},
                },
                {
                    "filename": "beta.txt",
                    "content_type": "text/plain",
                    "content": b"beta text",
                    "external_id": "file-2",
                    "metadata": {"id": "file-2"},
                },
            ],
            chat_id="chat-deadline",
            deadline=1.0,
        )

    index_id = service.chat_attachment_index_id("chat-deadline")
    documents = service.registry.list_documents(index_id)
    assert upserted == ["alpha.txt"]
    assert [document["filename"] for document in documents] == ["alpha.txt"]
    assert service.registry.list_jobs(index_id=index_id) == []


def test_chat_attachment_route_returns_504_on_deadline(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    if rag_server.app is None:
        pytest.skip("FastAPI is not installed")

    service = make_service(tmp_path)
    service.config.chat_attachment_timeout_sec = 1
    calls: list[dict[str, Any]] = []

    def raise_timeout(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        assert kwargs.get("deadline") is not None
        raise service_module.IndexingDeadlineExceeded("Chat attachment indexing timed out")

    monkeypatch.setattr(service, "index_chat_attachments", raise_timeout)
    rag_server.app.dependency_overrides[rag_server.require_api_key] = lambda: None
    rag_server.app.dependency_overrides[rag_server.get_config] = lambda: service.config
    rag_server.app.dependency_overrides[rag_server.get_service] = lambda: service
    try:
        client = TestClient(rag_server.app, raise_server_exceptions=False)
        response = client.post(
            "/chat-attachments/index",
            data={"payload": json.dumps({"chat_id": "chat-deadline"})},
            files=[("files", ("alpha.txt", b"alpha text", "text/plain"))],
        )
    finally:
        rag_server.app.dependency_overrides.clear()

    assert response.status_code == 504
    assert response.json()["detail"] == "Chat attachment indexing timed out"
    assert len(calls) == 1


def test_faiss_index_cache_reads_disk_once(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Cached", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])
    vector_store.clear_index_cache()

    faiss, _ = vector_store._imports()
    original_read_index = faiss.read_index
    calls = {"read_index": 0}

    def counting_read_index(*args, **kwargs):
        calls["read_index"] += 1
        return original_read_index(*args, **kwargs)

    monkeypatch.setattr(faiss, "read_index", counting_read_index)

    first = service.retrieve("alpha", index_ids=[index["id"]], top_k=5)
    second = service.retrieve("alpha", index_ids=[index["id"]], top_k=5)

    assert first["results"]
    assert second["results"]
    assert calls["read_index"] == 1


def test_faiss_index_cache_is_bounded_lru(monkeypatch, tmp_path):
    vector_store.clear_index_cache()
    monkeypatch.setattr(vector_store, "_INDEX_CACHE_MAX_SIZE", 2)

    try:
        build_index(tmp_path, "first", ["first"], [[1.0, 0.0, 0.0, 0.0]])
        build_index(tmp_path, "second", ["second"], [[0.0, 1.0, 0.0, 0.0]])
        cached_index, chunk_ids = vector_store._cached_index(tmp_path, "first")
        assert cached_index is not None
        assert chunk_ids == ["first"]

        build_index(tmp_path, "third", ["third"], [[0.0, 0.0, 1.0, 0.0]])

        keys = list(vector_store._INDEX_CACHE.keys())
        first_key = str((tmp_path / "first" / "vectors.faiss").resolve())
        second_key = str((tmp_path / "second" / "vectors.faiss").resolve())
        third_key = str((tmp_path / "third" / "vectors.faiss").resolve())

        assert len(keys) == 2
        assert first_key in keys
        assert second_key not in keys
        assert third_key in keys
    finally:
        vector_store.clear_index_cache()


def test_faiss_auto_index_type_selects_flat_and_hnsw(tmp_path):
    vector_store.clear_index_cache()
    faiss, _ = vector_store._imports()

    try:
        build_index(
            tmp_path,
            "flat",
            ["alpha", "beta"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            index_type="auto",
            hnsw_threshold_chunks=2,
        )
        flat_index, flat_chunks = vector_store._cached_index(tmp_path, "flat")

        build_index(
            tmp_path,
            "hnsw",
            ["alpha", "beta", "gamma"],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            index_type="auto",
            hnsw_threshold_chunks=2,
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=32,
        )
        hnsw_index, hnsw_chunks = vector_store._cached_index(tmp_path, "hnsw")
        hnsw_results = vector_store.search_index(tmp_path, "hnsw", [1.0, 0.0, 0.0, 0.0], top_k=1)

        assert type(flat_index).__name__ == "IndexFlatIP"
        assert flat_chunks == ["alpha", "beta"]
        assert "HNSW" in type(hnsw_index).__name__
        assert hnsw_index.metric_type == faiss.METRIC_INNER_PRODUCT
        assert hnsw_index.hnsw.efConstruction == 64
        assert hnsw_index.hnsw.efSearch == 32
        assert hnsw_chunks == ["alpha", "beta", "gamma"]
        assert hnsw_results[0][0] == "alpha"
    finally:
        vector_store.clear_index_cache()


def test_openai_compatible_embedding_batches_run_in_parallel_and_keep_order(tmp_path):
    embedding_client = SlowTrackingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=embedding_client)
    service.config.embedding_provider = "openai"
    service.config.embedding_batch_size = 1
    index = service.create_index({"name": "Parallel Embeddings", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="parallel.txt")
    chunk_ids = service.registry.replace_document_chunks(
        index["id"],
        document["id"],
        ["alpha text", "beta text", "gamma text", "delta text"],
    )

    service.rebuild_index_now(index["id"])

    cache_key = service._embedding_cache_key("fake-embed", query=False)
    cached = service.registry.get_chunk_embeddings(cache_key, chunk_ids)
    assert embedding_client.max_active > 1
    assert cached[chunk_ids[0]] == [1.0, 0.0, 0.0, 0.0]
    assert cached[chunk_ids[1]] == [0.0, 1.0, 0.0, 0.0]
    assert cached[chunk_ids[2]] == [0.0, 0.0, 1.0, 0.0]
    assert cached[chunk_ids[3]] == [0.0, 0.0, 0.0, 1.0]


def test_ollama_embedding_batches_remain_sequential(tmp_path):
    embedding_client = SlowTrackingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=embedding_client)
    service.config.embedding_provider = "ollama"
    service.config.embedding_batch_size = 1
    index = service.create_index({"name": "Sequential Embeddings", "embedding_model": "fake-embed"})
    document = service.registry.create_document(index["id"], filename="sequential.txt")
    service.registry.replace_document_chunks(
        index["id"],
        document["id"],
        ["alpha text", "beta text", "gamma text", "delta text"],
    )

    service.rebuild_index_now(index["id"])

    assert embedding_client.max_active == 1
    assert len(embedding_client.calls) == 4


def test_faiss_retrieve_and_rebuild_are_synchronized(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Concurrent FAISS", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_documents_now(index["id"], [alpha_doc["id"], beta_doc["id"]])
    vector_store.clear_index_cache()

    original_write = vector_store._write_faiss_atomic
    write_started = threading.Event()

    def slow_write(*args, **kwargs):
        write_started.set()
        threading.Event().wait(0.003)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(vector_store, "_write_faiss_atomic", slow_write)
    stop = threading.Event()
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    errors: list[BaseException] = []
    retrieve_counts: list[int] = []

    def record_error(exc: BaseException) -> None:
        with lock:
            errors.append(exc)

    def rebuild_worker() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(12):
                service.rebuild_index_now(index["id"])
        except BaseException as exc:
            record_error(exc)
        finally:
            stop.set()

    def retrieve_worker() -> None:
        count = 0
        try:
            barrier.wait(timeout=5)
            while not stop.is_set() or count < 5:
                result = service.retrieve("alpha", index_ids=[index["id"]], top_k=5, score_threshold=0.0)
                assert result["results"]
                count += 1
        except BaseException as exc:
            record_error(exc)
        finally:
            with lock:
                retrieve_counts.append(count)

    threads = [
        threading.Thread(target=rebuild_worker),
        threading.Thread(target=retrieve_worker),
        threading.Thread(target=retrieve_worker),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert write_started.is_set()
    assert not errors
    assert sum(retrieve_counts) >= 10
    cached_index, chunk_ids = vector_store._cached_index(service.config.indexes_path, index["id"])
    assert cached_index is not None
    assert cached_index.ntotal == len(chunk_ids)


def test_delete_index_and_retrieve_are_synchronized(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Delete Retrieve Race", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.index_document_now(document["id"])
    vector_store.clear_index_cache()

    original_cached_index = vector_store._cached_index
    retrieve_entered = threading.Event()

    def slow_cached_index(*args, **kwargs):
        retrieve_entered.set()
        threading.Event().wait(0.03)
        return original_cached_index(*args, **kwargs)

    monkeypatch.setattr(vector_store, "_cached_index", slow_cached_index)
    errors: list[BaseException] = []
    results: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []

    def retrieve_worker() -> None:
        try:
            results.append(service.retrieve("alpha", index_ids=[index["id"]], top_k=5, score_threshold=0.0))
        except BaseException as exc:
            errors.append(exc)

    def delete_worker() -> None:
        try:
            assert retrieve_entered.wait(timeout=5)
            deleted.append(service.delete_index(index["id"]))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=retrieve_worker), threading.Thread(target=delete_worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert deleted and deleted[0]["id"] == index["id"]
    assert results and isinstance(results[0]["results"], list)
    assert service.registry.get_index(index["id"]) is None
    assert service.retrieve("alpha", index_ids=[index["id"]], top_k=5, score_threshold=0.0)["results"] == []


def test_retrieve_reuses_query_embedding_for_same_model_indexes(tmp_path):
    client = CountingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    alpha_index = service.create_index({"name": "Alpha", "embedding_model": "fake-embed"})
    beta_index = service.create_index({"name": "Beta", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(alpha_index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(beta_index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_document_now(alpha_doc["id"])
    service.index_document_now(beta_doc["id"])
    client.calls = 0
    client.texts = 0

    result = service.retrieve("alpha", index_ids=[alpha_index["id"], beta_index["id"]], top_k=5)

    assert result["stats"]["query_embedding_calls"] == 1
    assert client.calls == 1
    assert client.texts == 1


def test_retrieval_query_variants_expand_kspd_tspd_question():
    variants = retrieval_query_variants(
        "Какие требования НМД нарушаются, если работник подключает оборудование "
        "одновременно в корпоративную и технологическую сети передачи данных?"
    )

    joined = "\n".join(variants).lower()
    assert len(variants) > 1
    assert "кспд" in joined
    assert "тспд" in joined
    assert "отдельное" in joined
    assert "запрещено" in joined


def test_retrieval_query_variants_expand_arm_games_to_software_policy():
    variants = retrieval_query_variants(
        "какие пункты нарушены у работника, который установил на свой рабочий АРМ игры? проверь всё",
        {},
    )

    joined = "\n".join(variants).lower()
    assert len(variants) > 1
    assert "самостоятельно устанавливать по" in joined
    assert "разрешенного по" in joined
    assert "арм" in joined


def test_retrieval_query_variants_use_configurable_synonyms():
    variants = retrieval_query_variants(
        "alpha policy",
        {"alpha": ["beta expansion", "gamma expansion"]},
    )

    assert "beta expansion" in variants
    assert "gamma expansion" in variants
    assert retrieval_query_variants("alpha policy", {}) == ["alpha policy"]


def test_llm_query_expansion_variants_parse_json_and_numbered_lines():
    assert llm_query_expansion_variants('{"queries":["alpha policy","beta policy"]}', max_variants=3) == [
        "alpha policy",
        "beta policy",
    ]
    assert llm_query_expansion_variants("1. gamma policy\n2) gamma policy\n- delta policy", max_variants=3) == [
        "gamma policy",
        "delta policy",
    ]


def test_rebuild_reuses_cached_chunk_embeddings(tmp_path):
    client = CountingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    index = service.create_index({"name": "Embedding Cache", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    service.index_documents_now(index["id"], [alpha_doc["id"], beta_doc["id"]])
    client.calls = 0
    client.texts = 0

    cached = service.rebuild_index_now(index["id"])

    assert cached["new_embeddings"] == 0
    assert cached["cached_embeddings"] == 2
    assert client.texts == 0

    service.registry.replace_document_chunks(index["id"], alpha_doc["id"], ["alpha changed"])
    changed = service.rebuild_index_now(index["id"])

    assert changed["new_embeddings"] == 1
    assert changed["cached_embeddings"] == 1
    assert client.texts == 1


def test_rebuild_deduplicates_duplicate_chunk_embeddings_by_text_hash(tmp_path):
    client = CountingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    service.config.embedding_batch_size = 1
    index = service.create_index({"name": "Duplicate Embeddings", "embedding_model": "fake-embed"})
    first_doc = service.registry.create_document(index["id"], filename="first.txt")
    first_chunk_ids = service.registry.replace_document_chunks(
        index["id"],
        first_doc["id"],
        ["alpha duplicate", "alpha duplicate", "beta unique"],
    )

    first = service.rebuild_index_now(index["id"])

    assert first["chunks"] == 3
    assert first["new_embeddings"] == 2
    assert first["cached_embeddings"] == 1
    assert client.texts == 2
    cache_key = service._embedding_cache_key("fake-embed", query=False)
    assert set(service.registry.get_chunk_embeddings(cache_key, first_chunk_ids)) == set(first_chunk_ids)

    second_doc = service.registry.create_document(index["id"], filename="second.txt")
    second_chunk_ids = service.registry.replace_document_chunks(index["id"], second_doc["id"], ["alpha duplicate"])
    client.calls = 0
    client.texts = 0

    second = service.rebuild_index_now(index["id"])

    assert second["chunks"] == 4
    assert second["new_embeddings"] == 0
    assert second["cached_embeddings"] == 4
    assert client.texts == 0
    assert second_chunk_ids[0] in service.registry.get_chunk_embeddings(cache_key, second_chunk_ids)


def test_rebuild_uses_fp16_embedding_cache_when_configured(tmp_path):
    client = CountingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    service.config.embedding_cache_dtype = "fp16"
    index = service.create_index({"name": "Embedding Cache fp16", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")

    service.index_document_now(alpha_doc["id"])
    cache_key = service._embedding_cache_key("fake-embed", query=False)
    with service.registry.connect() as conn:
        row = conn.execute(
            "SELECT dim, embedding_blob FROM chunk_embeddings WHERE model = ?",
            (cache_key,),
        ).fetchone()

    assert row is not None
    assert len(row["embedding_blob"]) == int(row["dim"]) * 2

    client.calls = 0
    client.texts = 0
    cached = service.rebuild_index_now(index["id"])

    assert cached["new_embeddings"] == 0
    assert cached["cached_embeddings"] == 1
    assert client.texts == 0
    result = service.retrieve("alpha", index_ids=[index["id"]], top_k=1, score_threshold=0.0)
    assert result["results"]


def test_document_pagination_and_filters(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Paged", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha-policy.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta-guide.txt", b"beta text", "text/plain")
    service.registry.set_document_status(alpha_doc["id"], "indexed")
    service.registry.set_document_status(beta_doc["id"], "failed", error="boom")

    first_page = service.registry.list_documents_page(index["id"], limit=1, offset=0)
    second_page = service.registry.list_documents_page(index["id"], limit=1, offset=1)
    query_page = service.registry.list_documents_page(index["id"], query="alpha")
    unindexed_page = service.registry.list_documents_page(index["id"], status="unindexed")

    assert first_page["total"] == 2
    assert len(first_page["documents"]) == 1
    assert len(second_page["documents"]) == 1
    assert query_page["total"] == 1
    assert query_page["documents"][0]["filename"] == "alpha-policy.txt"
    assert unindexed_page["total"] == 1
    assert unindexed_page["documents"][0]["filename"] == "beta-guide.txt"


def test_document_search_uses_document_fts_and_syncs_updates(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("Document FTS")
    alpha_doc = registry.create_document(index["id"], filename="alpha-policy.txt")
    beta_doc = registry.create_document(index["id"], filename="beta-guide.txt")
    registry.set_document_status(beta_doc["id"], DocumentStatus.FAILED, error="needle failure")

    statements: list[str] = []
    with registry.connect() as conn:
        conn.set_trace_callback(statements.append)
        try:
            page = registry.list_documents_page(index["id"], query="needle")
        finally:
            conn.set_trace_callback(None)

    joined_statements = "\n".join(statements)
    assert page["total"] == 1
    assert page["documents"][0]["id"] == beta_doc["id"]
    assert "document_fts" in joined_statements
    assert "MATCH" in joined_statements
    assert "LIKE" not in joined_statements

    assert registry.list_documents_page(index["id"], query="alpha")["documents"][0]["id"] == alpha_doc["id"]
    registry.soft_delete_document(beta_doc["id"])
    assert registry.list_documents_page(index["id"], query="needle")["total"] == 0


def test_registry_status_enums_store_string_values(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("Statuses")
    document = registry.create_document(index["id"], filename="alpha.txt")

    registry.set_document_status(document["id"], DocumentStatus.EXTRACTING)
    document_row = registry.get_document(document["id"])
    document_page = registry.list_documents_page(index["id"], status=DocumentStatus.EXTRACTING)

    assert document_row["status"] == "extracting"
    assert document_page["total"] == 1
    assert document_page["documents"][0]["id"] == document["id"]

    job = registry.create_job("index_document", index_id=index["id"], document_id=document["id"])
    registry.update_job(job["id"], JobStatus.RUNNING, message="test")
    jobs = registry.list_jobs(statuses=[JobStatus.RUNNING])

    assert registry.get_job(job["id"])["status"] == "running"
    assert [item["id"] for item in jobs] == [job["id"]]

    canceled_jobs = registry.request_cancel_jobs(job_id=job["id"])

    assert canceled_jobs[0]["status"] == "cancel_requested"
    assert registry.job_cancel_requested(job["id"]) is True


def test_registry_shared_connection_handles_parallel_operations(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    index = registry.create_index("Parallel Registry")
    worker_count = 4
    iterations = 20
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for item in range(iterations):
                document = registry.create_document(index["id"], filename=f"{worker_id}-{item}.txt")
                registry.set_document_status(document["id"], DocumentStatus.INDEXED)
                job = registry.create_job("index_document", index_id=index["id"], document_id=document["id"])
                registry.update_job(job["id"], JobStatus.COMPLETED, finished=True)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not [thread for thread in threads if thread.is_alive()]
    assert not errors
    assert registry.list_documents_page(index["id"], limit=200)["total"] == worker_count * iterations
    assert len(registry.list_jobs(statuses=[JobStatus.COMPLETED], limit=200)) == worker_count * iterations


def test_registry_nested_connect_rolls_back_outer_transaction(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    now = "2026-05-17T00:00:00"

    with pytest.raises(RuntimeError, match="outer boom"):
        with registry.connect() as conn:
            conn.execute(
                """
                INSERT INTO indexes
                    (id, name, description, embedding_model, chunk_size, chunk_overlap,
                     index_type, created_at, updated_at)
                VALUES (?, ?, '', '', 80, 10, 'auto', ?, ?)
                """,
                ("nested_idx", "Nested", now, now),
            )
            with registry.connect() as nested:
                nested.execute(
                    """
                    INSERT INTO documents
                        (id, index_id, filename, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("nested_doc", "nested_idx", "nested.txt", now, now),
                )
            raise RuntimeError("outer boom")

    assert registry.get_index("nested_idx") is None
    assert registry.get_document("nested_doc") is None


def test_rag_service_metrics_are_thread_safe(tmp_path):
    service = make_service(tmp_path)
    worker_count = 4
    iterations = 1000
    elapsed = 0.001
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(iterations):
                service._record_metric("stress", elapsed, {"worker": worker_id})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not [thread for thread in threads if thread.is_alive()]
    assert not errors
    metric = service.metrics_snapshot()["stress"]
    assert metric["count"] == worker_count * iterations
    assert metric["total_sec"] == pytest.approx(worker_count * iterations * elapsed)
    assert metric["avg_sec"] == pytest.approx(elapsed)
    assert metric["max_sec"] == pytest.approx(elapsed)
    assert isinstance(metric["last_extra"], dict)


def test_single_document_rebuild_debounce_batches_concurrent_jobs(tmp_path):
    service = make_service(tmp_path)
    service.config.rebuild_debounce_sec = 0.05
    index = service.create_index({"name": "Debounce", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text", "text/plain")
    barrier = threading.Barrier(2)
    rebuilds = 0
    rebuild_lock = threading.Lock()
    original_chunks = service.index_document_chunks_now
    original_rebuild = service.rebuild_index_now

    def synced_chunks(*args, **kwargs):
        result = original_chunks(*args, **kwargs)
        barrier.wait(timeout=5)
        return result

    def counting_rebuild(*args, **kwargs):
        nonlocal rebuilds
        with rebuild_lock:
            rebuilds += 1
        return original_rebuild(*args, **kwargs)

    service.index_document_chunks_now = synced_chunks
    service.rebuild_index_now = counting_rebuild
    errors = []

    def run(document_id, job_id):
        try:
            service.index_document_now(document_id, job_id=job_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(alpha_doc["id"], "job-alpha")),
        threading.Thread(target=run, args=(beta_doc["id"], "job-beta")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert rebuilds == 1
    assert service.registry.get_document(alpha_doc["id"])["status"] == "indexed"
    assert service.registry.get_document(beta_doc["id"])["status"] == "indexed"


def test_bulk_document_indexing_rebuilds_once(tmp_path):
    client = CountingEmbeddingClient()
    service = make_service(tmp_path, embedding_client=client)
    index = service.create_index({"name": "Bulk", "embedding_model": "fake-embed"})
    alpha_doc = service.save_upload(index["id"], "alpha.txt", b"alpha text " * 80, "text/plain")
    beta_doc = service.save_upload(index["id"], "beta.txt", b"beta text " * 80, "text/plain")
    rebuilds = 0
    statuses_before_rebuild = []
    original_rebuild = service.rebuild_index_now

    def counting_rebuild(*args, **kwargs):
        nonlocal rebuilds
        rebuilds += 1
        statuses_before_rebuild.append(
            [
                service.registry.get_document(alpha_doc["id"])["status"],
                service.registry.get_document(beta_doc["id"])["status"],
            ]
        )
        return original_rebuild(*args, **kwargs)

    service.rebuild_index_now = counting_rebuild

    result = service.index_documents_now(index["id"], [alpha_doc["id"], beta_doc["id"]])

    assert rebuilds == 1
    assert statuses_before_rebuild == [["vectorizing", "vectorizing"]]
    assert len(result["indexed"]) == 2
    assert result["failed"] == []
    assert service.registry.get_document(alpha_doc["id"])["status"] == "indexed"
    assert service.registry.get_document(beta_doc["id"])["status"] == "indexed"
    assert service.retrieve("alpha", index_ids=[index["id"]], top_k=5)["results"]
    assert service.retrieve("beta", index_ids=[index["id"]], top_k=5)["results"]


def test_indexing_job_auto_clears_ocr_gpu_cache(tmp_path):
    extraction_module._EASYOCR_READERS.clear()
    extraction_module._EASYOCR_READERS[(("ru", "en"), True, "0", "/tmp/models")] = object()
    service = make_service(tmp_path)
    index = service.create_index({"name": "OCR Cleanup", "embedding_model": "fake-embed"})
    job = service.registry.create_job("index_document", index_id=index["id"], document_id="doc-1")

    service.run_job(job["id"], lambda job_id=None: {"ok": True})

    stored = service.registry.get_job(job["id"])
    payload = json.loads(stored["result_json"])
    assert stored["status"] == "completed"
    assert extraction_module.easyocr_reader_count() == 0
    assert payload["ocr_gpu_cache"]["auto"] is True
    assert payload["ocr_gpu_cache"]["readers_before"] == 1


def test_delete_index_removes_documents_chunks_and_storage(tmp_path):
    service = make_service(tmp_path)
    index = service.create_index({"name": "Delete Me", "embedding_model": "fake-embed"})
    document = service.save_upload(index["id"], "alpha.txt", b"alpha text", "text/plain")
    service.registry.replace_document_chunks(index["id"], document["id"], ["alpha chunk"])
    job = service.registry.create_job("index_document", index_id=index["id"], document_id=document["id"])
    service.registry.update_job(job["id"], "running", message="test")
    index_dir = service.config.indexes_path / index["id"]
    index_dir.mkdir(parents=True)
    (index_dir / "vectors.faiss").write_text("x", encoding="utf-8")

    deleted = service.delete_index(index["id"])

    assert deleted["id"] == index["id"]
    assert service.registry.get_index(index["id"]) is None
    assert service.registry.list_documents(index["id"]) == []
    assert service.registry.active_chunks(index["id"]) == []
    assert not index_dir.exists()
    assert not (service.config.uploads_path / index["id"]).exists()
    assert service.registry.get_job(job["id"])["status"] == "cancel_requested"


def test_deep_batch_packing_splits_by_budget():
    docs = [
        {
            "source": f"doc-{index}.txt",
            "locator": f"абз. {index}",
            "quote": f"Факт {index}",
            "score": 0.9,
            "text": "важный контекст " * 120,
        }
        for index in range(8)
    ]

    batches, dropped = rag_server._pack_analysis_batches(
        docs,
        batch_chars=1800,
        max_batches=3,
    )

    assert 1 < len(batches) <= 3
    assert dropped > 0


def test_openwebui_filter_injects_context_without_generating(monkeypatch):
    rag_filter = Filter()

    called = {"retrieve": 0}

    def fake_post(path, payload):
        called["retrieve"] += 1
        assert path == "/retrieve"
        assert payload["index_ids"] == []
        assert payload["top_k"] == 70
        assert payload["score_threshold"] == 0.5
        return {
            "results": [
                {
                    "source": "alpha.txt",
                    "chunk_no": 0,
                    "score": 0.91,
                    "locator": "абз. 1",
                    "quote": "Alpha source text.",
                    "text": "[абз. 1] Alpha source text.",
                }
            ]
        }

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "What is alpha?"}]}

    result = asyncio.run(
        rag_filter.inlet(body, __metadata__={"chat_id": "chat-1"})
    )

    assert called["retrieve"] == 1
    assert result["messages"][0]["role"] == "user"
    assert "Контекст RAG для ответа:" in result["messages"][0]["content"]
    assert "Пачка 1/1" in result["messages"][0]["content"]
    assert "Alpha source text" in result["messages"][0]["content"]
    assert "Локатор: абз. 1" in result["messages"][0]["content"]
    assert "Цитата: «Alpha source text.»" in result["messages"][0]["content"]
    assert "<details>" in result["messages"][0]["content"]
    assert "<summary>Источники</summary>" in result["messages"][0]["content"]
    assert "Документ: alpha.txt; Локатор: абз. 1" in result["messages"][0]["content"]
    assert result["messages"][0]["content"].endswith("What is alpha?")

    result["messages"].append({"role": "assistant", "content": "Alpha answer."})
    outlet = asyncio.run(
        rag_filter.outlet(result, __metadata__={"chat_id": "chat-1"})
    )
    assert outlet["messages"][-1]["content"] == "Alpha answer."


def test_openwebui_filter_warns_when_sidecar_admin_config_is_unavailable(monkeypatch, capsys):
    rag_filter = Filter()
    rag_filter.valves.retrieval_top_k = 3
    rag_filter.valves.min_relevance_score = 0.11
    admin_available = {"value": False}
    retrieve_payloads = []
    emitted = []

    def fake_get(path, *args, **kwargs):
        assert path == "/config"
        if not admin_available["value"]:
            raise RuntimeError("RAG sidecar HTTP 401: api key required")
        return {"rag_enabled": True, "retrieval_top_k": 5}

    def fake_post(path, payload):
        assert path == "/retrieve"
        retrieve_payloads.append(payload)
        return {"results": []}

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(rag_filter, "_get_json", fake_get)
    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "alpha?"}]}

    asyncio.run(rag_filter.inlet(body, __event_emitter__=emit))

    notifications = [event for event in emitted if event.get("type") == "notification"]
    assert retrieve_payloads[-1]["top_k"] == 3
    assert retrieve_payloads[-1]["score_threshold"] == 0.11
    assert rag_filter._admin_config_unavailable is True
    assert notifications
    warning = notifications[-1]["data"]
    assert warning["type"] == "warning"
    assert "/config" in warning["content"]
    assert "api key" in warning["content"]
    assert "собственных valves" in warning["content"]
    log_output = capsys.readouterr().out
    assert "ZI_RAG sidecar /config недоступен" in log_output

    emitted.clear()
    retrieve_payloads.clear()
    asyncio.run(rag_filter.inlet(body, __event_emitter__=emit))
    assert retrieve_payloads[-1]["top_k"] == 3
    assert not [event for event in emitted if event.get("type") == "notification"]
    assert capsys.readouterr().out == ""

    admin_available["value"] = True
    rag_filter._admin_config_loaded_at -= 6.0
    rag_filter._admin_config_warned_at -= 61.0
    emitted.clear()
    retrieve_payloads.clear()
    asyncio.run(rag_filter.inlet(body, __event_emitter__=emit))

    assert retrieve_payloads[-1]["top_k"] == 5
    assert rag_filter._admin_config_unavailable is False
    assert rag_filter._admin_config_warned_at == 0.0
    assert not [event for event in emitted if event.get("type") == "notification"]
    assert "доступен снова" in capsys.readouterr().out


def test_openwebui_filter_indexes_attachment_then_retrieves(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_sidecar_admin_config", lambda: {})

    async def fake_files(file_items, user, **kwargs):
        assert [item["id"] for item in file_items] == ["file-1"]
        return [
            {
                "filename": "alpha.txt",
                "content_type": "text/plain",
                "content": b"alpha text",
                "external_id": "file-1",
                "metadata": {"id": "file-1", "name": "alpha.txt"},
            }
        ]

    calls = {"multipart": [], "retrieve": None}

    def fake_multipart(path, payload, files, *args):
        calls["multipart"].append((path, payload, files, args))
        assert path == "/chat-attachments/index"
        assert payload["chat_id"] == "chat-attach"
        assert payload["files"][0]["id"] == "file-1"
        return {
            "index_id": "owui_chat_chat-attach",
            "indexed_document_ids": ["doc-1"],
            "skipped": [],
            "failed": [],
        }

    def fake_post(path, payload):
        assert path == "/retrieve"
        calls["retrieve"] = payload
        return {"results": []}

    monkeypatch.setattr(rag_filter, "_openwebui_file_payloads", fake_files)
    monkeypatch.setattr(rag_filter, "_post_multipart", fake_multipart)
    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {
        "files": [
            {"type": "file", "id": "file-1", "name": "alpha.txt"},
            {"type": "file", "id": "img-1", "name": "img.png", "content_type": "image/png"},
        ],
        "metadata": {
            "chat_id": "chat-attach",
            "files": [{"type": "file", "id": "file-1", "name": "alpha.txt"}],
        },
        "messages": [{"role": "user", "content": "alpha?"}],
    }

    result = asyncio.run(rag_filter.inlet(body, __metadata__=body["metadata"], __user__={"id": "u1"}))

    assert calls["retrieve"]["extra_index_ids"] == ["owui_chat_chat-attach"]
    assert result["files"] == [{"type": "file", "id": "img-1", "name": "img.png", "content_type": "image/png"}]
    assert "files" not in result["metadata"]


def test_openwebui_filter_reuses_chat_attachment_index_without_new_files(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_sidecar_admin_config", lambda: {})
    called = {}

    def fake_post(path, payload):
        called["payload"] = payload
        return {"results": []}

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "next question"}]}

    asyncio.run(rag_filter.inlet(body, __metadata__={"chat_id": "chat-attach"}))

    assert called["payload"]["extra_index_ids"] == ["owui_chat_chat-attach"]


def test_openwebui_filter_batches_large_retrieval_result(monkeypatch):
    rag_filter = Filter()
    rag_filter.valves.retrieval_top_k = 25
    rag_filter.valves.max_context_chars = 4000
    rag_filter.valves.context_batch_chars = 1000
    rag_filter.valves.max_context_batches = 2
    rag_filter.valves.max_compact_sources = 4
    rag_filter.valves.adaptive_score_margin = 0.5

    docs = [
        {
            "source": f"doc-{index}.pdf",
            "chunk_no": index,
            "score": 0.9 - index * 0.01,
            "locator": f"стр. {index}; абз. 1",
            "quote": f"Факт {index}",
            "text": f"[стр. {index}; абз. 1] " + ("важный контекст " * 45),
        }
        for index in range(1, 13)
    ]

    def fake_post(path, payload):
        assert path == "/retrieve"
        assert payload["top_k"] == 25
        assert payload["score_threshold"] == 0.5
        return {"results": docs}

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "Собери все факты"}]}

    result = asyncio.run(rag_filter.inlet(body, __metadata__={"chat_id": "chat-2"}))
    content = result["messages"][0]["content"]

    assert "Найдено RAG-фрагментов: 12." in content
    assert "Пачка 1/2" in content
    assert "Пачка 2/2" in content
    assert "Компактная пачка остальных источников" in content
    assert "doc-1.pdf" in content
    assert "doc-12.pdf" not in content
    assert len(content) < 8000


def test_openwebui_filter_dedupes_and_filters_weak_noise(monkeypatch):
    rag_filter = Filter()
    rag_filter.valves.retrieval_top_k = 20
    rag_filter.valves.min_relevance_score = 0.72
    rag_filter.valves.adaptive_score_margin = 0.04
    rag_filter.valves.max_prompt_chunks = 10

    docs = [
        {
            "source": "passwords.docx",
            "chunk_no": 1,
            "score": 0.84,
            "locator": "абз. 10",
            "quote": "Пароль учетной записи должен быть сложным.",
            "text": "[абз. 10] Пароль учетной записи должен быть сложным.",
        },
        {
            "source": "passwords.docx",
            "chunk_no": 2,
            "score": 0.83,
            "locator": "абз. 10",
            "quote": "Пароль учетной записи должен быть сложным.",
            "text": "[абз. 10] Пароль учетной записи должен быть сложным.",
        },
        {
            "source": "unrelated.docx",
            "chunk_no": 3,
            "score": 0.79,
            "locator": "абз. 2",
            "quote": "Общие положения документа.",
            "text": "[абз. 2] Общие положения документа.",
        },
        {
            "source": "weak.docx",
            "chunk_no": 4,
            "score": 0.70,
            "locator": "абз. 3",
            "quote": "Пароль упомянут вскользь.",
            "text": "[абз. 3] Пароль упомянут вскользь.",
        },
    ]

    monkeypatch.setattr(rag_filter, "_post_json", lambda path, payload: {"results": docs})
    body = {"messages": [{"role": "user", "content": "пароль учетная запись"}]}

    result = asyncio.run(rag_filter.inlet(body, __metadata__={"chat_id": "chat-3"}))
    content = result["messages"][0]["content"]

    assert content.count("[1] Документ: passwords.docx") == 1
    assert "unrelated.docx" not in content
    assert "weak.docx" not in content


def test_filter_lexical_backfill_keeps_specific_requirements(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_sidecar_admin_config", lambda: {})
    rag_filter.valves.min_relevance_score = 0.72
    rag_filter.valves.adaptive_score_margin = 0.04
    rag_filter.valves.max_prompt_chunks = 10
    query = (
        "какие требования НМД нарушаются если работник подключает оборудование "
        "одновременно в корпоративную и технологическую сети передачи данных"
    )
    docs = [
        {
            "source": "glossary.xlsb",
            "chunk_no": 1,
            "score": 0.87,
            "text": "КСПД корпоративная сеть передачи данных ТСПД технологическая сеть передачи данных",
        },
        {
            "source": "method.docx",
            "chunk_no": 2,
            "score": 0.82,
            "text": (
                "Требования к средствам контроля доступа. Оборудование, подключаемое "
                "к технологической сети передачи данных, контролируется. Корпоративная "
                "и технологическая сети разделяются."
            ),
        },
    ]

    filtered, stats = rag_filter._filter_docs_for_prompt(query, docs)

    assert stats["score_floor"] > docs[1]["score"]
    assert any(doc["source"] == "method.docx" for doc in filtered)


def test_sidecar_analysis_filter_lexical_backfill_keeps_specific_requirements():
    query = (
        "какие требования НМД нарушаются если работник подключает оборудование "
        "одновременно в корпоративную и технологическую сети передачи данных"
    )
    docs = [
        {
            "source": "glossary.xlsb",
            "chunk_no": 1,
            "score": 0.87,
            "text": "КСПД корпоративная сеть передачи данных ТСПД технологическая сеть передачи данных",
        },
        {
            "source": "method.docx",
            "chunk_no": 2,
            "score": 0.82,
            "text": (
                "Требования к средствам контроля доступа. Оборудование, подключаемое "
                "к технологической сети передачи данных, контролируется. Корпоративная "
                "и технологическая сети разделяются."
            ),
        },
    ]

    filtered, stats = rag_server._filter_analysis_docs(query, docs, min_score=0.72, margin=0.04)

    assert stats["score_floor"] > docs[1]["score"]
    assert any(doc["source"] == "method.docx" for doc in filtered)


def test_filter_analysis_prioritizes_arm_software_policy_over_procedure_noise():
    query = "какие пункты нарушены у работника, который установил на свой рабочий АРМ игры? проверь всё"
    docs = [
        {
            "source": "service-check.docx",
            "chunk_no": 1,
            "score": 0.95,
            "text": (
                "Служебной проверкой установлено: сведения о времени, месте, "
                "обстоятельствах совершения правонарушения и сведения о работнике."
            ),
        },
        {
            "source": "allowed-assets.docx",
            "chunk_no": 2,
            "score": 0.78,
            "text": (
                "АРМ предоставляются работникам с предустановленным стандартным ПО. "
                "Работникам запрещается самостоятельно устанавливать ПО на АРМ. "
                "Допускается использование только разрешенного ПО."
            ),
        },
        {
            "source": "allowed-assets-copy.docx",
            "chunk_no": 3,
            "score": 0.78,
            "text": (
                "АРМ предоставляются работникам с предустановленным стандартным ПО. "
                "Работникам запрещается самостоятельно устанавливать ПО на АРМ. "
                "Допускается использование только разрешенного ПО."
            ),
        },
    ]

    filtered, stats = rag_server._filter_analysis_docs(query, docs, min_score=0.5, margin=0.3)

    assert stats["filtered"] == 2
    assert filtered[0]["source"] == "allowed-assets.docx"
    assert [doc["source"] for doc in filtered].count("allowed-assets-copy.docx") == 0


def test_query_terms_drop_question_noise_after_stemming():
    terms = text_utils.query_terms(
        "какие пункты нарушены у работника, который установил на свой рабочий АРМ игры? проверь всё"
    )

    assert "пункт" not in terms
    assert "котор" not in terms
    assert "прове" not in terms
    assert "арм" in terms
    assert "игры" in terms


def test_text_utils_matches_openwebui_filter_prompt_copy(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_sidecar_admin_config", lambda: {})
    rag_filter.valves.min_relevance_score = 0.72
    rag_filter.valves.adaptive_score_margin = 0.04
    rag_filter.valves.max_prompt_chunks = 3
    rag_filter.valves.min_query_term_hits = 1
    query = "пароль учетная запись доступ"
    docs = [
        {
            "source": "passwords.docx",
            "chunk_no": 1,
            "score": 0.9,
            "locator": "абз. 1",
            "quote": "Пароль должен быть сложным.",
            "text": "[абз. 1] Пароль учетной записи должен быть сложным.",
        },
        {
            "source": "passwords.docx",
            "chunk_no": 2,
            "score": 0.88,
            "locator": "абз. 2",
            "quote": "Дубликат.",
            "text": "[абз. 1] Пароль учетной записи должен быть сложным.",
        },
        {
            "source": "access.docx",
            "chunk_no": 3,
            "score": 0.84,
            "locator": "абз. 3",
            "quote": "Доступ отзывается.",
            "text": "[абз. 3] Доступ отзывается при увольнении.",
        },
    ]

    filter_result = rag_filter._filter_docs_for_prompt(query, docs)
    shared_result = text_utils.filter_docs_for_prompt(
        query,
        docs,
        min_score=0.72,
        margin=0.04,
        max_docs=3,
        min_hits=1,
    )

    assert rag_server._filter_analysis_docs is text_utils.filter_analysis_docs
    assert filter_result == shared_result


def test_openwebui_filter_deep_marker_replaces_outlet_answer(monkeypatch):
    rag_filter = Filter()
    called = {}

    def fake_post(path, payload, *args):
        called["path"] = path
        called["payload"] = payload
        called["timeout"] = args[0] if args else None
        return {
            "answer": "Deep final answer\n\n<details>\n<summary>Источники</summary>\n\n1. Документ: a\n</details>",
            "stats": {"batches": 2, "filtered": 5},
        }

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {
        "model": "g4-ultra:latest",
        "messages": [{"role": "user", "content": "/deep полный перечень требований"}],
    }

    result = asyncio.run(
        rag_filter.inlet(
            body,
            __metadata__={"chat_id": "deep-1"},
            __user_valves__={"deep_analysis_enabled": False, "deep_final_answer": True},
        )
    )

    assert called["path"] == "/analyze"
    assert called["payload"]["mode"] == "answer"
    assert called["payload"]["generation_model"] == "g4-ultra:latest"
    assert called["payload"]["top_k"] == 70
    assert called["payload"]["extra_index_ids"] == ["owui_chat_deep-1"]
    assert called["timeout"] == 900
    assert result["messages"][0]["content"].endswith("полный перечень требований")
    assert "Deep final answer" in result["messages"][0]["content"]

    result["messages"].append({"role": "assistant", "content": "placeholder"})
    outlet = asyncio.run(rag_filter.outlet(result, __metadata__={"chat_id": "deep-1"}))

    assert outlet["messages"][-1]["content"].startswith("Deep final answer")


def test_openwebui_filter_deep_failure_falls_back_to_retrieval_context(monkeypatch):
    rag_filter = Filter()
    calls: list[tuple[str, dict[str, Any]]] = []
    docs = [
        {
            "source": "allowed-assets.docx",
            "chunk_no": 1,
            "score": 0.95,
            "locator": "абз. 175 / абз. 176",
            "quote": "Работникам запрещается самостоятельно устанавливать ПО на АРМ.",
            "text": (
                "[абз. 175] АРМ предоставляются работникам с предустановленным стандартным ПО. "
                "[абз. 176] Работникам запрещается самостоятельно устанавливать ПО на АРМ. "
                "[абз. 177] Допускается использование только разрешенного ПО."
            ),
        }
    ]

    monkeypatch.setattr(rag_filter, "_get_json", lambda *args, **kwargs: {})

    def fake_post(path, payload, *args):
        calls.append((path, payload))
        if path == "/analyze":
            raise RuntimeError("Multi-pass analysis timed out")
        if path == "/retrieve":
            return {"results": docs}
        raise AssertionError(path)

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {
        "model": "local-llama",
        "messages": [
            {
                "role": "user",
                "content": "/deep какие пункты нарушены у работника, который установил на свой рабочий АРМ игры?",
            }
        ],
    }

    result = asyncio.run(
        rag_filter.inlet(
            body,
            __metadata__={"chat_id": "deep-fallback"},
            __user_valves__={"deep_analysis_enabled": False, "deep_final_answer": True},
        )
    )

    paths = [path for path, _payload in calls]
    assert paths == ["/analyze", "/retrieve"]
    assert calls[1][1]["top_k"] == 70
    content = result["messages"][0]["content"]
    assert "Контекст RAG для ответа" in content
    assert "Работникам запрещается самостоятельно устанавливать ПО на АРМ" in content


def test_openwebui_filter_emits_deep_progress_from_sidecar_job(monkeypatch):
    rag_filter = Filter()
    called = {"post": [], "stream": 0}

    def fake_post(path, payload, *args):
        called["post"].append(path)
        assert path == "/analyze/jobs"
        return {"id": "job-1", "status": "queued", "events": []}

    def fake_stream(path, *args):
        called["stream"] += 1
        assert path == "/analyze/jobs/job-1/events"
        events = [
            {
                "stage": "batch_start",
                "message": "Пачка 1/2: извлечение фактов",
                "batch": 1,
                "total_batches": 2,
            },
            {
                "stage": "batch_done",
                "message": "Пачка 1/2 готова",
                "batch": 1,
                "total_batches": 2,
                "note_excerpt": "Найдено требование с источником.",
            },
        ]
        yield "analysis", {"id": "job-1", "status": "running", "events": events}
        yield "done", {
            "id": "job-1",
            "status": "completed",
            "events": events,
            "result": {
                "answer": "Deep job answer\n\n<details><summary>Источники</summary>x</details>",
                "stats": {"batches": 2, "filtered": 7},
            },
        }

    emitted = []

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    monkeypatch.setattr(rag_filter, "_iter_sse_events", fake_stream)
    body = {"messages": [{"role": "user", "content": "/deep проанализируй все требования"}]}

    result = asyncio.run(rag_filter.inlet(body, __event_emitter__=emit))

    descriptions = [event.get("data", {}).get("description", "") for event in emitted]
    assert called["post"] == ["/analyze/jobs"]
    assert called["stream"] == 1
    assert any("анализ пачки 1/2" in item for item in descriptions)
    assert any("Найдено требование" in item for item in descriptions)
    assert "Deep job answer" in result["messages"][0]["content"]


def test_openwebui_filter_cancels_sidecar_deep_job_on_task_cancel():
    rag_filter = Filter()
    calls = []

    def fake_post(path, payload, *args):
        calls.append(path)
        if path == "/analyze/jobs":
            return {"id": "job-stop", "status": "running", "events": []}
        if path == "/analyze/jobs/job-stop/cancel":
            return {"id": "job-stop", "status": "cancel_requested", "events": []}
        raise AssertionError(path)

    def fake_get(path, *args):
        assert path == "/analyze/jobs/job-stop"
        return {"id": "job-stop", "status": "running", "events": []}

    def fake_stream(path, *args):
        assert path == "/analyze/jobs/job-stop/events"
        while True:
            time.sleep(0.2)
            yield "analysis", {"id": "job-stop", "status": "running", "events": []}

    async def emit(_event):
        return None

    rag_filter._post_json = fake_post
    rag_filter._get_json = fake_get
    rag_filter._iter_sse_events = fake_stream

    async def run_and_cancel():
        task = asyncio.create_task(
            rag_filter._run_deep_analysis(
                {"query": "deep"},
                60,
                emit,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())

    assert "/analyze/jobs" in calls
    assert "/analyze/jobs/job-stop/cancel" in calls


def test_openwebui_filter_auto_deep_context_mode(monkeypatch):
    rag_filter = Filter()
    called = {}

    def fake_post(path, payload, *args):
        called["path"] = path
        called["payload"] = payload
        return {
            "context": "Deep RAG multi-pass context\n[Пачка 1]\nФакт",
            "stats": {"batches": 1, "filtered": 3},
        }

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "сравни все требования к паролям"}]}

    result = asyncio.run(
        rag_filter.inlet(
            body,
            __metadata__={"chat_id": "deep-2"},
            __user_valves__={"deep_analysis_enabled": True, "deep_final_answer": False},
        )
    )

    assert called["path"] == "/analyze"
    assert called["payload"]["mode"] == "context"
    assert "Deep RAG multi-pass context" in result["messages"][0]["content"]

    result["messages"].append({"role": "assistant", "content": "model answer"})
    outlet = asyncio.run(rag_filter.outlet(result, __metadata__={"chat_id": "deep-2"}))

    assert outlet["messages"][-1]["content"] == "model answer"


def test_openwebui_filter_uses_sidecar_admin_deep_defaults(monkeypatch):
    rag_filter = Filter()
    called = {}

    monkeypatch.setattr(
        rag_filter,
        "_get_json",
        lambda path, *args, **kwargs: {
            "deep_analysis_enabled": True,
            "deep_final_answer": False,
            "deep_top_k": 44,
            "deep_timeout_sec": 123,
            "default_index_ids": ["IB"],
            "deep_trigger_phrases": ["сравни"],
        },
    )

    def fake_post(path, payload, *args):
        called["path"] = path
        called["payload"] = payload
        called["timeout"] = args[0] if args else None
        return {
            "context": "Deep RAG multi-pass context\n[Пачка 1]\nФакт",
            "stats": {"batches": 1, "filtered": 3},
        }

    monkeypatch.setattr(rag_filter, "_post_json", fake_post)
    body = {"messages": [{"role": "user", "content": "сравни требования"}]}

    result = asyncio.run(rag_filter.inlet(body, __metadata__={"chat_id": "deep-admin"}))

    assert called["path"] == "/analyze"
    assert called["payload"]["mode"] == "context"
    assert called["payload"]["top_k"] == 44
    assert called["payload"]["index_ids"] == ["IB"]
    assert called["timeout"] == 123
    assert "Deep RAG multi-pass context" in result["messages"][0]["content"]


def test_openwebui_filter_check_marker_posts_compliance_and_removes_files(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(rag_filter, "_get_json", lambda *args, **kwargs: {})

    async def fake_files(file_items, user, **kwargs):
        assert file_items[0]["id"] == "file-1"
        return [
            {
                "filename": "policy.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content": b"document",
                "external_id": "file-1",
                "metadata": {"id": "file-1", "name": "policy.docx"},
            }
        ]

    called = {}

    def fake_multipart(path, payload, files, *args):
        if path == "/chat-attachments/index":
            return {
                "index_id": "owui_chat_u1",
                "indexed_document_ids": ["doc-1"],
                "skipped": [],
                "failed": [],
            }
        called["path"] = path
        called["payload"] = payload
        called["files"] = files
        called["timeout"] = args[0] if args else None
        return {
            "answer": "Compliance answer\n\n<details><summary>Источники</summary>x</details>",
            "stats": {"sections": 2, "matrix_rows": 3},
        }

    monkeypatch.setattr(rag_filter, "_openwebui_file_payloads", fake_files)
    monkeypatch.setattr(rag_filter, "_post_multipart", fake_multipart)
    body = {
        "model": "g4-ultra:latest",
        "files": [{"type": "file", "id": "file-1", "name": "policy.docx"}],
        "metadata": {"files": [{"type": "file", "id": "file-1", "name": "policy.docx"}]},
        "messages": [{"role": "user", "content": "/check index:IB проверь на соответствие НМД"}],
    }

    result = asyncio.run(rag_filter.inlet(body, __metadata__=body["metadata"], __user__={"id": "u1"}))

    assert called["path"] == "/compliance/analyze"
    assert called["payload"]["nmd_index_ids"] == ["IB"]
    assert called["payload"]["generation_model"] == "g4-ultra:latest"
    assert called["payload"]["top_k"] == 24
    assert called["timeout"] == 1200
    assert "files" not in result
    assert "files" not in result["metadata"]
    assert "Compliance answer" in result["messages"][0]["content"]


def test_openwebui_filter_auto_compliance_requires_files(monkeypatch):
    rag_filter = Filter()
    monkeypatch.setattr(
        rag_filter,
        "_get_json",
        lambda *args, **kwargs: {
            "compliance_auto_enabled": True,
            "compliance_enabled": True,
        },
    )

    calls = {"multipart": 0}

    async def fake_files(file_items, user, **kwargs):
        return [
            {
                "filename": "x.pdf",
                "content_type": "application/pdf",
                "content": b"x",
                "external_id": "f1",
                "metadata": {"id": "f1", "name": "x.pdf"},
            }
        ]

    def fake_multipart(path, payload, files, *args):
        if path == "/chat-attachments/index":
            return {
                "index_id": "owui_chat_u1",
                "indexed_document_ids": ["doc-1"],
                "skipped": [],
                "failed": [],
            }
        calls["multipart"] += 1
        return {"answer": "ok", "stats": {}}

    monkeypatch.setattr(rag_filter, "_openwebui_file_payloads", fake_files)
    monkeypatch.setattr(rag_filter, "_post_multipart", fake_multipart)

    with_file = {
        "files": [{"type": "file", "id": "f1", "name": "x.pdf"}],
        "messages": [{"role": "user", "content": "проверь на соответствие НМД"}],
    }
    asyncio.run(rag_filter.inlet(with_file, __user__={"id": "u1"}))
    assert calls["multipart"] == 1

    without_file = {"messages": [{"role": "user", "content": "проверь на соответствие НМД"}]}
    asyncio.run(rag_filter.inlet(without_file, __user__={"id": "u1"}))
    assert calls["multipart"] == 1


def test_sidecar_compliance_uses_temporary_extraction(monkeypatch, tmp_path):
    service = make_service(tmp_path)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def list_models(self):
            return [{"name": "qwen3.6:latest"}]

        def chat(self, model, messages, temperature=0.1, num_predict=1024):
            content = messages[-1]["content"]
            if "JSON-массив" in content:
                return (
                    '[{"requirement":"использовать сложный пароль",'
                    '"nmd_source":"nmd.docx","nmd_locator":"абз. 1",'
                    '"nmd_quote":"Пароль должен быть сложным",'
                    '"checked_file":"check.txt","checked_locator":"абз. 1",'
                    '"checked_quote":"Пароль сложный","status":"соответствует",'
                    '"risk":"","recommendation":"нет"}]'
                )
            return "## Акт проверки\n\n| № | Статус |\n|---:|---|\n| 1 | соответствует |\n\n<details>\n<summary>Источники</summary>\n\n1. src\n</details>"

    def fake_extract(path, cfg):
        return "[абз. 1] Пароль сложный и соответствует требованиям."

    def fake_retrieve(query, **kwargs):
        return {
            "results": [
                {
                    "source": "nmd.docx",
                    "locator": "абз. 1",
                    "quote": "Пароль должен быть сложным",
                    "text": "[абз. 1] Пароль должен быть сложным.",
                    "score": 0.9,
                    "chunk_no": 1,
                }
            ]
        }

    monkeypatch.setattr(rag_server, "OllamaClient", FakeClient)
    monkeypatch.setattr(rag_server, "extract_text", fake_extract)
    monkeypatch.setattr(service, "retrieve", fake_retrieve)
    payload = rag_server.ComplianceRequest(
        query="проверь на соответствие НМД",
        nmd_index_ids=["IB"],
        generation_model="qwen3.6:latest",
        top_k=5,
        score_threshold=0.0,
    )

    result = rag_server.run_compliance_analysis(
        payload,
        [{"filename": "check.txt", "content_type": "text/plain", "content": b"hello"}],
        cfg=service.config,
        service=service,
    )

    assert result["checked_files"][0]["filename"] == "check.txt"
    assert result["stats"]["sections"] == 1
    assert result["matrix"][0]["status"] == "соответствует"
    assert "Акт проверки" in result["answer"]
    assert not any((service.config.uploads_path).glob("check.txt"))


def test_sidecar_compliance_enforces_wall_clock_deadline(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    service.config.compliance_timeout_sec = 5
    service.config.request_timeout_sec = 3
    now = [0.0]
    monkeypatch.setattr(rag_server.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        rag_server,
        "extract_text",
        lambda _path, _cfg: (
            "[абз. 1] Пароль сложный " + ("alpha " * 180)
            + "\n\n"
            "[абз. 2] Доступ ограничен " + ("beta " * 180)
        ),
    )
    monkeypatch.setattr(
        rag_server,
        "_filter_analysis_docs",
        lambda _query, docs, min_score=0.0, **_kwargs: (
            docs,
            {"raw": len(docs), "filtered": len(docs)},
        ),
    )
    retrieve_calls = {"count": 0}

    def fake_retrieve(_query, **_kwargs):
        retrieve_calls["count"] += 1
        return {
            "results": [
                {
                    "source": "nmd.docx",
                    "locator": "абз. 1",
                    "quote": "Требование",
                    "text": "[абз. 1] Требование alpha beta.",
                    "score": 0.9,
                    "chunk_no": 1,
                }
            ]
        }

    class FakeClient:
        chat_calls = 0
        prompts: list[str] = []

        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout", 0)

        def list_models(self):
            return [{"name": "deadline-model"}]

        def chat(self, model, messages, temperature=0.1, num_predict=1024):
            FakeClient.chat_calls += 1
            FakeClient.prompts.append(messages[-1]["content"])
            if FakeClient.chat_calls == 1:
                now[0] += 1.0
                return "[]"
            now[0] += 10.0
            return "[]"

    monkeypatch.setattr(service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_server, "OllamaClient", FakeClient)
    payload = rag_server.ComplianceRequest(
        query="проверь на соответствие НМД",
        generation_model="deadline-model",
        section_chars=1500,
        max_sections=2,
        top_k=5,
        score_threshold=0.0,
    )

    with pytest.raises(rag_server.HTTPException) as exc_info:
        rag_server.run_compliance_analysis(
            payload,
            [{"filename": "check.txt", "content_type": "text/plain", "content": b"hello"}],
            cfg=service.config,
            service=service,
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Compliance analysis timed out"
    assert retrieve_calls["count"] == 2
    assert FakeClient.chat_calls == 2
    assert all("Собери итоговый акт" not in prompt for prompt in FakeClient.prompts)
