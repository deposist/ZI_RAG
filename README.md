# ZI_RAG — External RAG sidecar for OpenWebUI

[![CI](https://github.com/deposist/ZI_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/deposist/ZI_RAG/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-see%20LICENSE-informational)](LICENSE)

ZI_RAG is a standalone FastAPI service that adds production-grade retrieval-augmented generation (RAG) to OpenWebUI **without patching its sources**. Integration happens through one OpenWebUI Function filter, so upgrades to OpenWebUI do not break the sidecar and the sidecar can be upgraded independently.

> 🇷🇺 Краткое русское резюме — внизу страницы. Полная операционная документация: [`OPENWEBUI_ZI_RAG.md`](OPENWEBUI_ZI_RAG.md).

---

## Why ZI_RAG

Native OpenWebUI Knowledge is fine for small, single-collection setups but quickly hits limits on real corpora: weak retrieval quality on long Russian documents, no hybrid lexical+dense search, no per-chat attachment isolation, no NMD-style compliance workflow, no multi-pass deep analysis. ZI_RAG fixes those without forking OpenWebUI:

- **Hybrid retrieval out of the box.** Dense FAISS (`IndexFlatIP` → `IndexHNSWFlat` auto-switch above 50 000 chunks) fused with SQLite FTS5 BM25 via Reciprocal Rank Fusion. MMR top-K selection, optional cross-encoder rerank, optional LLM query expansion (HyDE / multi-query).
- **Deep multi-pass RAG.** `/analyze` retrieves a wide candidate set, splits it into batches, asks the chat model to extract facts per batch, then synthesizes a final answer. Streamed via SSE so OpenWebUI shows live progress.
- **Compliance Check.** `/compliance/analyze` verifies attached documents against an NMD index with section-level retrieval, JSON matrix, citations and per-file report. Wall-clock-bounded.
- **Per-chat attachment indexing.** Files attached to an OpenWebUI chat are auto-indexed into a per-chat sidecar index `owui_chat_<chat_id>`. Native OpenWebUI Knowledge can stay disabled for those models to avoid double-RAG.
- **Locator-aware citations.** Chunks carry source locators (`пункт 3.1`, `абз. 12`, `стр. 4`, `лист`/`строка` for spreadsheets, message body and supported attachments for `.msg`).
- **Strict static checks.** Codebase passes `ruff`, `mypy --strict` and a 120-case pytest suite (CI-enforced).
- **Russian/English admin UI** with persisted locale, OpenAPI tags/summaries on every endpoint, and a single-file Function filter (`zi_rag_filter.py`) that's also exported as `zi_rag_filter.openwebui.json` for direct import.

---

## Architecture at a glance

```
┌─────────────────────┐    HTTP (X-API-Key)    ┌──────────────────────────────┐
│   OpenWebUI         │ ─────────────────────► │ ZI_RAG sidecar (FastAPI)     │
│                     │                        │                              │
│ Function filter     │ ◄───── results ─────── │  /retrieve  /analyze         │
│ zi_rag_filter.py    │                        │  /compliance/analyze         │
│                     │ ─── multipart upload ─►│  /chat-attachments/index     │
└─────────────────────┘                        │  /indexes /jobs /config /ui  │
                                               └──────────────┬───────────────┘
                                                              │
                                  ┌───────────────────────────┼────────────────────────────┐
                                  │                           │                            │
                          ┌───────▼────────┐       ┌──────────▼──────────┐       ┌─────────▼─────────┐
                          │ Embeddings     │       │ Generation (Ollama  │       │ Storage           │
                          │ Ollama or any  │       │ or OpenAI-compat.)  │       │ FAISS + SQLite    │
                          │ OpenAI-compat. │       │                     │       │ FTS5 + uploads    │
                          └────────────────┘       └─────────────────────┘       └───────────────────┘
```

Source layout:

```
openwebui_zi_rag/         FastAPI service package
  app.py                  app factory: routers, static UI, OpenAPI metadata
  server.py               python -m entry, uvicorn bootstrap, public re-exports
  config.py               pydantic-settings model + env/JSON loader (ZI_RAG_*)
  runtime.py              shared runtime state, dep-injection, API-key guard
  routes/                 admin/indexes/documents/jobs/analyze/compliance/chat-attachments
  services/               health, jobs, prompting, multi-pass + compliance pipelines
  indexing/               extraction, chunking, registry (SQLite), vector_store (FAISS)
  ollama_client.py        thin async/sync HTTP client with separated timeouts
  web/                    static admin UI (vanilla JS + i18n: en/ru)
openwebui_functions/
  zi_rag_filter.py        OpenWebUI Function (single-file filter)
  zi_rag_filter.openwebui.json   importable JSON export, kept in sync
tools/
  build_bundle.py         release bundler (allowlist-based)
  build_filter.py         JSON export regenerator
tests/test_openwebui_zi_rag.py  pytest suite (~120 cases)
```

---

## Install

### From source (development)

```bash
git clone https://github.com/deposist/ZI_RAG.git
cd ZI_RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r openwebui_zi_rag_requirements.txt
python3 -m openwebui_zi_rag
```

The sidecar binds to `127.0.0.1:8766` by default. Open `http://127.0.0.1:8766/ui` for the admin panel.

### From a release bundle

Each release attaches `openwebui_zi_rag_bundle.zip` produced by `tools/build_bundle.py`. The bundle contains the runtime package, the Function filter, the requirements file and the operations docs — nothing else (no caches, no storage, no tests).

```bash
unzip openwebui_zi_rag_bundle.zip -d /opt/zi_rag
cd /opt/zi_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r openwebui_zi_rag_requirements.txt
python3 -m openwebui_zi_rag
```

### System dependencies

- **Python 3.10+** (CI runs 3.12).
- **Tesseract** (`tesseract-ocr`, `tesseract-ocr-rus`) — optional CPU OCR fallback.
- **LibreOffice** (`libreoffice`) — required to extract legacy `.doc`, `.rtf`, `.odt`.
- **CUDA toolkit** — optional, only for GPU OCR via EasyOCR.

The CI workflow (`.github/workflows/ci.yml`) installs `tesseract-ocr tesseract-ocr-rus libreoffice` on Ubuntu and is the canonical reproducible setup.

---

## Wire it into OpenWebUI

1. Open **Admin Panel → Functions → Import**, upload `openwebui_functions/zi_rag_filter.py` (or its JSON twin).
2. Enable the function as a **Filter** for the target model(s).
3. Set valves:
   - `sidecar_url` — `http://host.docker.internal:8766` for Docker-hosted OpenWebUI, `http://127.0.0.1:8766` for native installs.
   - `api_key` — required when the sidecar is reachable from Docker / LAN. Mirror it in the sidecar config.
   - `sync_sidecar_admin_config` — keep on so admins control RAG behavior from the sidecar UI.
4. **Disable native OpenWebUI Knowledge** for the same models. Two RAG layers stacking on the same prompt is the most common cause of "model ignores my context" complaints.
5. In the sidecar UI (`http://127.0.0.1:8766/ui` → **Настройки**), pick the embedding model and a default index.

---

## Indexing your data

There are three ways to put data into ZI_RAG:

### 1. Sidecar admin UI

Drag-and-drop files at `http://127.0.0.1:8766/ui`. Multi-select supports queueing reindex jobs (`В очередь`) or forced reindex (`Принудительно`, cancels active jobs first). Status moves through `queued → extracting → vectorizing → indexed`; the FAISS vector store is rebuilt **once** at the end of a batch, not per document.

### 2. Local folder paths

Add a path under `Add path` (or `POST /indexes/{index_id}/documents/add-path`). Paths are accepted only when their resolved root is inside `allowed_source_roots`; everything else returns `403`. Set the allowlist in admin settings or via env (`ZI_RAG_ALLOWED_SOURCE_ROOTS=/data/docs,/srv/contracts`).

### 3. OpenWebUI chat attachments

When the user attaches files in a chat, the Function filter reads them from OpenWebUI runtime storage and sends them to `/chat-attachments/index`. They land in `owui_chat_<chat_id>` — a per-chat index, separate from your main corpus, scoped by chat id (or `session_id`/`message_id`/`user_id` as fallback). Duplicates are skipped by OpenWebUI file id + content hash; updates soft-delete the previous version.

Supported file types: `.txt .md .json .csv .log .htm .html .xml .doc .docx .odt .rtf .pdf .msg .xls .xlsx .xlsm .xlsb`. Images and OpenWebUI collections are left for OpenWebUI.

---

## Configuration

Configuration sources are merged in this order (later wins): defaults → env (`ZI_RAG_*`) → `openwebui_zi_rag_storage/config.json` → `.env`. Most settings are also editable live from the admin UI; the UI persists them to `config.json`.

A few of the most common keys (full list in `openwebui_zi_rag/config.py`):

| Key | Default | Purpose |
| --- | --- | --- |
| `storage_dir` | `./openwebui_zi_rag_storage` | Where uploads, FAISS, SQLite live. |
| `ollama_base_url` | `http://127.0.0.1:11434` | Ollama endpoint for embeddings and generation. |
| `api_key` | empty | Sets `X-API-Key` requirement for non-health endpoints. |
| `require_api_key_localhost` | `false` | Strict mode for multi-user hosts. |
| `allowed_source_roots` | `[]` | Allowlist for filesystem path indexing. |
| `embedding_provider` | `ollama` | Or `openai` for an OpenAI-compatible endpoint. |
| `embedding_model` | empty | Embedding model name. Empty = no embedding work runs. |
| `chunk_size` / `chunk_overlap` | `1200` / `120` | Text chunker budget (characters). |
| `embedding_batch_size` | `16` | Embedding request batch size. |
| `embedding_cache_dtype` | `fp32` | `fp32` or `fp16` (cache only; FAISS stays float32). |
| `top_k` | `8` | Default retrieval k for `/retrieve`. |
| `score_threshold` | `0.50` | Minimum transformed cosine score. |
| `retrieval_top_k` | `70` | Pre-filter k for the chat filter. |
| `adaptive_score_margin` | `0.20` | Max distance from best score kept in prompt. |
| `max_prompt_chunks` | `24` | Hard cap on chunks injected into the chat prompt. |
| `index_type` | `auto` | `auto` / `flat` / `hnsw`. `auto` switches at `hnsw_threshold_chunks`. |
| `hnsw_threshold_chunks` | `50000` | When `auto` upgrades Flat → HNSW. |
| `hnsw_m` / `hnsw_ef_construction` / `hnsw_ef_search` | `32` / `200` / `128` | HNSW knobs. |
| `query_expansion_enabled` | `false` | LLM-based HyDE / multi-query expansion. |
| `rerank_enabled` | `false` | Cross-encoder rerank via `/rerank`. |
| `deep_analysis_enabled` | `true` | Auto multi-pass for trigger phrases. |
| `deep_final_answer` | `true` | Sidecar produces the final answer (filter rewrites the assistant message). |
| `deep_force_all` | `false` | Force multi-pass for every question. |
| `deep_top_k` | `70` | Wide retrieval for deep mode. |
| `deep_timeout_sec` | `900` | Synchronous deep run cap. |
| `compliance_enabled` / `compliance_auto_enabled` | `true` / `true` | Compliance Check master/auto switches. |
| `compliance_index_ids` | `[]` | NMD index(es) used for `/check`. |
| `chat_attachments_enabled` | `true` | Per-chat attachment indexing. |
| `chat_attachment_index_prefix` | `owui_chat_` | Prefix for per-chat indexes. |
| `enable_ocr` | `false` | OCR for image-only PDFs. |
| `ocr_engine` | `easyocr` | Or `tesseract` for the CPU path. |
| `ocr_gpu` | `true` | EasyOCR on CUDA. |
| `connect_timeout_sec` / `request_timeout_sec` / `stream_idle_timeout_sec` | `10.0` / `120.0` / `120.0` | Separate Ollama HTTP timeouts. |

Every key has a matching `ZI_RAG_<UPPER_SNAKE>` environment variable. Examples:

```bash
ZI_RAG_STORAGE_DIR=/srv/zi-rag
ZI_RAG_API_KEY=$(openssl rand -hex 32)
ZI_RAG_ALLOWED_SOURCE_ROOTS=/data/docs,/srv/contracts
ZI_RAG_OLLAMA_BASE_URL=http://127.0.0.1:11434
ZI_RAG_EMBEDDING_PROVIDER=ollama
ZI_RAG_EMBEDDING_MODEL=bge-m3
ZI_RAG_INDEX_TYPE=auto
ZI_RAG_ENABLE_OCR=true
```

---

## Chat usage

In an OpenWebUI chat with the filter enabled:

| Trigger | Effect |
| --- | --- |
| Plain question | Fast hybrid retrieval, top-K chunks injected into the prompt. |
| Auto-deep trigger phrases (`сравни`, `найди противоречия`, `все требования`, …) | Multi-pass `/analyze` if `deep_analysis_enabled`. |
| `/deep <question>` | Force multi-pass for this turn. |
| `/check <question>` | Run Compliance Check on the current attachments. |
| `/check index:IB <question>` | Compliance Check using `IB` as the NMD index. |

Trigger phrases are configurable in **Настройки → Auto deep / Compliance**.

---

## REST API

All non-health endpoints require `X-API-Key: <api_key>` once `api_key` is set. Public liveness probes hit `/health`; container probes that need full diagnostics use `/health/full` (auth-required).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Public liveness (no abs paths, no metrics). |
| GET | `/health/full` | Full diagnostics + embedding-dimension warnings. |
| GET | `/metrics` | Prometheus-style metrics. |
| GET / PUT | `/config` | Read or update sidecar config (live). |
| GET | `/ollama/models`, `/embedding/models` | Discoverable model lists. |
| POST | `/ocr/cache/clear` | Clear OCR result cache. |
| GET / POST / DELETE | `/indexes` , `/indexes/{id}` | Index CRUD. |
| POST | `/indexes/{id}/rebuild` | Rebuild FAISS only (reuses chunks). |
| GET | `/indexes/{id}/documents` | Paginated document listing. |
| POST | `/indexes/{id}/documents/upload` | Single-file multipart upload. |
| POST | `/indexes/{id}/documents/upload-batch` | Multipart batch upload. |
| POST | `/indexes/{id}/documents/add-path` | Index a local path (allowlist-checked). |
| DELETE / POST | `/indexes/{id}/documents/{doc_id}` , `/delete` | Delete by id or batch. |
| POST | `/indexes/{id}/documents/{doc_id}/reindex` , `/reindex` | Queue reindex (single / batch). |
| POST | `/retrieve` | Hybrid retrieval with optional rerank/expansion. |
| POST | `/analyze` | Synchronous multi-pass deep RAG. |
| POST | `/analyze/jobs` | Async deep job (poll or SSE-stream). |
| GET | `/analyze/jobs/{id}` | Job snapshot. |
| GET | `/analyze/jobs/{id}/events` | SSE event stream. |
| POST | `/analyze/jobs/{id}/cancel` | Cooperative cancel. |
| POST | `/compliance/analyze` | Document-vs-NMD verification. |
| POST | `/chat-attachments/index` | Index OpenWebUI chat attachments. |
| GET | `/jobs`, `/jobs/{id}` | Indexing job queue. |
| POST | `/jobs/{id}/cancel`, `/indexes/{id}/jobs/cancel` | Cancel index jobs. |
| GET | `/ui` | Admin web UI (static). |
| GET | `/openapi.json`, `/docs`, `/redoc` | FastAPI-provided. |

Quick smoke test once the server is up:

```bash
curl -s http://127.0.0.1:8766/health | jq .

curl -s -H "X-API-Key: $ZI_RAG_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query":"что такое КСПД","top_k":5}' \
     http://127.0.0.1:8766/retrieve | jq '.results[0]'
```

---

## How retrieval works (short version)

1. **Chunking.** Documents are split with structure-aware boundaries (headings, paragraph breaks, numbered clauses) targeting `chunk_size` chars, `chunk_overlap` chars overlap.
2. **Indexing.** Embeddings are written to FAISS (`Flat` or `HNSW`); the same chunks are mirrored into SQLite FTS5 for BM25.
3. **Query expansion** (optional). LLM produces HyDE / multi-query variants; user-defined `query_synonyms` add domain-specific aliases (КСПД/ТСПД bundled).
4. **Hybrid search.** Dense top-K (`retrieval_top_k`) + BM25 top-K, fused via Reciprocal Rank Fusion. BM25-only hits get `score = 1 / (1 + max(0, fts_rank − 1))` — no hardcoded 0.72 plateau.
5. **Filter.** Score floor (`score_threshold`), adaptive margin, query-term-hit lexical filter, MMR diversification.
6. **Optional rerank.** Cross-encoder via OpenAI-compatible `/rerank` endpoint.
7. **Prompt packing.** `max_prompt_chunks` injected as full text in the first batch; the rest go into compact `<details>Источники</details>` lines so OpenWebUI doesn't truncate the prompt.

For deep mode the same retrieval runs with `deep_top_k`, then chunks are split into `deep_max_batches` × `deep_batch_chars` batches; each batch produces structured facts; the final pass synthesizes the answer.

---

## Deep RAG and Compliance

**Deep RAG** (`/analyze`, also auto-triggered for analytical phrases): wide retrieval → batched fact extraction → final synthesis. Streams progress via SSE — the OpenWebUI filter shows live status (`анализ пачки 3/8`, `финальный синтез`, …).

**Compliance Check** (`/compliance/analyze`, command `/check`): extracts attached files (`.doc .docx .xls .xlsx .pdf .msg` + text-like) into a temp dir, splits into sections (`compliance_section_chars`, capped by `compliance_max_sections`), retrieves NMD requirements per section, asks the model to produce findings, then renders a JSON matrix + per-file report. Temporary checked files are deleted after analysis and **never** added to permanent indexes (unless chat-attachment indexing is on, in which case they're also saved to the per-chat index for follow-ups).

Both modes need a generation model. There is **no production fallback**: pick one in admin UI, pass `generation_model` in the request, or set `ZI_RAG_DEEP_GENERATION_MODEL` / `ZI_RAG_COMPLIANCE_GENERATION_MODEL`. If the model is missing from `/api/tags`, the sidecar returns `409` with the available list (or `502` if `/api/tags` itself is down).

---

## OCR

PDFs with a text layer are read directly. Image-only pages use OCR when `enable_ocr=true`. Recommended GPU defaults:

```ini
ocr_engine        = easyocr
ocr_gpu           = true
ocr_languages     = rus+eng
ocr_model_storage_dir = <storage_dir>/easyocr_models
```

Pass `ZI_RAG_GPU` to bind a specific CUDA device. Set `ocr_engine=tesseract` only when you explicitly want the CPU path. After enabling OCR, reindex existing image-only PDFs.

---

## Storage layout

```
openwebui_zi_rag_storage/
  config.json            persisted admin settings
  registry.sqlite        documents, jobs, FTS5
  uploads/               raw uploaded files
  indexes/<index_id>/    vectors.faiss + vector_map.json + metadata
  easyocr_models/        downloaded EasyOCR weights (when OCR is on)
```

Move with `ZI_RAG_STORAGE_DIR=/some/where`. The whole `openwebui_zi_rag_storage/` tree is git-ignored — never commit it.

In-process FAISS cache: LRU of size 32. Indexes are evicted when deleted, replaced when rebuilt, and cleared whenever the live config refreshes (storage paths might have changed). To clear manually, restart the sidecar.

---

## Build, test, release

```bash
ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/
mypy openwebui_zi_rag/ --strict
pytest tests/test_openwebui_zi_rag.py --no-header -x

python3 tools/build_bundle.py --output dist/openwebui_zi_rag_bundle.zip
```

CI mirrors the exact commands above. Releases are cut by pushing a tag:

```bash
git tag v0.9.1
git push --tags
```

`.github/workflows/release.yml` rebuilds `openwebui_zi_rag_bundle.zip`, attaches it to a GitHub Release, and uses `RELEASE_NOTES_<version>.md` as the body when present.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/retrieve` returns empty `results` even for known content | Embedding model not set, or its dimension mismatches existing indexes | Set `embedding_model`; check `/health/full` `embedding_model_dimension.warnings`; force-reindex affected indexes. |
| Sidecar 401/403 from OpenWebUI Function | `api_key` mismatch between filter valves and sidecar config | Sync `api_key` in both places. Restart the filter from OpenWebUI Functions panel. |
| `/analyze` or `/compliance/analyze` returns 409 | Generation model not selected or missing from Ollama | Pick a model in admin UI, or `ollama pull <model>`. |
| `add-path` returns 403 | Path is outside `allowed_source_roots` | Add the parent directory to the allowlist. |
| Deep job 410 after sidecar restart | Deep jobs are in-memory only | OpenWebUI filter falls back to synchronous `/analyze` automatically. |
| OpenWebUI shows the same context twice | Native Knowledge still enabled for that model | Disable native Knowledge for models that use the ZI_RAG filter. |
| Image-only PDFs return empty text | `enable_ocr=false` | Enable OCR, ensure Tesseract or EasyOCR is installed, reindex. |

---

## Compatibility

- **Python** 3.10+ (CI: 3.12).
- **OpenWebUI** with the Function API (0.9.0+).
- **Ollama** for embeddings/generation, or any OpenAI-compatible embeddings endpoint (Giga, llama.cpp, OpenAI proper).
- **FAISS-CPU**. GPU OCR via EasyOCR + CUDA optional; Tesseract supported on CPU.

---

## License

See [`LICENSE`](LICENSE).

---

## 🇷🇺 Краткое резюме

ZI_RAG — внешний RAG-сайдкар для OpenWebUI. Не патчит исходники OpenWebUI: всё подключается через одну Function (`openwebui_functions/zi_rag_filter.py`). Поверх FAISS+BM25 даёт гибридный поиск, multi-pass deep-анализ (`/deep`), проверку документов на НМД (`/check`), индексацию вложений чата в отдельный per-chat индекс. UI на двух языках, OCR на GPU/CPU, поддержка `.doc/.docx/.xls/.xlsx/.pdf/.msg/.rtf/.odt/.txt/.md` и т.п.

Запуск:

```bash
pip install -r openwebui_zi_rag_requirements.txt
python3 -m openwebui_zi_rag
```

Админка: http://127.0.0.1:8766/ui · Health: http://127.0.0.1:8766/health · OpenAPI: http://127.0.0.1:8766/docs

Полная инструкция: [`OPENWEBUI_ZI_RAG.md`](OPENWEBUI_ZI_RAG.md).
