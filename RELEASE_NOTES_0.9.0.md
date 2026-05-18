# ZI_RAG 0.9.0

External RAG sidecar for OpenWebUI. Runs as a separate FastAPI service and integrates through one OpenWebUI Function filter — no patches to OpenWebUI sources.

This is the first public release. The codebase has been hardened through three rounds of code review (see `REVIEW_PLAN.md`); about 50 issues across correctness, security, performance, retrieval quality, and API have been closed.

## Highlights

- **Hybrid retrieval out of the box.** Dense FAISS (Flat → HNSW auto-switch above 50k chunks) fused with SQLite FTS5 BM25 via Reciprocal Rank Fusion, MMR top-K selection, optional cross-encoder rerank, optional LLM query expansion (HyDE / multi-query).
- **Deep multi-pass RAG.** `/analyze` retrieves a wide candidate set, splits into batches, asks Ollama to extract facts per batch, then synthesizes a final answer. Streamed via SSE; OpenWebUI Filter shows live progress.
- **Compliance Check.** `/check` runs document-against-NMD verification with section-level retrieval, JSON matrix, citations and per-file report. Wall-clock deadline-bounded.
- **Chat attachment indexing.** Files attached to an OpenWebUI chat are auto-indexed into a per-chat sidecar index (`owui_chat_<chat_id>`). Native OpenWebUI Knowledge can be disabled to avoid double-RAG.
- **Single-file Function filter.** `openwebui_functions/zi_rag_filter.py` is a single-file OpenWebUI Function. JSON export `zi_rag_filter.openwebui.json` kept in sync for direct import.
- **Strict static checks.** Codebase passes `ruff`, `mypy --strict` and 120 pytest cases. CI workflow included.
- **Russian/English admin UI** with persisted locale and OpenAPI tags/summaries on every endpoint.

## What's in the bundle

`openwebui_zi_rag_bundle.zip` (built via `python3 tools/build_bundle.py`) contains exactly what is needed for deployment:

- `openwebui_zi_rag/` — FastAPI sidecar package (routes, services, indexing, ollama client, web admin UI assets).
- `openwebui_functions/zi_rag_filter.py` and `zi_rag_filter.openwebui.json` — OpenWebUI Function (single-file filter).
- `openwebui_zi_rag_requirements.txt` — pinned-floor dependencies.
- `README.md`, `OPENWEBUI_ZI_RAG.md` — install + operations guide.

Storage (`openwebui_zi_rag_storage/`), caches and bundle itself are excluded.

## Install

```bash
unzip openwebui_zi_rag_bundle.zip -d /opt/zi_rag
cd /opt/zi_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r openwebui_zi_rag_requirements.txt
python3 -m openwebui_zi_rag
```

Sidecar URL: `http://127.0.0.1:8766/ui` (admin) and `http://127.0.0.1:8766/health` (public liveness).

Then in OpenWebUI Admin Panel → Functions → Import → upload `openwebui_functions/zi_rag_filter.py` (or the JSON export). Set `sidecar_url` and `api_key` in valves. Disable native Knowledge for the same models to avoid double RAG.

See `OPENWEBUI_ZI_RAG.md` for full setup, Giga Embeddings, Deep RAG, Compliance Check, OCR, FAISS cache and PDF/OCR notes.

## Compatibility

- Python 3.10+ (CI runs 3.12).
- OpenWebUI 0.9.0+ (Function API).
- Ollama for embeddings/generation, or any OpenAI-compatible embeddings endpoint (Giga, llama.cpp, OpenAI proper).
- FAISS-CPU. GPU OCR via EasyOCR + CUDA optional; Tesseract supported on CPU.

## Configuration highlights

- `embedding_provider` = `ollama` | `openai` (with `embedding_base_url`, `embedding_api_key`).
- `index_type` = `auto` | `flat` | `hnsw`. HNSW kicks in above `hnsw_threshold_chunks` (default 50000) with `M`/`efConstruction`/`efSearch` knobs.
- `embedding_cache_dtype` = `fp32` | `fp16` (cache only; FAISS still float32).
- `query_synonyms` JSON for retrieval-time expansion presets (КСПД/ТСПД bundled by default).
- `query_expansion_enabled` + `query_expansion_model` for LLM-based HyDE/multi-query expansion.
- `rerank_enabled` + `rerank_model` for cross-encoder rerank via OpenAI-compatible `/rerank`.
- `connect_timeout_sec` / `request_timeout_sec` / `stream_idle_timeout_sec` separate Ollama HTTP timeouts.
- `require_api_key_localhost` for strict mode on multi-user hosts.
- Wall-clock deadlines on Compliance and Chat Attachment indexing.

## Notes

- No production fallback for generation model. `Generation model` must be selected in admin UI or passed in the request; otherwise `/analyze` and `/compliance/analyze` return HTTP 409 with the available model list.
- `/health` is split into a public probe (no abs paths, no metrics) and `/health/full` (auth-required). Container probes that scrape the full payload should target `/health/full`.
- Deep-RAG analysis jobs are in-memory only. After sidecar restart, calls to a stale `job_id` return HTTP 410; the OpenWebUI Filter falls back to synchronous `/analyze` automatically.
- Embedding-model dimension mismatch is reported in `/health/full` `embedding_model_dimension.warnings`. Force-reindex affected indexes after switching embedding models.
- BM25-only hits no longer get a hardcoded score plateau of 0.72. `score = max(dense_score, 1.0 / (1.0 + max(0, fts_rank - 1)))`. Fixed snapshot ranking test guards against regression.

## Verified

```text
ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/   → All checks passed
mypy openwebui_zi_rag/ --strict                                   → no issues found in 29 source files
pytest tests/test_openwebui_zi_rag.py --no-header -x              → 120 passed
```

CI mirrors all three checks; release workflow rebuilds and attaches `openwebui_zi_rag_bundle.zip` automatically when a release tag (`0.x`, `1.x`, `vX.Y`) is pushed.

## Acknowledgements

OpenWebUI for the Function plugin model, FAISS for the vector index, pdfplumber/pypdfium2/EasyOCR/Tesseract for extraction, Ollama for the local LLM stack.
