# OpenWebUI Enhanced RAG Sidecar

This is an isolated RAG sidecar for OpenWebUI. It does not modify OpenWebUI
sources and does not import or reuse `app_final` storage.

## Run

```bash
pip install -r openwebui_zi_rag_requirements.txt
python -m openwebui_zi_rag
```

Default URL:

```text
http://127.0.0.1:8766/ui
```

## OpenWebUI Setup

1. Open OpenWebUI Admin Panel.
2. Import `openwebui_functions/zi_rag_filter.py` as a Function.
3. Enable it as a Filter for the target models.
4. Set Valves:
   - `sidecar_url`: `http://host.docker.internal:8766` for Docker OpenWebUI,
     or `http://127.0.0.1:8766` for non-Docker OpenWebUI.
   - `api_key`: required when the sidecar is reachable from Docker/LAN.
     `start-ai-stack.sh` auto-generates it in
     `<ZI_RAG_STORAGE_DIR>/config.json` when `ZI_RAG_HOST` is non-local and
     syncs it into the OpenWebUI Function valves.
   - `sync_sidecar_admin_config`: keep enabled if admins should control RAG
     behavior from the sidecar UI.
5. Configure embedding model and the default index in the sidecar UI.
6. Disable native OpenWebUI Knowledge for the same target models to avoid double RAG.

The module is designed to stay external to OpenWebUI. Do not patch OpenWebUI
source files: run the sidecar separately and import/update only the Function
file through OpenWebUI's normal Function mechanism.

## Admin UI

Open `http://127.0.0.1:8766/ui` and click `Настройки`.

If an API key is configured, the browser UI will ask for it before reading or
saving settings. This is expected: all non-health sidecar endpoints are protected
when an API key exists.

The sidecar settings dialog now contains the admin controls normally needed for
OpenWebUI chat behavior:

- `RAG включён`: global on/off switch for the filter.
- `Auto deep для сложных вопросов`: enables automatic multi-pass for trigger
  phrases.
- `Финальный ответ делает sidecar`: when enabled, `/analyze` produces the final
  answer and the filter replaces the last assistant message with it.
- `Deep для каждого вопроса`: forces multi-pass without requiring `/deep`.
- `Chat retrieval top K`, relevance margin, prompt chunk limits and context
  budgets: fast RAG retrieval and context packing controls.
- `Auto deep trigger phrases`: phrase list used by the automatic deep detector.
- `Context template`: prompt wrapper injected into the last user message for
  fast RAG.

`/deep <question>` always forces deep mode even when automatic deep analysis is
disabled.

## Recommended Defaults

The repository defaults are tuned for broad document search while keeping local
setup generic:

- `score_threshold=0.50`
- `adaptive_score_margin=0.20`
- `retrieval_top_k=70`
- `deep_analysis_enabled=true`
- `deep_final_answer=true`
- `deep_force_all=false`
- `deep_top_k=70`

Environment-specific values intentionally stay empty in the repository:

- `api_key`
- `default_index_ids`
- `allowed_source_roots`
- `embedding_model`, `embedding_base_url`, `embedding_api_key`
- `deep_generation_model`, `deep_generation_base_url`,
  `deep_generation_api_key`

Configure those in `http://127.0.0.1:8766/ui` or via environment variables for
each deployment. For Docker OpenWebUI, set the Function `sidecar_url` to
`http://host.docker.internal:8766`; for a non-Docker OpenWebUI, use
`http://127.0.0.1:8766`.

## Chat Attachment Indexing

When a user attaches files to an OpenWebUI chat message, the Function can read
those files from OpenWebUI runtime storage and send them to the sidecar. The
sidecar stores them in a separate per-chat index:

```text
owui_chat_<chat_id>
```

The first response waits for indexing within `chat_attachment_timeout_sec`, so
the attached files can be used immediately. Later messages in the same chat keep
using the same chat index even when no files are attached. Regular ZI_RAG
indexes still participate in retrieval; the chat index is added on top instead
of replacing the default/user-selected indexes.

Duplicates are skipped by OpenWebUI file id plus file hash. If a file with the
same OpenWebUI id changes, the previous active document is soft-deleted and the
new version is indexed. Successfully handed-off file/doc attachments are removed
from the OpenWebUI request payload so native OpenWebUI file RAG does not inject
the same context again. Images and collection/knowledge items are left for
OpenWebUI.

Admin controls are in `http://127.0.0.1:8766/ui` -> `Настройки` ->
`Chat Attachments`:

- `Индексировать вложения чата`: global on/off.
- `Index prefix`: index id prefix, default `owui_chat_`.
- `Max files`, `Max file MB`, `Timeout sec`: safety and wait limits.

If OpenWebUI does not provide `chat_id`, the Function falls back to
`session_id`, `message_id`, or user id and emits a status note. That fallback is
best-effort and may not be stable across chats.

## Giga Embeddings

The launcher can run the local 4-bit NF4 Giga embedding model as an
OpenAI-compatible embeddings endpoint:

```text
http://127.0.0.1:5010/v1
```

Example model path:

```text
~/models/Giga/Giga-Embeddings-instruct-4bit-nf4
```

If you use your own launcher or process manager, restart the embedding endpoint,
the sidecar, and OpenWebUI after changing embedding settings. For example:

```bash
systemctl --user restart embeddings
systemctl --user restart zi-rag
systemctl --user restart openwebui
```

ZI_RAG settings for this backend:

```text
embedding_provider=openai
embedding_base_url=http://127.0.0.1:5010/v1
embedding_model=giga-embeddings-instruct-4bit-nf4
embedding_batch_size=4
embedding_query_prefix=Instruct: Given a search query, retrieve relevant passages that answer the query. Query:
embedding_document_prefix=
```

Native OpenWebUI RAG should use:

```text
RAG_EMBEDDING_ENGINE=openai
RAG_OPENAI_API_BASE_URL=http://host.docker.internal:5010/v1
RAG_EMBEDDING_MODEL=giga-embeddings-instruct-4bit-nf4
RAG_EMBEDDING_BATCH_SIZE=4
```

The Giga model returns 2048-dimensional embeddings. Existing indexes built with
another embedding model must be force-reindexed after switching.

## Deep Multi-Pass RAG

The OpenWebUI Function supports a second mode for broad analytical questions.
Use `/deep <question>` to force it, or enable `Auto deep для сложных вопросов`
in the sidecar UI to trigger it automatically for questions such as "сравни",
"полный перечень", "все требования" or "найди противоречия".

Deep mode calls the sidecar `/analyze` endpoint. The sidecar retrieves a larger
candidate set, filters and deduplicates chunks, splits them into batches, asks
the selected Ollama chat model to extract facts from each batch, then either:

- returns a ready final answer when `deep_final_answer` is enabled;
- returns a structured deep context for the OpenWebUI model when it is disabled.

There is no built-in generation model fallback. Select `Generation model` in
sidecar settings, pass `generation_model` in the request, or set
`ZI_RAG_DEEP_GENERATION_MODEL` in the launcher environment. The selected model
must be present in Ollama `/api/tags`; otherwise `/analyze` returns HTTP 409
with the available model list. If `/api/tags` is unavailable, the sidecar
returns HTTP 502 instead of silently choosing a fallback.

## Compliance Check

Compliance Check checks one or more files attached to an OpenWebUI chat message
against NMD requirements stored in a ZI_RAG index. Use it when the user asks the
model to verify a document package, find violations, or prepare a compliance
matrix.

Run modes:

- `/check <question>` forces compliance analysis for the current attachments.
- `/check index:IB <question>` forces compliance analysis and uses `IB` as the
  NMD index.
- Automatic mode runs only when `Compliance Check` and `Авто-проверка вложений`
  are enabled and the message contains both attachments and trigger phrases such
  as "проверь на соответствие", "соответствует ли", "найди нарушения" or
  "сделай акт".

The sidecar extracts attached `doc`, `docx`, `xls`, `xlsx`, `pdf`, `msg` and
text-like files into a temporary directory, splits them into sections, retrieves
relevant NMD requirements for each section, and asks Ollama to produce findings.
Temporary checked files are deleted after analysis and are not added to
permanent ZI_RAG indexes.

When Chat Attachment Indexing is enabled, `/check` attachments are also saved to
the per-chat index so users can ask follow-up questions about the same files
after the compliance report.

The report is generated by the sidecar and replaces the OpenWebUI assistant
message, so citations and matrix statuses are not rewritten by the chat model.
The expected output contains a common act for the whole package, a compliance
matrix, per-file details, and an HTML `<details><summary>Источники</summary>`
source block.

Admin controls are in `http://127.0.0.1:8766/ui` -> `Настройки` ->
`Compliance Check`:

- `Проверка вложений включена`: global sidecar on/off.
- `Авто-проверка вложений`: enables trigger-based automatic checks.
- `Индекс НМД по умолчанию`: NMD index for checks; empty means the normal
  default index.
- `Разрешить user override index`: allows user valves and `/check index:...`.
- `Max files`, `Max file MB`, `Section chars`, `Max sections`,
  `Requirement top K`, `Timeout sec`: safety and retrieval budgets.
- `Generation model`: Ollama model for compliance analysis; empty means the
  deep-generation fallback.
- `Trigger phrases`: phrase list for automatic mode.

## Storage

The default storage directory is:

```text
./openwebui_zi_rag_storage
```

Set `ZI_RAG_STORAGE_DIR` or edit settings in the sidecar UI to move it.

For GitHub/public packaging, keep runtime data out of version control:
`openwebui_zi_rag_storage/`, uploads, FAISS indexes, SQLite databases,
`.pytest_cache/`, `__pycache__/`, and installed OpenWebUI copies are artifacts,
not source.

## FAISS in-memory cache

The sidecar keeps recently opened FAISS indexes in process memory so repeated
retrieval does not reread `vectors.faiss` and `vector_map.json` on every
request. The cache is an LRU with `_INDEX_CACHE_MAX_SIZE = 32`, so at most 32
indexes are retained; there is no idle-time TTL eviction task.

Approximate memory per cached index:

- `IndexFlatIP`: `4 * embedding_dim * chunk_count` bytes for float32 vectors,
  plus small FAISS/Python metadata overhead.
- `IndexHNSWFlat`: the same vector storage plus roughly
  `M * 4 * chunk_count` bytes for graph links, where `M` is `hnsw_m`
  (default 32), again plus FAISS overhead.

Cache entries are invalidated when an index is deleted and replaced when
`rebuild_index_now` writes a new FAISS index. A sidecar config refresh through
`_refresh_service` also clears the in-memory FAISS cache, because storage paths
or index settings may have changed. To clear the cache manually, restart the
sidecar process.

## Local Folder Indexing

Uploading files from the UI works immediately. Adding local filesystem paths
requires `allowed_source_roots` in sidecar settings. Paths outside that allowlist
are rejected.

The sidecar UI supports multi-select in the document list:

- check individual documents or use `Выбрать видимые`;
- use `Неиндексированные` to select only documents whose status is not
  `indexed`;
- `В очередь` queues reindex jobs and skips documents that already have active
  queued/running jobs;
- `Принудительно` cancels active jobs for the selected documents and queues new
  reindex jobs.

When several documents are queued from the UI or from `Добавить путь`, the
sidecar extracts and updates all selected document chunks first, then rebuilds
the FAISS vector index once at the end. This avoids the expensive
document-by-document full rebuild loop.

Document status becomes `indexed` only after the final vector index rebuild has
finished. During extraction and embedding/index rebuild the UI shows
`extracting` or `vectorizing`, so unfinished documents are still treated as
unindexed by the `Неиндексированные` selector.

Admins can also queue a vector-only rebuild with `POST /indexes/{index_id}/rebuild`.
It reuses existing chunks, marks affected documents as `vectorizing`, rebuilds
FAISS once, then marks them `indexed`.

## PDF OCR

PDF pages with a text layer are read directly. Image-only PDF pages use OCR when
`enable_ocr` is enabled.

Recommended GPU OCR settings:

```text
ocr_engine=easyocr
ocr_gpu=true
ocr_languages=rus+eng
ocr_model_storage_dir=<ZI_RAG_STORAGE_DIR>/easyocr_models
```

The launcher passes `CUDA_VISIBLE_DEVICES` to the sidecar through `ZI_RAG_GPU`
(default: the same GPU selected for Ollama). Use `ZI_RAG_OCR_ENGINE=tesseract`
only when you explicitly want the old CPU OCR path.

## Citations

Newly indexed documents include source locators in retrieved chunks:

- DOC/DOCX/TXT/PDF: paragraph labels such as `абз. 12`, detected numbered
  clauses such as `пункт 3.1`, and PDF pages such as `стр. 4`.
- XLS/XLSX: sheet and row labels.
- MSG: message body paragraphs and supported attachment locators.

After changing citation behavior, reindex existing documents so old chunks get
the new locators.
