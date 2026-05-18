# ZI_RAG / openwebui_zi_rag - план аудита и доработок

Документ-чеклист по результатам ревью модуля `openwebui_zi_rag` и
`openwebui_functions/zi_rag_filter.py`.

Статусы:
- [ ] not started
- [~] in progress
- [x] done / закрыто как not-a-bug
- [?] требует ручной проверки на стенде

Каждый пункт ссылается на файл/функцию, чтобы можно было быстро вернуться.

Важные уточнения повторной проверки:
- Каталог `/media/onel/videoanalytics/ZI_RAG` сейчас не является git-репозиторием.
  Коммиты возможны только если в рабочем окружении реализации появится `.git`.
- В этом окружении команда `python` отсутствует, использовать `python3`.
- `openwebui_functions/zi_rag_filter.openwebui.json` содержит копию фильтра.
  Любая правка `openwebui_functions/zi_rag_filter.py` должна синхронизировать
  JSON-экспорт вручную или через будущий `tools/build_filter.py`.

---

## 1. Реальные баги (high priority)

- [x] **1.1** `extract_msg.clean()` теряет строковые значения, потому что
      возвращает результат только в ветке `bytes`.
      Файл: `openwebui_zi_rag/indexing/extraction.py` (`extract_msg.clean`).
      Последствия: теряются `Subject/From/To/Cc/Bcc/Date`, `htmlBody` при
      пустом `body`, имена вложений.
      Fix: после ветки `bytes` всегда делать `return clean_text(str(value))`.
      Проверка: тест с fake `extract_msg.Message`, где заголовки и `htmlBody`
      заданы обычными строками.
      Статус проверки: закрыто после прогона
      `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`.

- [x] **1.2** `Registry._pack_embedding([])` - закрыто как not-a-bug.
      Текущая строка
      `return len(values), struct.pack(...) if values else b""` парсится как
      tuple `(len(values), (pack if values else b""))`, поэтому для пустого
      списка возвращает `(0, b"")`, а не `b""`.
      Файл: `openwebui_zi_rag/indexing/registry.py` (`_pack_embedding`).
      Действие: код менять не требуется. Если функция будет рефакториться,
      можно переписать на явный `if` только ради читаемости.

- [x] **1.3** В синхронных `/analyze` и `/compliance/analyze` не ловится
      `OllamaCancelled` -> 500 вместо контролируемого ответа.
      Файл: `openwebui_zi_rag/server.py` (`analyze`, `compliance_analyze`).
      Fix: добавить `except OllamaCancelled` до `except OllamaError`.
      Рекомендуемый статус: HTTP 499, detail вроде `"Analysis canceled"`.
      Проверка: тесты на оба sync endpoint; job-path `/analyze/jobs` уже
      обрабатывает отмену отдельно и не должен регрессировать.

- [x] **1.4** `_chat_streaming` падает на любой битой ndjson-строке
      (`json.loads(line)` без `try`).
      Файл: `openwebui_zi_rag/ollama_client.py` (`_chat_streaming`).
      Fix: оборачивать `json.loads` в `try/except json.JSONDecodeError`,
      логировать/пропускать строку, продолжать чтение стрима.
      Проверка: тест со стримом `bad-json`, валидной строкой и `done=true`.

- [x] **1.5** Дефолт и fallback generation-модели всё ещё завязаны на
      `"qwen3.6:latest"`.
      Файлы: `openwebui_zi_rag/config.py`, `openwebui_zi_rag/server.py`
      (`_resolve_generation_model`), `openwebui_zi_rag/web/app.js`,
      `OPENWEBUI_ZI_RAG.md` (документация).
      Текущее состояние: UI уже использует selector `/ollama/models`, но
      load/save и server fallback всё ещё подставляют `qwen3.6:latest`.
      Тесты `tests/test_openwebui_zi_rag.py` используют `qwen3.6:latest`
      как валидное тестовое имя - это не production fallback; менять их только
      если так проще обновить тест под новую логику.
      Fix:
      1. `SidecarConfig.deep_generation_model = ""`; `compliance_generation_model`
         оставить `""`.
      2. Удалить все fallback `"qwen3.6:latest"` из `_resolve_generation_model`
         и `web/app.js`.
      3. `_resolve_generation_model` лучше переписать на явный список кандидатов
         от caller: для Deep RAG `payload.generation_model`, затем
         `cfg.deep_generation_model`; для Compliance `payload.generation_model`,
         затем `cfg.compliance_generation_model`, затем `cfg.deep_generation_model`
         как осознанный shared fallback. Кандидат валиден только если он реально
         есть в `/api/tags` (с учётом aliases).
      4. Если валидной модели нет, поднимать `HTTPException 409` с сообщением
         `"Generation model is not configured. Select one of: ..."` и списком
         доступных моделей.
      5. Если `/api/tags` недоступен, возвращать понятную 502, а не скрытый
         fallback.
      6. В `OPENWEBUI_ZI_RAG.md` обновить раздел Deep Multi-Pass RAG: убрать
         фразу про default fallback и описать новое поведение «модель должна
         быть выбрана в settings или передана клиентом».
      Проверка: тесты на пустой конфиг, invalid requested + valid configured,
      отсутствие моделей, сохранение пустого значения из UI.

- [x] **1.6** Утечка и возможная коллизия per-request state в `Filter`.
      Файл: `openwebui_functions/zi_rag_filter.py`.
      Проблемы:
      - `_sources_by_key` / `_deep_answers_by_key` чистятся только в `outlet`;
        если `outlet` не вызвался, состояние висит бесконечно.
      - `_request_key` может стать просто `user.id`, если нет `chat_id` и
        `message_id`; параллельные запросы одного пользователя могут смешаться.
      Fix: хранить `created_at` вместе со значениями, TTL около 10 минут,
      чистить при каждом `inlet` и `outlet`; усилить ключ через доступные
      `chat_id`, `message_id`, `session_id`, `user.id`, hash последнего
      user-message и `id(body)` как fallback.
      Проверка: тест на TTL-очистку и тест на два параллельных body одного
      пользователя без `chat_id`.
      Важно: синхронизировать `openwebui_functions/zi_rag_filter.openwebui.json`.

- [x] **1.7** Гонка имён файлов в `save_upload`: между `exists()` и
      `write_bytes()` другой поток может записать тот же путь.
      Файл: `openwebui_zi_rag/indexing/service.py` (`save_upload`).
      Fix: не использовать `os.replace` поверх заранее проверенного имени,
      потому что он может перетереть чужой файл. Использовать финальное имя с
      непредсказуемым suffix (`secrets.token_hex`) и эксклюзивное создание:
      `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)` с retry.
      `filename` в registry можно оставить исходным sanitized именем, а
      `stored_path` - уникальным физическим путём. При ошибке записи или
      registry insert удалять частично созданный файл.
      Проверка: concurrency-тест из нескольких потоков с одинаковым filename.

- [x] **1.8** FAISS-файлы, cache и search не синхронизированы достаточно
      надёжно при параллельном rebuild/retrieve.
      Файл: `openwebui_zi_rag/indexing/vector_store.py`.
      Проблемы:
      - `build_index` пишет `vectors.faiss` и `vector_map.json` прямо в целевые
        файлы; `_cached_index` может прочитать частично записанную пару.
      - `_INDEX_CACHE_LOCK` защищает dict, но не полный цикл read/search/write.
      Fix: per-index `threading.RLock`; запись `vectors.faiss` и map во
      временные файлы в том же каталоге + `os.replace` после успешной записи;
      cache обновлять под тем же per-index lock; `index.search` оборачивать
      этим lock.
      Проверка: retrieve/rebuild в двух потоках, отсутствие ошибок чтения FAISS
      и несоответствия map/vector signature.

- [x] **1.9** Совокупный timeout compliance не ограничен wall-clock deadline.
      Файл: `openwebui_zi_rag/server.py` (`run_compliance_analysis`).
      Сейчас каждая секция и финальный проход могут ждать до
      `compliance_timeout_sec`; при 80 секциях server thread может жить намного
      дольше ожидания клиента.
      Fix: общий `deadline = time.monotonic() + compliance_timeout_sec`;
      перед каждым retrieve/chat проверять остаток; per-call timeout должен быть
      `min(short_timeout, remaining)`. При исчерпании deadline возвращать
      контролируемый 504/499 с понятным detail и не продолжать фоновой работы.
      Проверка: тест с fake client, который превышает deadline на второй секции.

- [x] **1.10** `OllamaClient.embed` глотает любую `OllamaError` и идёт в
      legacy `/api/embeddings`, маскируя OOM/auth/5xx.
      Файл: `openwebui_zi_rag/ollama_client.py` (`embed`, `_json_request`).
      Fix: ввести status-bearing исключение, например `OllamaHTTPError(code,
      detail)`. Fallback на `/api/embeddings` делать только для HTTP 404/405
      от `/api/embed`; 401/403/429/500/URLError прокидывать.
      Проверка: тесты на 404 fallback и на 500 без fallback.

- [x] **1.11** Полный перестрой `chunk_fts` при каждом `Registry.init()`.
      Файл: `openwebui_zi_rag/indexing/registry.py` (`init`).
      Сейчас на каждом старте sidecar и на каждом `_refresh_service`
      выполняется `DELETE FROM chunk_fts` + `INSERT ... SELECT FROM chunks`
      безусловно. На больших индексах (десятки/сотни тысяч чанков) это
      секунды-десятки секунд DELETE+INSERT и значительный рост WAL.
      `replace_document_chunks` уже синхронизирует FTS на upsert, поэтому
      повседневный rebuild не нужен.
      Fix: запускать перестрой только если миграция `chunk_fts` (версия 4)
      реально применяется впервые. Варианты:
      1. Проверять `cursor.rowcount` от
         `INSERT OR IGNORE INTO schema_version (4, ...)`: если 1 - значит
         запись только что вставлена, делать full rebuild. Если 0 - пропустить.
      2. Дополнительно или вместо: при `SELECT COUNT(*) FROM chunk_fts == 0`
         и `EXISTS chunks WHERE active=1 AND deleted_at IS NULL` тоже выполнять
         восстановление (на случай повреждённого FTS).
      Проверка: тест, который дважды зовёт `Registry(db_path)` и убеждается,
      что `chunk_fts` не пересобирается; тест миграции с пустого FTS на полный.

- [x] **1.12** Документ остаётся в статусе `vectorizing`, если
      `_rebuild_index_debounced` бросает исключение в `index_document_now`.
      Файл: `openwebui_zi_rag/indexing/service.py` (`index_document_now`).
      `index_documents_now` это уже учитывает (`_set_documents_status(...,
      DocumentStatus.FAILED, error=...)`), а в single-document пути нет.
      Job через `run_job` корректно становится `failed`, но статус документа
      висит как `vectorizing`, пока его не переоткроет следующий reindex.
      Fix: обернуть rebuild и финальный `set_document_status(INDEXED)` в
      try/except, при ошибке проставлять `DocumentStatus.FAILED` с текстом
      ошибки и пробрасывать исключение наверх.
      Проверка: тест с fake rebuild, который кидает RuntimeError; проверить,
      что документ оказывается в `failed`.

- [x] **1.13** `index_chat_attachments` не уважает wall-clock deadline.
      Файл: `openwebui_zi_rag/indexing/service.py` (`index_chat_attachments`,
      `index_documents_now`).
      Сейчас `chat_attachment_timeout_sec` живёт только на стороне OpenWebUI
      filter (HTTP read timeout). На стороне sidecar
      `routes/chat_attachments.py` запускает работу в `asyncio.to_thread`, и
      она не прерывается при превышении таймаута клиентом. При больших
      вложениях клиент отвалится по таймауту, а sidecar продолжит индексировать
      и держать поток.
      Fix: симметрично пункту 1.9. Завести общий `deadline = time.monotonic()
      + chat_attachment_timeout_sec`, прокидывать его в `index_documents_now`
      через новый параметр или контекст; перед каждым документом и rebuild
      проверять остаток. При исчерпании deadline возвращать понятный 504/499
      и не продолжать фоновой работы.
      Проверка: тест с длинным upsert, гарантировать прерывание.

- [x] **1.14** `rebuild_index_documents_now` падает на пустом индексе.
      Файл: `openwebui_zi_rag/indexing/service.py` (`rebuild_index_documents_now`).
      Сейчас `if not vectorizing_ids: raise ValueError("No chunked documents
      to rebuild")`. UI вызывает `POST /indexes/{id}/rebuild` без
      `document_ids` в том числе после массового удаления, и получает 409
      «No chunked documents to rebuild», хотя ожидание - пустой ребилд.
      `delete_documents` уже умеет fallback в `build_index([], [])`.
      Fix: при пустом выборе делать атомарный empty rebuild без ошибки,
      возвращать `{"index_id": ..., "chunks": 0, "embedding_dim": 0}`.
      Проверка: тест rebuild для индекса без документов, и тест rebuild
      после массового удаления документов через REST.

- [x] **1.15** Гонка `delete_index` с активным `retrieve()`.
      Файл: `openwebui_zi_rag/indexing/service.py` (`delete_index`),
      `openwebui_zi_rag/indexing/vector_store.py`.
      `delete_index` под капотом делает `shutil.rmtree(indexes_path /
      index_id)`. Между этим и идущим параллельно `retrieve()` нет общей
      синхронизации: vector_store держит per-index `RLock`, но он не виден
      сервису. Возможно, `retrieve()` пройдёт `vector_path.exists()` и упадёт
      на `faiss.read_index` уже после rmtree, или, наоборот, прочитает
      частично удалённую карту.
      Fix: завести общий per-index lock, разделяемый между `service` и
      `vector_store` (например через `vector_store.acquire_index_lock(
      indexes_path, index_id)`); `delete_index` берёт write-lock, retrieve и
      build - read-lock; rmtree выполнять под write-lock + invalidate cache
      под тем же lock.
      Проверка: concurrency-тест, в котором retrieve и delete_index
      запускаются одновременно, не должно быть исключений.

---

## 2. Архитектура и качество кода

- [x] **2.1** `server.py` - более 2000 строк. Разделить на пакет:
      Стадия 2 в работе: вынос FastAPI app, runtime state, schemas, health
      и routes по доменам.
      - `routes/indexes.py`
      - `routes/documents.py`
      - `routes/jobs.py`
      - `routes/analyze.py`
      - `routes/compliance.py`
      - `routes/chat_attachments.py`
      - `routes/admin.py`
      - `services/analysis.py` (`run_multi_pass_analysis`, `run_compliance_analysis`)
      - `services/prompting.py` (форматирование doc-блоков, sources)
      - `app.py` - фабрика приложения.
      Делать после закрытия 1.1-1.10 и покрытия тестами.
      Статус: стадия 1 выполнена - prompt/source formatting вынесены в
      `services/prompting.py`, orchestration Deep RAG/Compliance - в
      `services/analysis.py`; `server.py` уменьшен до ~1450 строк.
      Осталось разнести route registration по `routes/*` и добавить
      полноценную фабрику приложения.

- [x] **2.2** Дублирование между `Filter` и `server.py`:
      `_filter_docs_for_prompt` / `_filter_analysis_docs`, `_clean_quote_text`,
      `_query_terms`, `_query_term_hits`, `_score`, `_trim_text`,
      `_compact_dialog_context`.
      Подход: вынести общую логику в `openwebui_zi_rag/text_utils.py`, а в
      single-file фильтр копировать через новый `tools/build_filter.py`.
      Скрипт должен также обновлять `openwebui_functions/zi_rag_filter.openwebui.json`.
      До появления build-скрипта любые изменения фильтра синхронизировать с JSON
      вручную.

- [x] **2.3** `SidecarConfig` (dataclass + ручной merge env/json) ->
      Pydantic v2 `BaseSettings` / `pydantic-settings` с `Field(ge=, le=)`.
      Важно: `pydantic-settings` сейчас отсутствует в
      `openwebui_zi_rag_requirements.txt`; добавить зависимость отдельно и
      проверить совместимость FastAPI/Pydantic.
      Поскольку модуль ставится с нуля, миграция старого `config.json` не
      требуется, но имена полей сохранить.

- [x] **2.4** `Registry.connect()` на каждый вызов открывает новое sqlite-
      соединение. Перед заменой измерить эффект. Если менять, использовать
      небольшой пул или shared connection с `check_same_thread=False` и
      `threading.Lock`, сохранив WAL, commit/rollback семантику и тесты
      параллельных операций.

- [x] **2.5** Глобальные `_config`, `_service`, `_MODEL_CACHE`, `_ANALYSIS_JOBS`
      -> DI через `app.state` для тестируемости.

- [x] **2.6** `retrieval_query_variants` хардкодит правила про КСПД/ТСПД/Интернет.
      Вынести в конфиг `query_synonyms` (JSON-словарь), оставить текущие правила
      дефолтным preset.

- [x] **2.7** В `OllamaClient` один `timeout` отвечает и за HTTP-запрос, и за
      чтение всего стрима. Разделить:
      `connect_timeout`, `request_timeout`, `stream_idle_timeout`.

- [x] **2.8** Статусы документов и jobs задаются строками. Объявить Enum:
      `DocumentStatus(pending, extracting, vectorizing, indexed, failed,
      canceled, deleted)` и `JobStatus(queued, running, cancel_requested,
      completed, failed, canceled)`.

- [x] **2.9** Миграции схемы: есть детектор отдельных колонок, но нет таблицы
      `schema_version`. Завести версионирование и явные миграции.

- [x] **2.10** Упаковка артефактов: `openwebui_zi_rag_bundle.zip` лежит в корне.
      Не править zip вручную. Если нужна публикация сборки, завести отдельный
      build-скрипт и явно перечислить, какие файлы входят в bundle.

- [x] **2.11** Indirection через `sys.modules["openwebui_zi_rag.server"]`
      и `from .. import server as _server` в routes ради monkeypatch.
      Файлы: `openwebui_zi_rag/runtime.py` (`make_ollama_client`),
      `openwebui_zi_rag/services/analysis.py` (`_sync_analysis_service_dependencies`),
      `openwebui_zi_rag/routes/analyze.py`,
      `openwebui_zi_rag/routes/compliance.py`.
      Решение пользователя: вариант **B** - оставить как есть, поведение
      намеренное.
      Реализация (минимальная):
      1. В `runtime.make_ollama_client` добавить docstring/комментарий,
         объясняющий, зачем класс резолвится через `sys.modules`
         (совместимость с `monkeypatch.setattr(rag_server, "OllamaClient",
         ...)` в тестах).
      2. В `services/analysis._sync_analysis_service_dependencies` добавить
         docstring, что функция нужна для test-time подмены атрибутов
         `server.OllamaClient` / `server.extract_text` /
         `server.run_multi_pass_analysis`.
      3. В `routes/analyze.py` и `routes/compliance.py` добавить
         комментарий рядом с `from .. import server as _server` (один уже
         частично есть), указать, что поздний импорт требуется для совместимости
         с monkeypatch.
      Никакого рефакторинга DI/`app.state` не делаем.
      Проверка: см. 7.20 (тест не нужен, behavior не меняется).

- [x] **2.12** Race на `_metrics` в `RagService`.
      Файл: `openwebui_zi_rag/indexing/service.py` (`_record_metric`).
      Несколько потоков одновременно дёргают `_record_metric` (background
      indexing job + retrieve в FastAPI thread pool). dict-операции в CPython
      атомарны, но `count`/`avg_sec` читаются и пишутся не транзакционно -
      возможны искажённые aggregates.
      Fix: завести `threading.Lock`, оборачивать update/read; `metrics_snapshot`
      должен делать копию под локом.
      Проверка: stress-тест на 4 потока, каждый делает _record_metric, итоговые
      count/total_sec совпадают с ожидаемыми.

- [x] **2.13** Persistance очереди Deep-RAG jobs.
      Файлы: `openwebui_zi_rag/services/jobs.py`, `openwebui_zi_rag/runtime.py`
      (`_install_runtime_state`).
      `analysis_jobs` хранятся только в `app.state`. Любой рестарт sidecar
      теряет очередь, OpenWebUI получает 404 на следующем polling/SSE
      запросе.
      Решение пользователя: вариант **A** - минимальный, без таблицы в SQLite.
      Реализация:
      1. На старте sidecar (`_install_runtime_state` или
         `configure_runtime_state`) очищать `app.state.zi_rag_analysis_jobs`
         (в этом и состоит «потеря» очереди при рестарте, но явно).
      2. В `routes/analyze.py` и `services/jobs.py`:
         - `read_analyze_job`, `cancel_analyze_job`, `analysis_job_event_stream`
           при отсутствии job_id отдавать **HTTP 410 Gone** с понятным detail
           (`"Sidecar restarted, multi-pass job is gone. Retry from filter."`)
           вместо текущего 404.
         - В SSE-event `error` для пропавшего job_id класть тот же текст.
      3. Filter `_run_deep_analysis` уже умеет fallback на синхронный
         `/analyze` через `except Exception`. Убедиться, что он корректно
         реагирует на 410 (логически - то же поведение, что 404; ничего не
         менять, но добавить regression-тест в filter-тестах, если есть).
      Проверка: см. 7.26 - тест на 410 от read/cancel/SSE для несуществующего
      job_id.

- [x] **2.14** Возможные частичные коммиты при вложенных
      `Registry.connect()` под `RLock`.
      Файл: `openwebui_zi_rag/indexing/registry.py` (`_RegistryConnectionContext`).
      `RLock` реентерабелен, поэтому вложенный `connect()` берёт его повторно;
      на выходе из внутреннего контекста делается commit/rollback уже совершённой
      части работы. Сейчас вложенных вызовов нет, но любая будущая бизнес-логика
      может неявно их создать (`registry.foo()` внутри `registry.bar()` под одним
      контекстом) и получить частичный коммит.
      Fix: считать глубину контекста (`_depth` в `Registry`) и коммитить/откатывать
      только на самом внешнем уровне (depth == 0). На вложенных уровнях контекст
      должен только проверять `exc` и пробрасывать его.
      Проверка: тест с искусственно вложенным `connect()` и принудительным
      исключением во вложенном уровне; убедиться, что внешний rollback откатывает
      всю транзакцию.

---

## 3. Безопасность

- [x] **3.1** `require_api_key` пропускает `127.0.0.1` без ключа.
      Поведение по умолчанию оставить текущим, но добавить опциональный строгий
      режим.
      Файлы: `openwebui_zi_rag/config.py`, `openwebui_zi_rag/server.py`,
      `openwebui_zi_rag/web/index.html`, `openwebui_zi_rag/web/app.js`.
      Fix:
      1. Новое поле: `require_api_key_localhost: bool = False`.
      2. Env override: `ZI_RAG_REQUIRE_API_KEY_LOCALHOST=1`.
      3. Если strict включён и `api_key` пустой, `update_config` должен
         отклонять конфиг понятной 400/409 ошибкой; server startup тоже должен
         явно сообщать о некорректном конфиге.
      4. В `require_api_key`: если strict `True`, не пропускать localhost без
         валидного ключа.
      5. UI `Настройки`: чекбокс "Требовать API key для localhost
         (для multi-user машин)".
      Проверка: localhost без ключа в default mode проходит; strict+key требует
      ключ; strict без key отклоняется.

- [x] **3.2** Сообщения `PermissionError`/`FileNotFoundError` иногда отдают
      абсолютные пути.
      Файлы: `openwebui_zi_rag/server.py`, `openwebui_zi_rag/indexing/service.py`.
      Fix: добавить helper для публичных ошибок, который оставляет только
      basename или нейтральное сообщение (`"File not found"`, `"Path is outside
      allowed_source_roots"`). В логах можно оставить полный путь.
      Проверка: add-path outside allowlist и удалённый `stored_path` не раскрывают
      абсолютный путь в HTTP detail.

- [x] **3.3** `_post_multipart` / `_multipart_filename` уже фильтрует CR/LF/TAB,
      кавычки, backslash, NUL и управляющие символы `< 32`, но не фильтрует
      `DEL` (`0x7f`) и другие непечатаемые Unicode control chars.
      Файл: `openwebui_functions/zi_rag_filter.py`.
      Fix: в `_multipart_filename` заменить непечатаемые символы через
      `ch.isprintable()` или явную проверку control chars; сохранить текущую
      защиту от CR/LF/header injection.
      Проверка: расширить `test_upload_filename_sanitizers_block_header_and_path_tricks`
      кейсом `bad\x7fname.txt`; синхронизировать JSON-экспорт фильтра.

- [x] **3.4** `/health` отдаёт абсолютные пути и доступен без API key.
      Файл: `openwebui_zi_rag/services/health.py` (`build_health_payload`),
      `openwebui_zi_rag/routes/admin.py` (`health`, `metrics`).
      Сейчас payload содержит `storage_dir`, `registry` как абсолютные пути
      и не закрывается `require_api_key` (мониторинг).
      Это противоречит политике из п. 3.2 (мы убрали path-leak в ошибках,
      но в норме их отдаём).
      Решение пользователя: вариант **A** - убрать абсолютные пути из
      публичного `/health`, перенести полный payload в авторизованный
      `/health/full` (или новый `/health/full` рядом с уже существующим
      `/metrics`).
      Реализация:
      1. `build_health_payload` разделить на `build_public_health_payload`
         (без `storage_dir`, `registry`, `metrics`,
         `embedding_model_dimension`) и `build_full_health_payload`
         (текущий полный payload).
      2. `GET /health` возвращает публичный payload без `Depends(require_api_key)`.
         Поля: `status`, `version`, `checks` (с агрегированными статусами; без
         `index_id` в FAISS-проверке - заменить на `index_count`).
      3. Новый `GET /health/full` под `Depends(require_api_key)` отдаёт
         полный payload (бывший `/health`).
      4. `OPENAPI_ROUTE_METADATA` обновить - добавить `("GET", "/health/full")`.
      5. Если нужно - оставить `/metrics` как есть (он уже авторизован).
      Проверка: см. 7.22.

- [x] **3.5** Filter молча работает на дефолтах, если sidecar требует
      `require_api_key_localhost` и фильтр запущен без API key.
      Файл: `openwebui_functions/zi_rag_filter.py` (`_sidecar_admin_config`).
      `_get_json("/config")` обёрнут в `try/except: data = {}`, ошибка
      съедается, `_admin_config_cache` остаётся пустым, и
      `sync_sidecar_admin_config` молча перестаёт работать (cache TTL 5 секунд,
      потом снова пустой результат).
      Решение пользователя: реализовать **оба** варианта (логи + notification).
      Реализация:
      1. **Логи.** В `_sidecar_admin_config` логировать причину ошибки через
         `print(...)` (попадёт в OpenWebUI logs). Не спамить: писать только
         когда статус сменился (раньше работало, сейчас сломалось, или
         наоборот).
      2. **Notification.** Завести флаг `_admin_config_unavailable` и текстовое
         поле `_admin_config_error_message`. При вызове `inlet`, если флаг
         установлен и `event_emitter` есть, эмитить **один раз** на запрос
         `notification` с типом `warning` и текстом
         `"ZI_RAG sidecar /config недоступен (...). Фильтр работает на
         собственных valves."` Не повторять для последующих сообщений того же
         chat-а - сохранить ts последней нотификации в `_admin_config_warned_at`
         и эмитить не чаще раз в N секунд (например 60).
      3. Синхронизировать `openwebui_functions/zi_rag_filter.openwebui.json`
         через `python3 tools/build_filter.py`. Регрессионный тест 7.14
         должен проходить.
      Проверка: см. 7.23.

- [x] **3.6** `/embedding/models` для openai-провайдера блокирует event loop.
      Файл: `openwebui_zi_rag/routes/admin.py` (`embedding_models`).
      `urllib.request.urlopen(request, timeout=cfg.request_timeout_sec)`
      вызывается синхронно в FastAPI route, до 120 секунд (и больше при
      сетевых тормозах). UI зависает, и параллельные запросы тоже.
      Fix: обернуть HTTP-вызов в `await asyncio.to_thread(...)` (или вынести
      сетевую часть в helper и вызывать через `to_thread`), сохранить
      существующее кеширование через `_cached_payload`.
      Проверка: тест с фейковой сетью, замеряющей, что route не блокирует
      event loop (проверка по asyncio через `asyncio.wait_for`).

---

## 4. Производительность

- [x] **4.1** Только `IndexFlatIP`. Подключить автоматический выбор:
      - до 50k чанков - `IndexFlatIP`;
      - от 50k чанков - `IndexHNSWFlat` с inner product для нормализованных
        векторов (`M=32`, `efConstruction=200`, `efSearch=128`).
      Параметры в конфиг: `index_type` (`auto`/`flat`/`hnsw`, default `auto`),
      `hnsw_threshold_chunks`, `hnsw_m`, `hnsw_ef_construction`,
      `hnsw_ef_search`.
      Важно: FAISS search должен сохранить текущую нормализацию и преобразование
      score. HNSW не поддерживает удаление без rebuild, но текущий pipeline и так
      rebuild-ит индекс целиком.

- [x] **4.2** Дедуп чанков перед эмбеддингом.
      Не удалять duplicate chunks из выдачи без отдельного решения: одинаковые
      фрагменты могут быть в разных документах. Безопасный вариант - кешировать
      embedding по `text_hash + model + prefix`, а затем привязывать один vector
      к нескольким chunk_id.

- [x] **4.3** Параллельные батчи эмбеддингов для OpenAI-совместимого endpoint
      (`ThreadPoolExecutor(max_workers=2..4)`). Для Ollama оставить sequential.
      Проверить порядок результатов и backpressure при 429/5xx.

- [x] **4.4** `pdf_render_scale` в конфиг (текущий хардкод 3.0 -> дефолт 2.5).

- [x] **4.5** Хранить embedding cache в SQLite как `float16` опционально
      (`embedding_cache_dtype=fp32|fp16`, default `fp32`). FAISS всё равно
      строить из `float32`; добавить тест допуска на восстановление.

- [x] **4.6** `_INDEX_CACHE` без LRU, держит всё в памяти. Ограничить
      `OrderedDict` на 16-32 индекса. Согласовать с per-index lock из 1.8.

- [x] **4.7** `UPSERT` для chunks вместо полного `DELETE+INSERT` при reindex.
      Делать только после введения стабильного ключа чанка (`document_id`,
      `chunk_no`, `text_hash`) и очистки исчезнувших чанков, иначе легко сломать
      FK и embedding cache.

- [x] **4.8** `list_documents_page` использует `LIKE %term%` по 5 полям.
      Файл: `openwebui_zi_rag/indexing/registry.py` (`list_documents_page`).
      Без полнотекстового индекса это full scan по `documents`. На больших
      индексах поиск тормозит UI.
      Fix варианты:
      1. Завести FTS5-индекс `document_fts(filename, source_path, external_id)`
         и переключить поиск с `LIKE` на `MATCH`. Ranking уже не нужен (показ
         в UI), достаточно фильтра.
      2. Альтернатива: оставить `LIKE`, но добавить SQLite FTS только когда
         `total > 5000`. Принимать решение по `EXPLAIN QUERY PLAN`.
      Проверка: микробенчмарк `list_documents_page` для 50k документов до и
      после изменения; SQL должен использовать FTS index.

- [x] **4.9** `_INDEX_CACHE` не выгружает FAISS по неактивности.
      Файл: `openwebui_zi_rag/indexing/vector_store.py` (`_INDEX_CACHE`).
      `OrderedDict` ограничен 32 индексами, но без TTL: на больших HNSW каждый
      индекс может занимать сотни MB RAM, и они держатся, пока не вытеснятся
      по LRU.
      Решение пользователя: вариант **A** - документация без TTL, фоновую
      выгрузку не вводим.
      Реализация:
      1. В `OPENWEBUI_ZI_RAG.md` добавить раздел «FAISS in-memory cache».
         Описать:
         - кеш ограничен `_INDEX_CACHE_MAX_SIZE = 32` индексами (LRU);
         - примерный объём памяти на индекс: `4 * embedding_dim * chunk_count`
           байт для `IndexFlatIP`; для HNSW - тот же объём + графовая
           надстройка `M * 4 * chunk_count` байт;
         - кеш сбрасывается при `delete_index`, `rebuild_index_now`,
           `_refresh_service` (через `clear_index_cache` /
           `invalidate_index_cache`);
         - при необходимости очистить кеш руками - перезапустить sidecar.
      2. Никаких новых конфигов / фоновых тасков не вводить.
      Проверка: ручная (нет smoke-теста на текст документации).

---

## 5. Качество поиска

- [x] **5.1** Гибридный поиск BM25 (SQLite FTS5) + dense FAISS + RRF fusion.

- [x] **5.2** Опциональный кросс-энкодер реранк через тот же
      OpenAI-совместимый endpoint; включать при `len(results) > 10`.

- [x] **5.3** MMR при отборе top-K вместо простого дедупа по тексту.

- [x] **5.4** Опциональное расширение запроса через LLM (HyDE / multi-query).

- [x] **5.5** Гибридный score: BM25 hardcode `min(0.95, max(threshold, 0.72))`.
      Файл: `openwebui_zi_rag/indexing/service.py` (`retrieve`).
      Сейчас любой BM25-хит автоматически получает score >= 0.72, независимо
      от его реального BM25-rank. Это значит, что для индекса с
      `score_threshold` 0.6 пользователь увидит BM25-фрагменты со score 0.72
      и они пройдут все нижние пороги. `filter_docs_for_prompt` опирается на
      `score`, и это искажает ranking.
      Решение пользователя: вариант **A** - нормализовать BM25 как
      lexical_score, `score = max(dense, lexical_normalized)`. Имена полей
      внешнего API не меняем (`score`, `hybrid_score`, `retrieval_score`,
      `rerank_score` остаются как сейчас).
      **Условие мерджа:** синтетический regression-тест должен показать, что
      качество ranking «до/после» не падает на тестовом наборе. Если падает -
      откатить и обсудить.
      Реализация:
      1. В `retrieve()` заменить hardcoded `lexical_score = min(0.95,
         max(threshold, 0.72))` на нормализацию по rank.
         Формула: `lexical_score = 1.0 / (1.0 + max(0, fts_rank - 1))`
         (rank 1 → 1.0, rank 2 → 0.5, rank 10 → 0.1; rank приходит из
         `fts_rank`, заполняется при fusion).
         Дополнительно ограничить `lexical_score` сверху коэффициентом
         (например `min(0.95, ...)`), чтобы BM25-only хит не доминировал
         над высокоскоринговым dense.
      2. Финальный `score = max(dense_score, lexical_score)`.
         Если хит только BM25 (`dense_score == 0`), `score = lexical_score`;
         если есть dense - dense обычно выше, lexical работает как нижняя
         граница.
      3. Сохранить отдельные поля `dense_score`, `bm25_rank`, `hybrid_score`
         (RRF), `retrieval_sources` без изменений.
      4. Обновить или удалить тесты, которые ловят старое плато 0.72 для
         BM25-only хитов (если такие есть).
      5. Синтетический ranking-тест (под 7.25):
         - Подготовить набор: ~10 чанков, `query`, эталонный ranking
           (lexical-only matches, dense-only matches, гибрид).
         - Старый ranking зафиксировать как baseline (snapshot перед
           изменением, через `git diff` или текстовый файл).
         - Новый ranking сравнивать с baseline по NDCG@k или по точному
           совпадению top-K.
         - Если новый ranking хуже - откатить и обсудить.
      Проверка: см. 7.25.

---

## 6. API/UX

- [x] **6.1** SSE-стриминг для `/analyze` вместо polling по
      `/analyze/jobs/{id}`.

- [x] **6.2** Реальный `/health`: пинг Ollama `/api/tags`, чтение любого
      FAISS-индекса, статус SQLite.
      Сейчас `/health` всегда 200, если приложение поднялось.

- [x] **6.3** OpenAPI tags/summaries на эндпоинтах.

- [x] **6.4** i18n: вынести строки в `messages.ru.json`/`messages.en.json`.

- [x] **6.5** В `/health` отдавать `embedding_model_dimension` по индексам;
      предупреждать о расхождении с текущим конфигом.

- [x] **6.6** UI model selectors: показывать пустое состояние "модель не
      выбрана" и ошибку `/ollama/models` отдельно от пустого списка моделей.
      Связано с 1.5.

- [x] **6.7** README не описывает, что `openwebui_zi_rag_bundle.zip` -
      артефакт сборки.
      Файл: `README.md`.
      `tools/build_bundle.py` пересобирает zip; в репозитории файл уже лежит,
      но в README нет упоминания, что его нельзя править вручную и как
      пересобирать.
      Fix: добавить в README раздел `Build`/`Release` со ссылкой на
      `python3 tools/build_bundle.py` и пояснением, что zip - артефакт.
      Проверка: ручная (нет smoke-теста на текст README).

---

## 7. Тесты

- [x] **7.1** MSG с непустыми строковыми `Subject/From/To`, `htmlBody` и именем
      вложения (под 1.1).
- [x] **7.2** `_pack_embedding([])` - ручная AST/runtime-проверка показала,
      что баг из 1.2 отсутствует. Автотест добавлять только если функция будет
      переписываться.
- [x] **7.3** Отмена sync `/analyze` и `/compliance/analyze` через
      `OllamaCancelled` (под 1.3).
- [x] **7.4** `_chat_streaming` с битой ndjson-строкой (под 1.4).
- [x] **7.5** TTL и отсутствие коллизии `_sources_by_key` /
      `_deep_answers_by_key` (под 1.6).
- [x] **7.6** Конкурентный `save_upload` с одинаковым именем (под 1.7).
- [x] **7.7** Конкурентный retrieve/rebuild/search FAISS (под 1.8).
- [x] **7.8** Общий deadline compliance (под 1.9).
- [x] **7.9** `OllamaClient.embed`: fallback только на HTTP 404/405 (под 1.10).
- [x] **7.10** Resolution generation-модели без fallback `qwen3.6:latest`
      (под 1.5).
- [x] **7.11** Strict localhost API key и маскировка path-leak ошибок
      (под 3.1/3.2).
- [x] **7.12** Smoke-тесты `/health`, `/config`, `/retrieve` поверх временного
      `SidecarConfig`.
- [x] **7.13** Прогон `ruff` + `pyright`/`mypy --strict` на пакете
      `openwebui_zi_rag` после декомпозиции `server.py`.
      Статус: закрыто через `mypy --strict`; pyright не устанавливался,
      потому что plan acceptance допускает `pyright`/`mypy --strict` как
      альтернативы. Ruff чист по `openwebui_zi_rag/`,
      `openwebui_functions/`, `tools/`, `tests/`.
      - **Реальный баг найден и исправлен** ruff F821:
        `services/analysis.py` использовала `final_prompt` без импорта в
        ветке Deep RAG `mode="answer"`. Добавлен импорт; синтез финального
        ответа multi-pass теперь не падает.
      - Опциональные runtime-зависимости без stubs (`easyocr`, `extract_msg`,
        `faiss`, `pypdfium2`, `pytesseract`, `pandas`) покрыты точечными
        секциями `mypy.ini`; новые stubs/dependencies не устанавливались.
      - Ложные срабатывания на `try/except ImportError` для FastAPI/uvicorn
        закрыты локальными `# type: ignore[assignment, misc]`.
      - Реальные annotation gaps исправлены в `config.py`,
        `indexing/{registry,service,vector_store}.py`, `ollama_client.py`,
        `services/{health,jobs}.py`, `routes/{admin,analyze}.py`,
        `runtime.py`, `server.py` и `services/analysis.py`.
      Проверка:
      `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
      -> Success: no issues found in 29 source files.
      `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
      -> 102 passed.
      `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
      -> All checks passed!
- [x] **7.14** Регрессионный тест синхронизации
      `openwebui_functions/zi_rag_filter.py` и
      `openwebui_functions/zi_rag_filter.openwebui.json`: парсить JSON,
      сравнивать поле `content` с содержимым `.py` (допустимо сравнение через
      `rstrip("\n")`). Падение этого теста - сигнал, что JSON-экспорт устарел.
      Связано с 2.2.

- [x] **7.15** Идемпотентность `Registry.init()` для FTS (под 1.11).
      Тест: дважды инициализировать `Registry` поверх одной БД, до и после
      перестроить FTS из чанков; проверить, что `chunk_fts` не пересоздаётся
      и что после повреждения (ручного `DELETE FROM chunk_fts`) восстановление
      срабатывает только при следующем `init()` если включена fallback-ветка.

- [x] **7.16** Документ переходит в `failed`, если rebuild падает (под 1.12).
      Тест: подменить `_rebuild_index_debounced`, чтобы он бросал
      `RuntimeError`, дернуть `index_document_now`, проверить, что статус
      документа в БД `failed`, error непустой; параллельно job через `run_job`
      становится `failed`.

- [x] **7.17** Wall-clock deadline для chat attachments (под 1.13).
      Тест с fake `extract_text`/`embedding_client`, который дольше
      `chat_attachment_timeout_sec`; убедиться, что sidecar возвращает
      контролируемую ошибку (504/499 или RuntimeError) и не продолжает
      обработку оставшихся документов после превышения.

- [x] **7.18** Empty rebuild не падает (под 1.14).
      Тест: `rebuild_index` индекс без документов и индекс после массового
      удаления; убедиться, что ответ `chunks: 0`, `embedding_dim: 0` и
      FAISS-файлы корректно очищены.

- [x] **7.19** Конкурентный `delete_index` + `retrieve` (под 1.15).
      Тест: запустить retrieve и delete_index в потоках; ни один из них не
      должен падать; после delete_index retrieve возвращает пустой результат
      или 404 на индекс, без ValueError из FAISS.

- [x] **7.20** DI вместо monkeypatch (под 2.11).
      Решение пользователя: вариант **B** в 2.11 - оставить как есть,
      рефакторинг не делаем. Тесты остаются на текущем `monkeypatch.setattr(
      rag_server, ...)`. Дополнительный тест не нужен; пункт закрыт как
      «принято решение не реализовывать».

- [x] **7.21** Метрики thread-safe (под 2.12).
      Stress-тест: 4 потока, каждый зовёт `_record_metric("test", ...)` 1000
      раз; итоговый `count == 4000`, `total_sec` с допуском.

- [x] **7.22** `/health` без абсолютных путей и без API key (под 3.4).
      Тест unauthenticated GET `/health`: проверить, что в payload нет
      абсолютных путей (никакого `cfg.storage_path` / `cfg.registry_path`
      целиком), нет `metrics`, нет `embedding_model_dimension`. Только
      `status`, `version`, агрегированные `checks`. Статус-коды по веткам
      остаются прежними (200 / 503).
      Также тест authorized GET `/health/full`: содержит полный payload
      (storage_dir, registry, metrics, embedding_model_dimension);
      без API key возвращает 401 (или 401 и localhost-проход, если
      `require_api_key_localhost == False`).

- [x] **7.23** Filter поведение при 401 от sidecar /config (под 3.5).
      Тесты:
      1. Фейковый sidecar возвращает 401 на `GET /config`. Прогнать
         `inlet` через `Filter()` (как делают существующие тесты фильтра,
         см. `tests/test_openwebui_zi_rag.py`). Убедиться, что фильтр
         продолжает работать на собственных valves (например, `rag_enabled`
         читается из valves), не падает и не зацикливается.
      2. Зафиксировать через `event_emitter`-mock, что хотя бы один
         `notification` warning отправлен с текстом, упоминающим
         `/config`/`api key`/`unavailable`.
      3. После «починки» (200 от `/config` через 60+ секунд)
         `_admin_config_warned_at` сбрасывается, повторных warning нет.
      4. Регрессионный тест 7.14 продолжает проходить
         (`zi_rag_filter.openwebui.json` синхронизирован).

- [x] **7.24** `/embedding/models` не блокирует event loop (под 3.6).
      Тест: подменить openai-провайдер медленным `urlopen`-ответом; вызвать
      эндпоинт через TestClient внутри `asyncio.wait_for`; убедиться, что
      `to_thread` пропускает остальные корутины.

- [x] **7.25** BM25 score нормализация (под 5.5).
      Тесты:
      1. Юнит-тест нормализации: смоделировать retrieval с BM25-only хитом
         на rank 1, 2, 5, 10. Убедиться, что `lexical_score` пропорционален
         `1/(1+rank-1)` и не равен константе 0.72.
      2. Hybrid: dense_score=0.85 + BM25 rank 1. Финальный
         `score == max(0.85, 1.0) == 1.0`. Если решено ограничить сверху
         (`min(0.95, ...)`), проверить, что cap корректен.
      3. Synthetic ranking baseline:
         - Файл `tests/data/bm25_ranking_baseline.json` (создаётся в этом
           пункте) с фиксированной выдачей retrieval из тестового набора
           (10-20 чанков, 5 запросов).
         - Тест прогоняет retrieval, сравнивает текущий top-K с baseline.
         - Acceptance: top-K совпадает или NDCG@5 не хуже baseline (выбрать
           одну метрику).
         - Если хуже - тест падает, изменения откатываются.
      4. Тест на отсутствие плато 0.72: BM25 rank 5 даёт `score < 0.72`
         (после нормализации `lexical = 1/5 = 0.2`).

- [x] **7.26** Deep-RAG job 410 после restart sidecar (под 2.13).
      Тесты:
      1. Создать analysis job через `POST /analyze/jobs`, дождаться его в
         `app.state.zi_rag_analysis_jobs`.
      2. Симулировать restart: вызвать `configure_runtime_state(...)` или
         `_install_runtime_state(...)` (реализация под 2.13 должна чистить
         dict при настройке).
      3. `GET /analyze/jobs/{job_id}` → 410 с detail, упоминающим
         `restart`/`gone`.
      4. `POST /analyze/jobs/{job_id}/cancel` → 410.
      5. SSE `GET /analyze/jobs/{job_id}/events` → событие `error` с тем же
         текстом и закрытие стрима.

---

## 8. Решения, которые надо подтвердить с пользователем

- [x] **8.1** Имя дефолтной generation-модели - решено: убрать хардкод,
      пользователь выбирает модель в UI из списка `/ollama/models`.
      См. п. 1.5.
- [x] **8.2** Ужесточение `require_api_key` - решено: опционально через флаг
      конфига, по умолчанию текущее поведение. См. п. 3.1.
- [x] **8.3** FAISS-индекс - решено: автоматический выбор по размеру
      (`IndexFlatIP` до 50k чанков, `IndexHNSWFlat` выше). Установка с нуля,
      миграция существующих индексов не требуется. См. п. 4.1.
- [x] **8.4** Переход на Pydantic v2 `BaseSettings` допустим, но требует
      добавления `pydantic-settings` и отдельного этапа. Установка с нуля,
      миграция старого `config.json` не нужна. См. п. 2.3.

- [x] **8.5** Persistance Deep-RAG jobs (под 2.13). Решение принято:
      вариант **A** - на restart чистить очередь, отдавать 410 с явным
      сообщением. Таблица в SQLite не вводится. См. п. 2.13.

- [x] **8.6** `/health` без авторизации vs path-leak (под 3.4). Решение
      принято: вариант **A** - публичный `/health` без абсолютных путей и
      без metrics; новый авторизованный `/health/full` отдаёт полный payload.
      `/metrics` остаётся как сейчас. См. п. 3.4.

- [x] **8.7** TTL eviction для `_INDEX_CACHE` (под 4.9). Решение принято:
      вариант **A** - документация без TTL, фоновую выгрузку не вводим.
      См. п. 4.9.

- [x] **8.8** Изменение score-API в retrieval (под 5.5). Решение принято:
      вариант **A** - нормализованный BM25 как lexical_score, итоговый
      `score = max(dense, lexical_normalized)`. Имена полей API не меняются.
      Условие мерджа: synthetic ranking-тест (см. 7.25 пункт 3) должен
      показать, что качество ranking не падает на тестовом наборе. Если
      падает - откатываем и обсуждаем. См. п. 5.5.

---

## История изменений

- 2026-05-17 - закрыт 5.5/7.25: BM25-only score больше не получает
  hardcoded plateau `min(0.95, max(threshold, 0.72))`; добавлен
  `_lexical_score_from_fts_rank()` с формулой
  `min(0.95, 1 / (1 + max(0, fts_rank - 1)))`, итоговый retrieval
  `score = max(dense_score, lexical_score)`. Создан baseline
  `tests/data/bm25_ranking_baseline.json`; synthetic ranking gate показал
  точное совпадение top-K с baseline, деградации нет. Добавлены тесты на
  rank normalization, dense+BM25 cap и отсутствие плато 0.72 для rank 5.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 120 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 3.5/7.23: `Filter._sidecar_admin_config()` теперь
  фиксирует недоступность `/config`, пишет `print(...)` в OpenWebUI logs
  только при смене состояния, а `inlet()` отправляет throttled warning
  notification через `event_emitter` не чаще одного раза в 60 секунд.
  При восстановлении `/config` состояние и warning throttle сбрасываются.
  `openwebui_functions/zi_rag_filter.openwebui.json` синхронизирован через
  `python3 tools/build_filter.py`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 116 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 2.13/7.26: runtime install явно очищает
  in-memory очередь Deep-RAG jobs; потерянные job_id теперь возвращают
  HTTP 410 Gone с detail
  `"Sidecar restarted, multi-pass job is gone. Retry from filter."` для
  read/cancel, а SSE endpoint отвечает 410 text/event-stream с `error`
  event и тем же текстом. Добавлен regression test с созданием job,
  симуляцией restart через `configure_runtime_state()` и проверкой
  read/cancel/SSE. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 115 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 3.4/7.22: `/health` разделён на публичный payload
  (`status`, `version`, агрегированные `checks`, без абсолютных путей,
  metrics и embedding dimensions) и авторизованный `/health/full` со старым
  полным payload. Добавлен OpenAPI metadata для `/health/full`, обновлены
  health regression tests, включая 401 для полного endpoint без API key.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 114 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 4.9: в `OPENWEBUI_ZI_RAG.md` добавлен раздел
  `FAISS in-memory cache` с LRU-лимитом 32 индекса, формулами памяти для
  Flat/HNSW, правилами инвалидации и ручной очисткой через рестарт sidecar.
  `runtime.refresh_service()` теперь дополнительно вызывает
  `clear_index_cache()`, чтобы `_refresh_service` соответствовал описанному
  сбросу cache. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 114 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 2.11: добавлены docstrings/комментарии к
  `runtime.make_ollama_client`, server-facade sync shim и поздним импортам в
  Deep/Compliance routes. Поведение намеренно оставлено через
  `sys.modules["openwebui_zi_rag.server"]` и `from .. import server as _server`
  ради совместимости с `monkeypatch.setattr(rag_server, ...)`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 114 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - зафиксированы решения раунда 2 группы C: 2.11 → B (оставить
  monkeypatch indirection, добавить docstrings); 2.13/8.5 → A (на restart
  чистить очередь Deep-RAG jobs и отдавать 410 вместо 404); 3.4/8.6 → A
  (публичный `/health` без абсолютных путей и metrics, новый
  авторизованный `/health/full`); 3.5 → оба варианта (логи + notification
  через event_emitter, throttle 60 сек); 4.9/8.7 → A (документация без
  фоновой TTL eviction); 5.5/8.8 → A с условием (нормализованный BM25
  lexical_score, мерджим только если synthetic ranking-тест не показывает
  деградацию). Закрыт 7.20 как «принято решение не реализовывать»
  (под 2.11 → B). Добавлен новый тест 7.26 под 2.13. Открытыми остались:
  2.11 (docstrings), 2.13, 3.4, 3.5, 4.9, 5.5, 7.22, 7.23, 7.25, 7.26.

- 2026-05-17 - закрыт 4.8: добавлена миграция `schema_version=5`
  (`document_fts`) для полнотекстового поиска документов по filename,
  source/stored path, external_id и error; FTS синхронизируется при
  create/status update/delete и восстанавливается на init при пустом FTS с
  активными документами. `list_documents_page(query=...)` теперь использует
  `document_fts MATCH` вместо `LIKE %term%` для нормальных токенов, с
  fallback на LIKE для слишком коротких запросов. В открытых решениях
  2.13/8.5 будущая migration Deep-RAG jobs сдвинута на `schema_version=6`.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 114 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 2.14: `Registry.connect()` получил depth-counter и
  флаг rollback-required; commit/rollback выполняется только на внешнем
  уровне, а вложенный exception помечает всю транзакцию на rollback.
  Добавлен regression test, где inner `connect()` успешно выходит, затем
  outer context падает, и обе записи откатываются. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 113 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 1.15/7.19: в `vector_store` добавлен публичный
  `acquire_index_lock()`, использующий тот же per-index `RLock`, что
  FAISS read/search/write; `RagService.delete_index()` теперь под этим lock
  отменяет jobs, удаляет registry-запись, директории индекса/uploads и
  инвалидирует cache. Добавлен concurrency test `delete_index` + `retrieve`.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 112 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 3.6/7.24: `/embedding/models` переведён в async route
  с async cache wrapper; загрузка Ollama models и OpenAI-compatible
  `/models` выполняется через `asyncio.to_thread`, а OpenAI HTTP-логика
  вынесена в helper. Добавлен regression test, проверяющий, что `urlopen`
  выполняется не в thread event loop. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 111 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 1.13/7.17: для chat attachments добавлен общий
  wall-clock deadline от `chat_attachment_timeout_sec`; route прокидывает
  deadline в `RagService`, сервис проверяет его между upsert документов,
  chunking, embedding batch и rebuild, а `/chat-attachments/index` возвращает
  HTTP 504 с detail `"Chat attachment indexing timed out"`. Добавлены тесты
  на остановку после первого долгого upsert и HTTP 504 route mapping.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 110 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 6.7: в `README.md` добавлен раздел
  `Build / Release`, где `openwebui_zi_rag_bundle.zip` описан как
  генерируемый артефакт, указан способ пересборки через
  `python3 tools/build_bundle.py`, `--output` и исключение storage/cache/
  SQLite/FAISS/runtime файлов. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 108 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 2.12/7.21: `_record_metric()` и
  `metrics_snapshot()` в `RagService` защищены отдельным lock; snapshot
  делает копию метрик и `last_extra` под lock перед округлением. Добавлен
  stress-test на 4 потока и 4000 обновлений одной метрики. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 108 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 1.14/7.18: `rebuild_index_documents_now()` теперь
  разрешает пустой rebuild, атомарно очищает FAISS/map через
  `build_index(..., [], [])`, сбрасывает `embedding_dim` индекса в `0` и
  возвращает `chunks: 0`, `embedding_dim: 0` без ошибки. Добавлены тесты для
  пустого индекса и REST-сценария после массового удаления документов.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 107 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 1.12/7.16: `index_document_now()` теперь оборачивает
  rebuild и финальный перевод документа в `indexed`; при ошибке rebuild
  документ переводится в `failed` с текстом ошибки, при отмене - в
  `canceled`. Добавлен regression test для прямого вызова и пути через
  `run_job`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 105 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - закрыт 1.11/7.15: `Registry.init()` больше не делает
  безусловный `DELETE+INSERT` для `chunk_fts`; полный rebuild запускается
  только при первом применении schema migration `chunk_fts` или при пустом
  FTS с активными чанками. Добавлены regression tests на пропуск повторного
  rebuild и восстановление пустого FTS. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 104 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.

- 2026-05-17 - раунд 2 ревью: добавлены новые открытые пункты по результатам
  повторного аудита `openwebui_zi_rag` после закрытия 1.1-7.14. Затронутые
  разделы: 1.11-1.15 (FTS rebuild на старте, документ застрявший в
  `vectorizing`, deadline для chat attachments, empty rebuild,
  delete_index race), 2.11-2.14 (DI вместо monkeypatch, threadsafe
  metrics, persistence Deep-RAG jobs, nested registry connect),
  3.4-3.6 (path-leak в /health, filter без admin config, async
  embedding/models endpoint), 4.8-4.9 (LIKE-поиск по documents,
  TTL для FAISS cache), 5.5 (BM25 score нормализация), 6.7 (README про
  bundle), 7.15-7.25 (соответствующие тесты), 8.5-8.8 (открытые решения
  для пользователя). Зафиксированы baseline-проверки:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 102 passed.

- 2026-05-17 - закрыт 7.13: `mypy --strict` очищен по
  `openwebui_zi_rag/` без установки новых stubs/dependencies. Добавлен
  `mypy.ini` с точечным игнорированием missing/untyped imports только для
  опциональных runtime-зависимостей (`easyocr`, `extract_msg`, `faiss`,
  `pypdfium2`, `pytesseract`, `pandas`). Исправлены strict annotation gaps
  и `Returning Any` в runtime/routes/services/indexing, типизированы FAISS
  wrapper/cache, job stream/cancel paths, config settings sources, OpenAI
  embeddings parsing и server analysis shim. Pyright не устанавливался:
  acceptance закрыт через `mypy --strict`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
  -> Success: no issues found in 29 source files.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 102 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
- 2026-05-17 - частично закрыт 7.13: запущен ruff (chist по
  `openwebui_zi_rag/`, `openwebui_functions/`, `tools/`, `tests/`).
  Mypy `--strict` нашёл 62 ошибки в 14 файлах. **Баг исправлен**:
  ruff F821 показал, что `services/analysis.py` использовал
  `final_prompt` без импорта в финальном синтезе Deep RAG `mode="answer"` -
  добавлен в импорт-блок prompting. Полный mypy-pass перенесён в
  отдельный шаг; pyright не установлен. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 102 passed.
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
  -> All checks passed!
- 2026-05-17 - закрыт 2.1: декомпозиция `server.py` завершена. Извлечены
  модули: `schemas.py`, `runtime.py`, `app.py` (фабрика приложения),
  `openapi_meta.py`, `services/health.py`, `services/jobs.py`, и пакет
  `routes/{admin,indexes,documents,chat_attachments,analyze,compliance,jobs}.py`.
  `server.py` уменьшен с 1491 до 231 строк и теперь работает как тонкий
  фасад с публичной поверхностью для тестов и `python -m openwebui_zi_rag`.
  `runtime.set_get_config_provider` пробрасывает `monkeypatch.setattr(
  rag_server, "get_config", ...)` в зависимости. `make_ollama_client`
  через ленивый lookup `sys.modules["openwebui_zi_rag.server"]` уважает
  monkeypatched `OllamaClient`. Идентичность `rag_server.get_config` и
  `runtime.get_config` сохранена для `app.dependency_overrides`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 102 passed.

- 2026-05-17 - закрыт 6.4: добавлены
  `openwebui_zi_rag/web/messages.ru.json` и `messages.en.json`, loader
  i18n в админке, переключатель RU/EN, `data-i18n` для основных статических
  элементов и `t()` для ключевых динамических сообщений/toast/confirm.
  Добавлен regression test на полноту RU/EN ключей. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 102 passed.
- 2026-05-17 - закрыт 5.4: добавлено опциональное LLM query expansion
  для retrieval: `query_expansion_enabled`, `query_expansion_model`,
  лимиты variants/tokens, парсинг JSON/строчного ответа LLM, расширенные
  query variants для dense/BM25 поиска и stats без падения retrieval при
  ошибке LLM. Настройки добавлены в админку. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 101 passed.
- 2026-05-17 - закрыт 5.2: добавлен опциональный cross-encoder rerank
  через OpenAI-compatible `/rerank` endpoint с настройками
  `rerank_enabled`, `rerank_model`, `rerank_min_results`, `rerank_top_n`;
  retrieval реранжирует candidate pool перед MMR и отдаёт
  `rerank_score`/`retrieval_score`/rerank stats. Настройки добавлены в
  админку. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 99 passed.
- 2026-05-17 - закрыт 5.1: добавлен SQLite FTS5 `chunk_fts`
  (schema version 4) с синхронизацией при replace/delete chunks; retrieval
  объединяет dense FAISS ranks и BM25 ranks через RRF, отдаёт
  `hybrid_score`/`retrieval_sources` и сохраняет MMR финальный top-K.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 96 passed.
- 2026-05-17 - закрыт 2.3: `SidecarConfig` переведён на
  `pydantic-settings.BaseSettings` с `Field(ge/le/gt)` и validators;
  `load_config()` использует `config.json` как низкоприоритетный settings
  source, env перекрывает JSON, direct kwargs сохраняют наивысший приоритет.
  В `openwebui_zi_rag_requirements.txt` добавлен `pydantic-settings`.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 94 passed.
- 2026-05-16 - закрыт 6.1: добавлен SSE endpoint
  `/analyze/jobs/{job_id}/events` для async Deep RAG jobs, OpenAPI metadata и
  regression test; OpenWebUI filter читает SSE stream для прогресса multi-pass
  анализа и сохраняет fallback на polling для старых sidecar. JSON-экспорт
  фильтра обновлён через `tools/build_filter.py`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 93 passed.
- 2026-05-16 - закрыт 5.3: retrieval после score sort и text dedupe
  выбирает финальный `top_k` через MMR по сохранённым chunk embeddings;
  при отсутствии embeddings сохраняется прежний порядок, приватные MMR-ключи
  не попадают в API response. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 92 passed.
- 2026-05-16 - закрыт 2.2: общие text helpers вынесены в
  `openwebui_zi_rag/text_utils.py`, `server.py` использует их напрямую,
  фильтр остаётся single-file с regression test на совпадение copied prompt
  filtering; добавлен `tools/build_filter.py` для атомарной синхронизации
  `.openwebui.json` из `.py`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 91 passed.
- 2026-05-16 - закрыт 2.5: runtime-состояние FastAPI перенесено в
  `app.state` через helpers (`config`, `service`, model cache,
  analysis jobs/lock); `_refresh_service` обновляет state и чистит cache,
  добавлен regression test восстановления/isolation state. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 89 passed.
- 2026-05-16 - закрыт 2.4: `Registry` переведён на shared SQLite connection
  с `check_same_thread=False`, `RLock`, явными commit/rollback в
  `connect()`-контексте и `close()`; добавлен тест параллельных операций.
  Локальная метрика `list_indexes` 1000 вызовов: было ~0.236 ms/call,
  стало ~0.084 ms/call. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 88 passed.
- 2026-05-16 - закрыт 2.8: добавлены `DocumentStatus` и `JobStatus`;
  `Registry` принимает enum-статусы и хранит прежние строковые значения,
  сервис использует enum в основных путях индексации/jobs. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 87 passed.
- 2026-05-16 - закрыт 4.2: добавлен persistent
  `embedding_text_cache` по `cache_key + text_hash`; duplicate chunk texts
  эмбеддятся один раз, а vector привязывается ко всем соответствующим
  `chunk_id` через существующий per-chunk cache. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 80 passed.
- 2026-05-16 - закрыт 6.3: для FastAPI routes добавлены OpenAPI tags и
  summaries через централизованный metadata mapping; regression test проверяет
  все описанные path operations в `/openapi.json`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 81 passed.
- 2026-05-16 - закрыт 2.7: `OllamaClient` получил отдельные
  `connect_timeout`, `request_timeout` и `stream_idle_timeout`; config/env
  добавлены как `connect_timeout_sec`, `request_timeout_sec`,
  `stream_idle_timeout_sec`, а server/factory прокидывают их в Ollama-вызовы.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 83 passed.
- 2026-05-16 - закрыт 2.6: правила расширения запросов для КСПД/ТСПД/Интернет
  вынесены в `query_synonyms` JSON-словарь конфига с дефолтным preset;
  `retrieve()` использует `SidecarConfig.query_synonyms`, пустой словарь
  отключает расширения. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 84 passed.
- 2026-05-16 - закрыт 2.9: Registry создаёт `schema_version` и фиксирует
  явные версии `initial_schema`, `documents_external_metadata`,
  `embedding_text_cache`; добавлен метод `schema_versions()` и regression
  test. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 85 passed.
- 2026-05-16 - закрыт 2.10: добавлен `tools/build_bundle.py`, который
  атомарно собирает `openwebui_zi_rag_bundle.zip` из явно перечисленных
  deploy-файлов и исключает storage/cache/SQLite/FAISS/runtime artefacts;
  zip добавлен в `.gitignore`, существующий root bundle не пересобирался.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 86 passed.
- 2026-05-16 - закрыт 4.4: добавлен `pdf_render_scale` в конфиг,
  env `ZI_RAG_PDF_RENDER_SCALE`, API `/config` и UI админки; PDF OCR
  EasyOCR/Tesseract используют настраиваемый scale с дефолтом 2.5.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 65 passed.
- 2026-05-16 - закрыт 4.6: `_INDEX_CACHE` переведён на bounded LRU
  `OrderedDict` с лимитом 32, cache hits обновляют recency, запись новых
  индексов вытесняет старые без изменения per-index `RLock`.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 66 passed.
- 2026-05-16 - закрыт 6.2: `/health` теперь проверяет SQLite,
  пингует Ollama `/api/tags`, пробует открыть существующий FAISS-индекс и
  отдаёт `503` с structured payload при ошибках; UI читает degraded health
  без остановки загрузки остальных данных. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 68 passed.
- 2026-05-16 - закрыт 6.6: generation model selectors различают
  "Модель не выбрана", пустой список моделей и ошибку `/ollama/models`;
  добавлен regression test UI-строк. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 69 passed.
- 2026-05-16 - закрыт 7.12: добавлен smoke test FastAPI для `/health`,
  `/config` и `/retrieve` поверх временного `RagService` с fake Ollama и
  локальным tiny FAISS-индексом. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 70 passed.
- 2026-05-16 - закрыт 4.1: дефолт `index_type` сменён на `auto`,
  добавлены HNSW-настройки конфига/env/UI, rebuild выбирает Flat при
  `chunk_count <= hnsw_threshold_chunks` и HNSW выше порога с inner product
  для нормализованных векторов. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 73 passed.
- 2026-05-16 - закрыт 4.3: для OpenAI-compatible embedding providers
  batch-вызовы выполняются через bounded `ThreadPoolExecutor` с максимум 4
  workers и сохранением порядка результатов; Ollama-путь остаётся sequential.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 75 passed.
- 2026-05-16 - закрыт 4.5: добавлен `embedding_cache_dtype=fp32|fp16`
  с env `ZI_RAG_EMBEDDING_CACHE_DTYPE`, API `/config` и UI; SQLite cache
  умеет писать fp16 half-float blob и читать старые fp32 blob по длине,
  FAISS rebuild по-прежнему получает float32 через vector store. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 77 passed.
- 2026-05-16 - закрыт 6.5: `/health` отдаёт
  `embedding_model_dimension` с текущей embedding-моделью, размерностями
  индексов и предупреждениями, если индекс построен на модели, отличной от
  текущего `config.embedding_model`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 78 passed.
- 2026-05-16 - закрыт 4.7: `replace_document_chunks` больше не делает полный
  `DELETE+INSERT`; неизменившиеся `chunk_no + text` сохраняют `chunk_id` и
  embedding cache, а изменённые/удалённые чанки чистят только свои
  `chunk_embeddings`. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 79 passed.
- 2026-05-16 - закрыт 3.3: `_multipart_filename` в фильтре заменяет
  непечатаемые символы через `str.isprintable()`, включая `DEL` и Unicode
  control chars; JSON-экспорт фильтра синхронизирован, sanitizer test расширен.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 64 passed.
- 2026-05-16 - закрыт 3.1/3.2/7.11: добавлен
  `require_api_key_localhost` с env `ZI_RAG_REQUIRE_API_KEY_LOCALHOST`,
  strict-конфиг без `api_key` отклоняется, localhost bypass отключается в
  strict mode; публичные path errors маскируют абсолютные пути. UI получил
  чекбокс настройки. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 64 passed.
- 2026-05-16 - закрыт 1.9/7.8: `run_compliance_analysis` получил общий
  wall-clock deadline по `compliance_timeout_sec`, проверки до/после
  extract/retrieve/chat/final pass и HTTP 504 при исчерпании времени;
  Ollama client timeout ограничивается остатком deadline. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 61 passed.
- 2026-05-16 - закрыт 1.8/7.7: FAISS vector/map операции синхронизированы
  per-index `RLock`, запись `vectors.faiss` и `vector_map.json` идёт через
  temp-файлы и `os.replace`, cache/read/search выполняются под тем же lock;
  добавлен concurrency test retrieve/rebuild/search. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 60 passed.
- 2026-05-16 - закрыт 1.10/7.9: добавлен `OllamaHTTPError` со статусом,
  `OllamaClient.embed` fallback-ится на legacy `/api/embeddings` только для
  HTTP 404/405 от `/api/embed`; 500 и другие ошибки больше не маскируются.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 59 passed.
- 2026-05-16 - закрыт 1.7/7.6: `save_upload` пишет загруженные файлы в
  уникальный physical `stored_path` через `os.open(... O_EXCL ...)` с random
  suffix, сохраняет исходный sanitized `filename` в registry и удаляет файл
  при ошибке записи/insert; добавлен concurrency test с одинаковым filename.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 57 passed.
- 2026-05-16 - закрыт 1.6/7.5 и 7.14: `Filter` хранит per-request state с
  `created_at`, чистит orphaned entries по TTL на каждом `inlet/outlet`,
  усиливает request key для параллельных запросов одного пользователя;
  JSON-экспорт фильтра синхронизирован и покрыт regression test. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 56 passed.
- 2026-05-16 - закрыт 1.5/7.10: убран production fallback
  `qwen3.6:latest`, default `deep_generation_model` теперь пустой,
  generation model валидируется по `/api/tags` с HTTP 409/502; UI сохраняет
  пустое значение без подстановки, документация обновлена. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 53 passed.
- 2026-05-16 - закрыт 1.4/7.4: `_chat_streaming` пропускает malformed
  ndjson-строки в stream-е Ollama и продолжает читать валидные chunks;
  добавлен regression test. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 50 passed.
- 2026-05-16 - закрыт 1.3/7.3: sync `/analyze` и `/compliance/analyze`
  возвращают HTTP 499 при `OllamaCancelled`; добавлены regression tests.
  Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 49 passed.
- 2026-05-16 - закрыт 1.1/7.1: `extract_msg.clean()` сохраняет строковые
  заголовки MSG, fallback `htmlBody` и имена вложений; добавлен regression
  test. Проверка:
  `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
  -> 47 passed.
- 2026-05-16 - план зафиксирован после первичного ревью.
- 2026-05-16 - закрыты решения по 8.1 и 8.2: дефолт generation-модели
  убирается (выбор пользователя из UI), strict-режим api_key - опционально
  через конфиг с дефолтом "выключен".
- 2026-05-16 - закрыты решения по 8.3 и 8.4: установка с нуля,
  FAISS - автовыбор Flat/HNSW по 50k порогу, Pydantic v2 BaseSettings
  допустим без миграции старого config.json.
- 2026-05-16 - повторная проверка плана: 1.2 закрыт как not-a-bug,
  3.3 отмечен уже реализованным, исправлены рекомендации по atomic upload,
  FAISS locking, generation model fallback, `pydantic-settings`, JSON-экспорту
  фильтра и запуску тестов через `python3`.
- 2026-05-16 - дополнено: в 1.5 добавлена правка `OPENWEBUI_ZI_RAG.md` и
  явное замечание, что `qwen3.6:latest` в тестах остаётся как тестовое имя.
  Добавлен п. 7.14 - регрессионный тест синхронизации `.py` и
  `.openwebui.json` фильтра. Перепроверка кода подтвердила точность
  правок пользователя по 1.2 (Python operator precedence: запятая в
  `return` создаёт tuple раньше `if/else`).
- 2026-05-16 - повторно открыт п. 3.3: sanitizer фильтра покрывает C0 control
  chars `< 32`, но пропускает `DEL` (`0x7f`), поэтому нужен маленький fix и
  расширение regression test.
- 2026-05-16 - финальная сверка плана:
  - 1.2 not-a-bug подтверждён runtime: `_pack_embedding([])` -> `(0, b'')`.
  - 3.3 повторно открыт: `'bad\x7fname.txt'` действительно проходит через
    текущий `_multipart_filename` без изменений; `\u0085` и `\u200b` тоже.
    Рекомендуемая замена `ord(ch) >= 32` -> `ch.isprintable()`: сохраняет
    кириллицу, эмодзи и пробел, отбрасывает DEL и Unicode controls.
  - 7.14: на момент сверки `.py` и `.openwebui.json` совпадают точно, но
    допущение `rstrip("\n")` оставлено как защита от хвостовых переводов
    строк.
  - 1.5: фиксирован каскад моделей. Deep: `payload.generation_model` ->
    `cfg.deep_generation_model`. Compliance: `payload.generation_model` ->
    `cfg.compliance_generation_model` -> `cfg.deep_generation_model` как
    shared fallback. `qwen3.6:latest` из `_resolve_generation_model` убрать.
  Содержательных правок плана не потребовалось, план готов к реализации.
