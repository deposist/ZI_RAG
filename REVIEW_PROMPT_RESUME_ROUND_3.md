# Промт для запуска раунда 3 ревью в новом диалоге

Решения по группе C приняты. Скопируй текст между маркерами ниже в новый
диалог. Промт рассчитан на Claude Opus / Codex / любой автономный агент с
доступом к файловой системе.

---

## START PROMPT

Ты работаешь над репозиторием ZI_RAG - RAG-сайдкар для OpenWebUI.

**Корень проекта:** `/media/onel/videoanalytics/ZI_RAG`

**Главный план:** `REVIEW_PLAN.md`. Читай его первым. Каждый пункт имеет id
(1.1, 2.3, 4.6 и т.д.) и статус (`[ ]`, `[~]`, `[x]`, `[?]`). Закрытые `[x]`
не реализуй заново.

### Текущее состояние

Все пункты разделов 1-7 первого и большая часть второго раунда закрыты.
В предыдущем раунде закрыты группы A и B второго раунда: 1.11, 1.12, 1.13,
1.14, 1.15, 2.12, 2.14, 3.6, 4.8, 6.7 и тесты 7.15-7.19, 7.21, 7.24.

Решения по группе C приняты пользователем и зафиксированы в плане:

- **2.11 → B (минимум):** оставить текущую indirection через
  `sys.modules["openwebui_zi_rag.server"]` и `from .. import server as _server`
  в routes. Без рефакторинга DI. Добавить docstrings/комментарии,
  объясняющие почему так сделано (совместимость с `monkeypatch.setattr(
  rag_server, ...)` в тестах). Тест 7.20 закрыт как «принято решение не
  реализовывать».

- **2.13 / 8.5 → A (минимум):** на старте sidecar чистить очередь
  `app.state.zi_rag_analysis_jobs` (явно, не неявно); `read_analyze_job`,
  `cancel_analyze_job` и SSE `analysis_job_event_stream` для пропавшего
  job_id отдают **HTTP 410 Gone** с detail
  `"Sidecar restarted, multi-pass job is gone. Retry from filter."` (вместо
  текущего 404). Filter уже падает в fallback на синхронный `/analyze`
  через `except Exception` - менять его не нужно, только проверить.
  Таблица `analysis_jobs` в SQLite не вводится.

- **3.4 / 8.6 → A:** разделить `/health` на публичный (без авторизации, без
  абсолютных путей и без metrics: только `status`, `version`, агрегированные
  `checks`) и авторизованный `/health/full` с полным payload. `/metrics`
  оставить как есть. `OPENAPI_ROUTE_METADATA` обновить.

- **3.5 → оба варианта (логи + notification):**
  1. Логи: в `_sidecar_admin_config` через `print(...)` в OpenWebUI logs;
     писать только при смене статуса (раньше работало → сломалось, или
     наоборот), не спамить.
  2. Notification: флаг `_admin_config_unavailable` + `_admin_config_warned_at`;
     один раз на запрос эмитить через `event_emitter` `notification` с
     типом `warning` и текстом
     `"ZI_RAG sidecar /config недоступен (...). Фильтр работает на
     собственных valves."`. Throttle - не чаще раз в 60 секунд.
  Синхронизировать `openwebui_functions/zi_rag_filter.openwebui.json`
  через `python3 tools/build_filter.py`.

- **4.9 / 8.7 → A (документация):** в `OPENWEBUI_ZI_RAG.md` добавить раздел
  «FAISS in-memory cache». Описать LRU размер 32 индексов, формулы памяти
  для Flat и HNSW, инвалидацию при `delete_index`/`rebuild_index_now`/
  `_refresh_service`, ручную очистку через рестарт sidecar. Никаких новых
  конфигов и фоновых тасков.

- **5.5 / 8.8 → A с условием:** заменить hardcoded
  `lexical_score = min(0.95, max(threshold, 0.72))` на нормализацию по rank.
  Формула: `lexical = 1.0 / (1.0 + max(0, fts_rank - 1))`, опционально
  ограничить сверху `min(0.95, ...)`. Финальный
  `score = max(dense_score, lexical_score)`. Имена полей API не меняем.
  **Условие мерджа:** synthetic ranking-тест (см. 7.25 пункт 3) должен
  показать, что top-K совпадает с baseline или NDCG@5 не хуже. Если хуже -
  **откатить** и сообщить мне.

### Что уже подтверждено как baseline

```
/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/
→ All checks passed!

/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict
→ Success: no issues found in 29 source files

/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x
→ 114 passed
```

Любая правка должна сохранять эти три проверки зелёными.

### Открытые пункты этого раунда

Реализуй все по очереди:

1. **2.11** - docstrings/комментарии без рефакторинга.
2. **2.13** - 410 для пропавших Deep-RAG jobs + явная очистка очереди на старте.
   Тест: **7.26**.
3. **3.4** - публичный `/health` (без paths) + авторизованный `/health/full`.
   Тест: **7.22**.
4. **3.5** - filter: логи + notification при 401 от sidecar /config; throttle 60s.
   Не забыть `tools/build_filter.py` для синхронизации JSON-экспорта.
   Тест: **7.23**.
5. **4.9** - раздел документации в `OPENWEBUI_ZI_RAG.md`. Без теста.
6. **5.5** - нормализация BM25 score; перед мерджем прогнать synthetic
   ranking-тест.
   Тест: **7.25** (4 подтеста, включая baseline).

### Рекомендуемый порядок реализации

A. Простые (без новых файлов и решений):
   - 2.11 (комментарии)
   - 4.9 (документация)
   - 3.4 + 7.22 (split /health)

B. Тесты + точечные правки:
   - 2.13 + 7.26 (410 на пропавший job)
   - 3.5 + 7.23 (filter logs + notification)

C. Под условием (требует baseline-теста):
   - 5.5 + 7.25 (BM25 нормализация). Делать **последним**, чтобы остальные
     тесты не зависели от изменения ranking.

### Что делать в каждой итерации

1. Перед правкой пункта поставь в `REVIEW_PLAN.md` статус `[~]`.
2. Реализуй scoped fix согласно зафиксированному решению (см. тело пункта
   в плане - там подробности).
3. Напиши/обнови соответствующий тест из раздела 7.
4. Прогон baseline:
   ```
   cd /media/onel/videoanalytics/ZI_RAG
   /media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x
   /media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/
   /media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict
   ```
5. После успешной проверки поставь `[x]` в плане и добавь строку в
   `## История изменений` с датой и кратким описанием.
6. Если пункт требует ручной проверки на стенде, поставь `[?]` и допиши
   конкретную инструкцию.
7. **Особо для 5.5:** если synthetic ranking-тест показывает деградацию
   качества - не закрывай пункт, откати изменения и сообщи мне результат.

### Жёсткие ограничения

- `openwebui_functions/zi_rag_filter.py` остаётся single-file. Любая правка
  обязана быть синхронизирована с
  `openwebui_functions/zi_rag_filter.openwebui.json` через
  `python3 tools/build_filter.py`. Регрессионный тест 7.14 проверяет это.
- Не вводить новые зависимости без обоснования.
- Не ломать имена полей `config.json`, схему SQLite, внешний API эндпоинтов.
  Для 3.4: `/health` остаётся, добавляется `/health/full`. Старые поля в
  публичном `/health` (`status`, `version`, `checks`) сохраняются.
- Не править `openwebui_zi_rag_bundle.zip` вручную; для пересборки -
  `python3 tools/build_bundle.py`.
- Не запускать долгоживущие процессы (uvicorn, watchers) в shell.

### Зафиксированные решения (всё из плана)

- **Generation model (8.1):** дефолт пустой, пользователь выбирает в UI.
- **Strict api_key для localhost (8.2):** опционально, по умолчанию выкл.
- **FAISS-индекс (8.3):** `index_type=auto`, Flat до 50k, HNSW выше.
- **Pydantic v2 BaseSettings (8.4):** реализован.
- **Persistance Deep-RAG jobs (8.5):** A - чистить на restart, 410 detail.
- **/health (8.6):** A - split на публичный и `/health/full`.
- **TTL для FAISS cache (8.7):** A - только документация.
- **BM25 score (8.8):** A - нормализация, мерджим под условием
  ranking-теста.

### Git и окружение

- В `/media/onel/videoanalytics/ZI_RAG` нет `.git`. Не делать коммиты.
- Только `python3` (не `python`).
- venv: `/media/onel/videoanalytics/app_final/.venv/bin/python`.

### Стиль кода

- Python 3.10+, `from __future__ import annotations`.
- mypy --strict проходит, не ломай. `# type: ignore[code]` всегда с
  конкретным error-code.
- Сообщения логов и UI - i18n через `messages.ru.json`/`messages.en.json`.
  Backend HTTP detail - на русском, кроме технических английских строк
  (`"Sidecar restarted, multi-pass job is gone. Retry from filter."` -
  английский, потому что фильтр на той же длине волны и парсит текст
  только для логов).
- Атомарная запись файлов - tmp + `os.replace`.
- Все SQL-параметры - placeholder'ы, без f-string интерполяции значений.

### Тесты

- Основной запуск:
  `cd /media/onel/videoanalytics/ZI_RAG && /media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
- Покрывать каждый закрытый пункт минимум одним тестом из раздела 7.
- Не использовать сетевые/GPU/Ollama-вызовы. Подменять
  `OllamaClient`/`OpenAIEmbeddingClient` через monkeypatch как раньше.
- Для concurrency-тестов держать таймауты короткими и детерминированными.
- Для 7.25 baseline-файл `tests/data/bm25_ranking_baseline.json` создаётся в
  этом раунде (до изменения формулы), потом используется как ground truth.

### Первый шаг

1. Прочитай `REVIEW_PLAN.md` целиком, особенно открытые пункты и блок
   «История изменений» за 2026-05-17.
2. Подтверди мне план реализации по порядку A → B → C.
3. Начинай с 2.11 (docstrings) - это разогрев на 5-10 минут.

## END PROMPT

---

## Как пользоваться

1. Открой новый диалог.
2. Скопируй всё между `## START PROMPT` и `## END PROMPT`.
3. Дай агенту доступ к файлу `REVIEW_PLAN.md` (через #File или просьбу
   прочитать).
4. Проверь, что агент действительно прочитал план и зафиксированные
   решения, прежде чем правит код.
