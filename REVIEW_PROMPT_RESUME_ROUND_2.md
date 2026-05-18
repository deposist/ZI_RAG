# Промт для запуска раунда 2 ревью в новом диалоге

Скопируй текст между маркерами ниже в новый диалог. Промт рассчитан на
Claude Opus / Codex / любой автономный агент с доступом к файловой системе.

---

## START PROMPT

Ты работаешь над репозиторием ZI_RAG - RAG-сайдкар для OpenWebUI.

**Корень проекта:** `/media/onel/videoanalytics/ZI_RAG`

**Главный план:** `REVIEW_PLAN.md`. Читай его первым. Каждый пункт имеет id
(1.1, 2.3, 4.6 и т.д.) и статус (`[ ]`, `[~]`, `[x]`, `[?]`). Закрытые `[x]`
не реализуй заново.

### Текущее состояние

Все пункты разделов 1-7 первого раунда (1.1-1.10, 2.1-2.10, 3.1-3.3,
4.1-4.7, 5.1-5.4, 6.1-6.6, 7.1-7.14, 8.1-8.4) закрыты.

После повторного аудита открыты новые пункты раунда 2:
- `1.11`-`1.15` - реальные баги (FTS rebuild, vectorizing-висюля, deadline
  для chat attachments, empty rebuild, delete_index race);
- `2.11`-`2.14` - архитектура (DI вместо monkeypatch, threadsafe metrics,
  persistence Deep-RAG jobs, nested registry connect);
- `3.4`-`3.6` - безопасность и эксплуатация (path-leak в /health, filter
  без admin config, async embedding/models endpoint);
- `4.8`-`4.9` - производительность (LIKE по documents, TTL для FAISS cache);
- `5.5` - качество поиска (нормализация BM25 score);
- `6.7` - README про bundle;
- `7.15`-`7.25` - тесты под новые пункты;
- `8.5`-`8.8` - открытые решения для пользователя.

### Что уже подтверждено как baseline

```
/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/
→ All checks passed!

/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict
→ Success: no issues found in 29 source files

/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x
→ 102 passed
```

Любая правка должна сохранять эти три проверки зелёными.

### Первый шаг

1. Прочитай `REVIEW_PLAN.md` целиком. Особо: разделы 1-8 и `## История
   изменений` за 2026-05-17.
2. Выведи список открытых пунктов раунда 2 с короткими названиями.
3. Предложи порядок работы (см. «Приоритеты» ниже) и дождись моего
   подтверждения перед правками кода.

### Что делать в каждой итерации

1. Перед правкой пункта поставь в `REVIEW_PLAN.md` статус `[~]`.
2. Реализуй минимальный scoped fix по выбранному пункту.
3. Напиши или обнови соответствующий тест из раздела 7 (7.15-7.25 для
   нового раунда).
4. Прогон baseline:
   `cd /media/onel/videoanalytics/ZI_RAG && /media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
5. После успешной проверки поставь `[x]` в плане и добавь строку в
   `## История изменений` с датой и кратким описанием.
6. Если пункт требует ручной проверки на стенде, поставь `[?]` и допиши
   конкретную инструкцию.
7. Любые новые проблемы, найденные по ходу, заводи в плане новыми пунктами,
   но не реализуй без отдельного подтверждения, кроме случаев, когда это
   блокирует выбранный fix.

### Приоритеты по умолчанию

Группа A - локальные правки с быстрыми тестами (рекомендую начать с них):

1. `1.11` - FTS rebuild только при первой миграции (под 7.15).
2. `1.12` - документ в `failed`, если rebuild упал (под 7.16).
3. `1.14` - empty rebuild не падает (под 7.18).
4. `2.12` - threadsafe `_record_metric` (под 7.21).
5. `6.7` - README про bundle (без теста).

Группа B - средний эффорт, требуют дизайна:

6. `1.13` - wall-clock deadline для chat attachments (под 7.17). Симметрично
   уже сделанному в compliance (1.9).
7. `3.6` - `/embedding/models` через `asyncio.to_thread` (под 7.24).
8. `1.15` - lock между `delete_index` и `retrieve` (под 7.19). Аккуратно с
   уже существующими per-index lock в `vector_store`.
9. `2.14` - depth-counter в `Registry.connect()` (без отдельного теста, но
   с тестом «вложенный rollback», его описание есть в 2.14).

Группа C - решения с пользователем (8.5-8.8); сначала спросить:

10. `2.11` / `7.20` - заменить monkeypatch-через-server на DI через
    `app.state`. Это касается всех тестов; сначала согласовать стратегию.
11. `2.13` / `8.5` - persistence Deep-RAG jobs (вариант A или B).
12. `3.4` / `8.6` - формат `/health` (что оставить публичным, что перенести в
    авторизованный endpoint).
13. `4.9` / `8.7` - TTL для FAISS cache (документация vs background eviction).
14. `5.5` / `8.8` - нормализация BM25 score (изменит ranking).
15. `3.5` / `7.23` - filter поведение при 401 от sidecar `/config`.

Не делай группу C без подтверждения.

### Жёсткие ограничения

- `openwebui_functions/zi_rag_filter.py` остаётся single-file. Это требование
  OpenWebUI Functions.
- Любая правка `zi_rag_filter.py` обязана быть синхронизирована с
  `openwebui_functions/zi_rag_filter.openwebui.json`. Регрессионный тест 7.14
  это проверяет. Используй `tools/build_filter.py` для синхронизации.
- Не ломать имена полей `config.json`, схему SQLite, API эндпоинты без
  согласования. Установка модуля идёт с нуля, но совместимость имён важна
  для документации и UI.
- Не править `openwebui_zi_rag_bundle.zip` вручную; для пересборки -
  `python3 tools/build_bundle.py`.
- Не запускать долгоживущие процессы (uvicorn, watchers) в shell.
- Не коммитить storage (`openwebui_zi_rag_storage/`, FAISS, SQLite, OCR-модели,
  кэши).

### Зафиксированные решения

- **Generation model (8.1):** дефолт пустой, пользователь выбирает в UI.
  Если модели нет в `/api/tags` - 409 с понятным detail. См. 1.5.
- **Strict api_key для localhost (8.2):** опционально, по умолчанию выкл.
  См. 3.1.
- **FAISS-индекс (8.3):** `index_type=auto`, Flat до 50k, HNSW выше. См. 4.1.
- **Pydantic v2 BaseSettings (8.4):** реализован через `pydantic-settings`,
  `config.json` как JSON settings source. См. 2.3.

### Git и окружение

- В `/media/onel/videoanalytics/ZI_RAG` нет `.git` - не пытайся делать
  коммиты.
- В системе есть только `python3` (не `python`).
- Используй venv `/media/onel/videoanalytics/app_final/.venv/bin/python`
  для всех команд.

### Стиль кода

- Python 3.10+, `from __future__ import annotations` уже стоит - сохранять.
- Типизация через `dict[str, Any]`, `list[...]`, `|` (mypy --strict проходит,
  не ломай).
- Никаких новых зависимостей без обоснования.
- Сообщения логов и UI - используют i18n через `messages.ru.json`/
  `messages.en.json`. Backend сообщения (HTTP detail) - на русском, кроме
  специально английских (`"Path is outside allowed_source_roots"` и подобных
  технических).
- Атомарная запись файлов - через tmp + `os.replace`.
- Все SQL-параметры - placeholder'ы, без f-string интерполяции значений.
- `# type: ignore[code]` всегда с конкретным error-code.

### Тесты

- Основной запуск:
  `cd /media/onel/videoanalytics/ZI_RAG && /media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
- Покрывать каждый закрытый баг минимум одним тестом из раздела 7.
- Не использовать сетевые/GPU/Ollama-вызовы в тестах: подменять
  `OllamaClient` / `OpenAIEmbeddingClient` фейками. В test-файле уже есть
  `FakeEmbeddingClient`, `CountingEmbeddingClient`, `CapturingEmbeddingClient`.
- Для concurrency-тестов держать таймауты короткими и детерминированными.

## END PROMPT

---

## Как пользоваться

1. Открой новый диалог.
2. Скопируй всё между `## START PROMPT` и `## END PROMPT`.
3. Дай агенту доступ к файлу `REVIEW_PLAN.md` (через #File или просьбу
   прочитать).
4. Проверь, что агент действительно вывел список открытых пунктов раунда 2,
   прежде чем правит код.
