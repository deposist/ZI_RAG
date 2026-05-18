# Промт для продолжения работы по 7.13 в новом диалоге

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

Все пункты разделов 1-6 закрыты ([x]). Открыт только п. 7.13 (статус `[~]` -
in progress). Это последний пункт плана.

### Что уже сделано по 7.13

1. Установлен `ruff` в venv:
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m pip install ruff`.
2. Прогон ruff чист:
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`
   -> All checks passed!
3. Найден и исправлен **реальный баг** через ruff F821:
   `openwebui_zi_rag/services/analysis.py` использовал `final_prompt`
   в синтезе Deep RAG `mode="answer"` без импорта. Добавлен импорт
   `final_prompt` из `.prompting`.
4. Тесты не сломаны: 102 passed.

### Что осталось

Прогон `mypy --strict` нашёл 62 ошибки в 14 файлах. Команда:

```
/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict
```

Категории ошибок:

A. **Опциональные зависимости без stubs** (надо или установить stubs, или
   добавить `# type: ignore[import-untyped]`):
   - `easyocr` -> import-not-found
   - `extract_msg` -> import-not-found
   - `pypdfium2` -> import-untyped
   - `pytesseract` -> import-untyped
   - `pandas` -> import-untyped (есть `pandas-stubs`)
   Решение: добавить в `pyproject.toml`/`mypy.ini` секцию для этих модулей
   или явные `# type: ignore`. Не устанавливать heavy stubs (`pandas-stubs`)
   в venv без согласования.

B. **Ложные срабатывания на try/except ImportError для FastAPI/uvicorn**:
   - `runtime.py:18,20`, `app.py:17,18`, `server.py:27`.
   - Конструкция вида `try: from fastapi import HTTPException; except: HTTPException = None`
     ломает strict mode. Решение: `# type: ignore[assignment, misc]` на
     строках присваивания внутри `except`.

C. **Реальные annotation gaps** (надо исправить):
   - `config.py:228` - функция без аннотации.
   - `config.py:358` - `int` vs `float` несовместимость.
   - `indexing/service.py:1242` - `list[float] | None` vs `list[float]`.
   - `indexing/service.py:1456` - tuple shape несовпадение.
   - `indexing/service.py:1585` - параметр без аннотации.
   - `services/health.py:113` - `list[str]` vs `dict` target (warnings list
     внутри embedding_model_dimension).
   - `services/jobs.py:54,87,116,169` - return annotation и
     `Item "None" has no attribute "is_set"` (нужны is None checks).
   - `routes/admin.py:32,41` - `_cached_payload` без аннотаций и
     `Returning Any`.
   - `routes/analyze.py:65,97,123` - `job: dict[str, Any]`, return type,
     `cancel_event.set()` под None-guard.

D. **server.py re-export вопросы**:
   - `server.py:30` - `_IMPORT_ERROR = exc` vs `None`. Решение:
     `_IMPORT_ERROR: Exception | None = None` на верхнем уровне.
   - `server.py:159-162` - "Module ... does not explicitly export attribute".
     Атрибуты `OllamaClient/extract_text/...` присваиваются в
     `_analysis_service` через `_sync_analysis_service_dependencies`, но
     mypy их не видит. Решение: либо добавить эти имена в `__all__` в
     `services/analysis.py` как Any-атрибуты модуля, либо `# type: ignore[attr-defined]`.
   - `server.py:163` - `make_ollama_client` сигнатура: `runtime` версия
     принимает `connect_timeout` аргументом, что отличает её от той, что в
     `services/analysis.py`. Унифицировать сигнатуры или явно `cast`.

E. `runtime.py` - множественные `Returning Any`:
   - `_resolve_config`, `model_cache`, `analysis_jobs`, `get_config`,
     `get_service`, `model_to_dict` (через `model.dict()`).
   - Решение: возвращать через `cast(SidecarConfig, ...)` /
     `cast(RagService, ...)` или явно типизировать `_runtime_state()`.

### Стратегия

Делай порциями A → B → C → D → E. После каждой порции:

1. Прогоняй mypy:
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m mypy openwebui_zi_rag/ --strict`
2. Прогоняй тесты:
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m pytest tests/test_openwebui_zi_rag.py --no-header -x`
3. Прогоняй ruff:
   `/media/onel/videoanalytics/app_final/.venv/bin/python -m ruff check openwebui_zi_rag/ openwebui_functions/ tools/ tests/`

Когда mypy `--strict` чист или каждая оставшаяся ошибка осознанно глушится
с пояснением в комментарии - закрой 7.13 как `[x]` в `REVIEW_PLAN.md` и
допиши строку в `## История изменений` с датой и итогом.

### Pyright

`pyright` сейчас не установлен. План его не требует обязательно (только как
альтернативу `mypy --strict`). Можно либо:
- установить (`pip install pyright`) и прогнать,
- либо отметить в плане, что mypy `--strict` достаточно для acceptance.

Согласовать со мной, если будет много расхождений с mypy.

### Git и окружение

- В `/media/onel/videoanalytics/ZI_RAG` нет `.git` - не пытайся
  делать коммиты.
- В системе есть только `python3` (не `python`).
- Используй venv `/media/onel/videoanalytics/app_final/.venv/bin/python`
  для всех команд.
- Не запускай долгие процессы (uvicorn, watchers).
- `openwebui_functions/zi_rag_filter.py` - single-file. Если меняешь -
  синхронизируй `openwebui_functions/zi_rag_filter.openwebui.json` через
  `tools/build_filter.py`.

### Стиль

- Python 3.10+, типизация через `dict[str, Any]`, `|`,
  `from __future__ import annotations`.
- `# type: ignore[code]` с указанием конкретного error-code, не пустой.
- Не вводить новые зависимости без обоснования. Стабы (`pandas-stubs`,
  `types-extract_msg` и т.п.) - только если нет другого пути и согласовано.
- Помнить о тестах: они активно используют `monkeypatch.setattr(
  rag_server, ...)` и `app.dependency_overrides`. Не ломать identity
  `rag_server.get_config` и публичную поверхность server.py.

### Первый шаг

1. Прочитай `REVIEW_PLAN.md` целиком (особенно раздел 7.13 и историю).
2. Запусти mypy и сверь со списком категорий A-E выше.
3. Предложи порядок (обычно A → B → C → D → E) и начинай с A.

## END PROMPT

---

## Как пользоваться

1. Открой новый диалог.
2. Скопируй всё между `## START PROMPT` и `## END PROMPT`.
3. Дай агенту доступ к файлу `REVIEW_PLAN.md` (через #File или
   просьбу прочитать).
4. Проверь, что агент действительно увидел план и mypy-output, прежде чем
   правит код.
