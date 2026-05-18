"""OpenAPI tag/summary metadata applied to FastAPI routes."""

from __future__ import annotations

from typing import Any


OPENAPI_ROUTE_METADATA: dict[tuple[str, str], dict[str, Any]] = {
    ("GET", "/health"): {"tags": ["Health"], "summary": "Проверить состояние sidecar"},
    ("GET", "/health/full"): {"tags": ["Health"], "summary": "Проверить полное состояние sidecar"},
    ("GET", "/metrics"): {"tags": ["Health"], "summary": "Получить runtime metrics"},
    ("POST", "/ocr/cache/clear"): {"tags": ["OCR"], "summary": "Очистить OCR GPU cache"},
    ("GET", "/config"): {"tags": ["Config"], "summary": "Прочитать конфигурацию"},
    ("PUT", "/config"): {"tags": ["Config"], "summary": "Обновить конфигурацию"},
    ("GET", "/ollama/models"): {"tags": ["Models"], "summary": "Получить модели Ollama"},
    ("GET", "/embedding/models"): {"tags": ["Models"], "summary": "Получить embedding-модели"},
    ("GET", "/indexes"): {"tags": ["Indexes"], "summary": "Список индексов"},
    ("POST", "/indexes"): {"tags": ["Indexes"], "summary": "Создать индекс"},
    ("DELETE", "/indexes/{index_id}"): {"tags": ["Indexes"], "summary": "Удалить индекс"},
    ("GET", "/indexes/{index_id}/documents"): {"tags": ["Documents"], "summary": "Список документов индекса"},
    ("POST", "/indexes/{index_id}/documents/upload"): {"tags": ["Documents"], "summary": "Загрузить документ"},
    ("POST", "/indexes/{index_id}/documents/upload-batch"): {"tags": ["Documents"], "summary": "Загрузить пакет документов"},
    ("POST", "/chat-attachments/index"): {"tags": ["Chat Attachments"], "summary": "Индексировать chat attachments"},
    ("POST", "/indexes/{index_id}/documents/add-path"): {"tags": ["Documents"], "summary": "Добавить документы из пути"},
    ("DELETE", "/indexes/{index_id}/documents/{document_id}"): {"tags": ["Documents"], "summary": "Удалить документ"},
    ("POST", "/indexes/{index_id}/documents/delete"): {"tags": ["Documents"], "summary": "Удалить выбранные документы"},
    ("POST", "/indexes/{index_id}/documents/{document_id}/reindex"): {"tags": ["Documents"], "summary": "Переиндексировать документ"},
    ("POST", "/indexes/{index_id}/documents/reindex"): {"tags": ["Documents"], "summary": "Переиндексировать выбранные документы"},
    ("POST", "/indexes/{index_id}/rebuild"): {"tags": ["Indexes"], "summary": "Перестроить FAISS-индекс"},
    ("POST", "/retrieve"): {"tags": ["Retrieval"], "summary": "Выполнить RAG retrieval"},
    ("POST", "/analyze/jobs"): {"tags": ["Analysis"], "summary": "Запустить async Deep RAG анализ"},
    ("GET", "/analyze/jobs/{job_id}"): {"tags": ["Analysis"], "summary": "Получить async анализ"},
    ("GET", "/analyze/jobs/{job_id}/events"): {"tags": ["Analysis"], "summary": "Стримить async анализ через SSE"},
    ("POST", "/analyze/jobs/{job_id}/cancel"): {"tags": ["Analysis"], "summary": "Отменить async анализ"},
    ("POST", "/analyze"): {"tags": ["Analysis"], "summary": "Выполнить sync Deep RAG анализ"},
    ("POST", "/compliance/analyze"): {"tags": ["Compliance"], "summary": "Выполнить compliance-анализ"},
    ("GET", "/jobs"): {"tags": ["Jobs"], "summary": "Список indexing jobs"},
    ("GET", "/jobs/{job_id}"): {"tags": ["Jobs"], "summary": "Получить indexing job"},
    ("POST", "/jobs/{job_id}/cancel"): {"tags": ["Jobs"], "summary": "Отменить indexing job"},
    ("POST", "/indexes/{index_id}/jobs/cancel"): {"tags": ["Jobs"], "summary": "Отменить jobs индекса"},
    ("GET", "/ui"): {"tags": ["UI"], "summary": "Открыть админский UI"},
}


def apply_openapi_route_metadata(app: Any) -> None:
    if app is None:
        return
    for route in app.routes:
        path = str(getattr(route, "path", "") or "")
        methods = {str(method).upper() for method in getattr(route, "methods", set())}
        for method in sorted(methods):
            metadata = OPENAPI_ROUTE_METADATA.get((method, path))
            if not metadata:
                continue
            route.tags = list(metadata["tags"])
            route.summary = str(metadata["summary"])


__all__ = ["OPENAPI_ROUTE_METADATA", "apply_openapi_route_metadata"]
