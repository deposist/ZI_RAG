"""
title: Enhanced RAG Sidecar
author: local
version: 0.9.0
required_open_webui_version: 0.9.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class Filter:
    REQUEST_STATE_TTL_SEC = 600.0

    class Valves(BaseModel):
        sidecar_url: str = "http://host.docker.internal:8766"
        api_key: str = ""
        timeout_sec: int = 60
        sync_sidecar_admin_config: bool = Field(
            default=True,
            description="Use admin defaults from the sidecar /config screen.",
        )
        include_sources: bool = True
        rag_enabled: bool = True
        deep_analysis_enabled: bool = Field(
            default=True,
            description="Admin default: automatically use multi-pass analysis for complex questions.",
        )
        deep_final_answer: bool = Field(
            default=True,
            description="Admin default: return sidecar's final multi-pass answer.",
        )
        deep_force_all: bool = Field(
            default=False,
            description="Force multi-pass analysis for every question.",
        )
        deep_trigger_phrases: str = Field(
            default=(
                "проанализируй все\n"
                "проанализировать все\n"
                "проверь все\n"
                "проверь всё\n"
                "проверь все документы\n"
                "проверь весь пакет\n"
                "сравни\n"
                "сравнить\n"
                "полный перечень\n"
                "все требования\n"
                "все нарушения\n"
                "найди противоречия\n"
                "найти противоречия\n"
                "сделай отчет\n"
                "сделай отчёт\n"
                "подготовь отчет\n"
                "подготовь отчёт\n"
                "ничего не пропусти\n"
                "полный анализ\n"
                "по всем документам"
            ),
            description="One auto deep trigger phrase per line.",
        )
        compliance_enabled: bool = Field(
            default=True,
            description="Admin default: enable attached-document compliance checks.",
        )
        compliance_auto_enabled: bool = Field(
            default=True,
            description="Automatically run compliance checks for attached documents and matching questions.",
        )
        compliance_allow_user_index_override: bool = Field(
            default=True,
            description="Allow users and /check index:... to override compliance NMD indexes.",
        )
        compliance_trigger_phrases: str = Field(
            default=(
                "проверь на соответствие\n"
                "проверка нмд\n"
                "соответствует ли\n"
                "найди нарушения\n"
                "найти нарушения\n"
                "сделай акт\n"
                "подготовь акт\n"
                "проведи проверку\n"
                "проверить документ\n"
                "матрица соответствия\n"
                "compliance"
            ),
            description="One compliance auto-trigger phrase per line.",
        )
        compliance_requirement_top_k: int = Field(
            default=24,
            description="How many NMD requirements to retrieve for each checked section.",
        )
        compliance_section_chars: int = Field(
            default=8000,
            description="Target section size for checked attachments.",
        )
        compliance_max_sections: int = Field(
            default=80,
            description="Maximum checked attachment sections.",
        )
        compliance_timeout_sec: int = Field(
            default=1200,
            description="Timeout for synchronous compliance analysis.",
        )
        compliance_max_files: int = Field(
            default=10,
            description="Maximum attached files read by the filter before sending to sidecar.",
        )
        compliance_max_file_mb: int = Field(
            default=256,
            description="Maximum single attached file size read by the filter.",
        )
        chat_attachments_enabled: bool = Field(
            default=True,
            description="Automatically index OpenWebUI chat attachments in a per-chat sidecar index.",
        )
        chat_attachment_index_prefix: str = Field(
            default="owui_chat_",
            description="Prefix for per-chat attachment indexes created by the sidecar.",
        )
        chat_attachment_max_files: int = Field(
            default=10,
            description="Maximum attached files indexed for one chat request.",
        )
        chat_attachment_max_file_mb: int = Field(
            default=256,
            description="Maximum single attached file size indexed from OpenWebUI.",
        )
        chat_attachment_timeout_sec: int = Field(
            default=900,
            description="Timeout for synchronous chat attachment indexing.",
        )
        retrieval_top_k: int = Field(
            default=70,
            description="How many sidecar chunks to retrieve before prompt packing.",
        )
        min_relevance_score: float = Field(
            default=0.50,
            description="Minimum transformed cosine score accepted from the sidecar.",
        )
        adaptive_score_margin: float = Field(
            default=0.20,
            description="Keep chunks no farther than this from the best retrieved score.",
        )
        max_prompt_chunks: int = Field(
            default=24,
            description="Maximum filtered chunks considered for prompt packing.",
        )
        min_query_term_hits: int = Field(
            default=1,
            description="Minimum query term matches for lexical noise filtering.",
        )
        max_context_chars: int = Field(
            default=32000,
            description="Hard character budget for injected RAG context.",
        )
        context_batch_chars: int = Field(
            default=10000,
            description="Target character budget for one full-text RAG batch.",
        )
        max_context_batches: int = Field(
            default=3,
            description="Maximum number of full-text RAG batches to inject.",
        )
        max_compact_sources: int = Field(
            default=8,
            description="Maximum overflow chunks listed compactly with quotes.",
        )
        deep_top_k: int = Field(
            default=70,
            description="How many chunks the sidecar retrieves for multi-pass analysis.",
        )
        deep_batch_chars: int = Field(
            default=10000,
            description="Target character budget for one multi-pass batch.",
        )
        deep_max_batches: int = Field(
            default=10,
            description="Maximum number of multi-pass batches.",
        )
        deep_timeout_sec: int = Field(
            default=900,
            description="Timeout for synchronous multi-pass sidecar analysis.",
        )
        context_template: str = (
            "Используй контекст RAG ниже как приоритетный источник. Контекст может "
            "быть разбит на пачки: просматривай все пачки последовательно и не "
            "игнорируй поздние пачки только из-за их номера. Не придумывай локаторы "
            "и цитаты. Если ответа в контексте нет, честно скажи, что в базе знаний "
            "ответ не найден.\n\n"
            "{knowledge}"
        )

    class UserValves(BaseModel):
        rag_enabled: bool = True
        deep_analysis_enabled: bool = Field(
            default=False,
            description="Automatically use multi-pass analysis for complex questions.",
        )
        deep_final_answer: bool = Field(
            default=True,
            description="Return sidecar's final multi-pass answer instead of only deep context.",
        )
        index_ids: str = Field(
            default="",
            description="Comma-separated sidecar index ids. Empty means sidecar default index.",
        )
        compliance_enabled: bool = Field(
            default=True,
            description="Enable attached-document compliance checks.",
        )
        compliance_index_ids: str = Field(
            default="",
            description="Comma-separated NMD index ids for compliance checks. Empty means sidecar default.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "https://docs.openwebui.com/img/logo.svg"
        self._sources_by_key: dict[str, dict[str, Any]] = {}
        self._deep_answers_by_key: dict[str, dict[str, Any]] = {}
        self._admin_config_cache: dict[str, Any] = {}
        self._admin_config_loaded_at = 0.0
        self._admin_config_unavailable = False
        self._admin_config_error_message = ""
        self._admin_config_warned_at = 0.0

    def _request_key(self, body: dict, metadata: Optional[dict], user: Optional[dict]) -> str:
        metadata = metadata or {}
        body_metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        user = user or {}
        parts: list[str] = []
        stable_id_found = False
        for field in ("chat_id", "message_id", "session_id"):
            value = metadata.get(field) or body_metadata.get(field) or body.get(field)
            if value:
                stable_id_found = True
                parts.append(f"{field}:{value}")
        user_id = user.get("id")
        if user_id:
            parts.append(f"user:{user_id}")
        text_hash = self._request_text_hash(body)
        if text_hash:
            parts.append(f"text:{text_hash}")
        if not stable_id_found:
            parts.append(f"body:{id(body)}")
        return "|".join(str(part) for part in parts if str(part).strip()) or f"body:{id(body)}"

    def _request_text_hash(self, body: dict) -> str:
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            return ""
        last_user = self._last_user_message(messages)
        text = self._message_text(last_user or {}).strip()
        marker = "Вопрос пользователя:\n"
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
        if not text:
            return ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _state_entry(self, value: Any) -> dict[str, Any]:
        return {"created_at": time.monotonic(), "value": value}

    def _cleanup_request_state(self) -> None:
        now = time.monotonic()
        for storage in (self._sources_by_key, self._deep_answers_by_key):
            stale_keys = []
            for key, entry in storage.items():
                if not isinstance(entry, dict) or "created_at" not in entry:
                    stale_keys.append(key)
                    continue
                try:
                    age = now - float(entry.get("created_at") or 0.0)
                except Exception:
                    age = self.REQUEST_STATE_TTL_SEC + 1.0
                if age > self.REQUEST_STATE_TTL_SEC:
                    stale_keys.append(key)
            for key in stale_keys:
                storage.pop(key, None)

    def _store_sources(self, key: str, sources: list[dict[str, Any]]) -> None:
        self._sources_by_key[key] = self._state_entry(sources)

    def _store_deep_answer(self, key: str, answer: str) -> None:
        self._deep_answers_by_key[key] = self._state_entry(answer)

    def _pop_deep_answer(self, key: str) -> str:
        entry = self._deep_answers_by_key.pop(key, None)
        if isinstance(entry, dict) and "value" in entry:
            return str(entry.get("value") or "")
        return str(entry or "")

    def _model_values(self, value: Any) -> dict[str, Any]:
        if isinstance(value, BaseModel):
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value.dict()
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _user_valves(
        self,
        user: Optional[dict],
        user_valves: Optional[UserValves | dict],
    ) -> UserValves:
        values: dict[str, Any] = {}
        if isinstance(user, dict):
            values.update(self._model_values(user.get("valves")))
        values.update(self._model_values(user_valves))
        if "default_index_ids" in values and "index_ids" not in values:
            values["index_ids"] = values["default_index_ids"]
        known_fields = set(getattr(self.UserValves, "model_fields", None) or self.UserValves.__fields__)
        values = {key: value for key, value in values.items() if key in known_fields}
        return self.UserValves(**values)

    def _last_user_message(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None

    def _message_text(self, message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "\n".join(parts)
        return str(content)

    def _parse_ids(self, value: str) -> list[str]:
        return [part.strip() for part in (value or "").split(",") if part.strip()]

    def _locator(self, doc: dict[str, Any]) -> str:
        value = str(doc.get("locator") or "").strip()
        if value:
            return value
        text = str(doc.get("text") or "")
        locators = []
        for match in re.finditer(r"\[([^\]]*(?:абз\.|стр\.|пункт|таблица|лист|строка|тело письма)[^\]]*)\]", text, re.I):
            locator = match.group(1).strip()
            if locator and locator not in locators:
                locators.append(locator)
            if len(locators) >= 3:
                break
        return " / ".join(locators) or f"chunk {doc.get('chunk_no')}"

    def _quote(self, doc: dict[str, Any], max_chars: int = 420) -> str:
        value = str(doc.get("quote") or "").strip()
        if not value:
            value = re.sub(r"\[[^\]]+\]", " ", str(doc.get("text") or ""))
        value = re.sub(r"(?:\s*\|\s*(?:-|–|—|v|V|x|X|✓|✔)?\s*){3,}", " ", value)
        value = re.sub(r"\s+\|(?=\s*[.;,]|$)", " ", value)
        value = re.sub(r"\|\s+", "| ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = value.strip(" |")
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.valves.api_key:
            headers["X-API-Key"] = self.valves.api_key
        return headers

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        url = self.valves.sidecar_url.rstrip("/") + path
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers())
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_sec or self.valves.timeout_sec,
            ) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RAG sidecar HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RAG sidecar unavailable: {exc}") from exc

    def _post_json(
        self, path: str, payload: dict[str, Any], timeout_sec: int | None = None
    ) -> dict[str, Any]:
        return self._request_json(path, payload=payload, timeout_sec=timeout_sec)

    def _get_json(self, path: str, timeout_sec: int | None = None) -> dict[str, Any]:
        return self._request_json(path, payload=None, timeout_sec=timeout_sec)

    def _set_admin_config_available(self) -> None:
        if self._admin_config_unavailable:
            print("ZI_RAG sidecar /config доступен снова; синхронизация admin valves восстановлена.")
        self._admin_config_unavailable = False
        self._admin_config_error_message = ""
        self._admin_config_warned_at = 0.0

    def _set_admin_config_unavailable(self, error_message: str) -> None:
        message = str(error_message or "unknown error").strip()
        if not self._admin_config_unavailable:
            print(
                "ZI_RAG sidecar /config недоступен "
                f"({message}). Фильтр работает на собственных valves."
            )
        self._admin_config_unavailable = True
        self._admin_config_error_message = message

    def _iter_sse_events(self, path: str, timeout_sec: int):
        url = self.valves.sidecar_url.rstrip("/") + path
        headers = self._headers()
        headers["Accept"] = "text/event-stream"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            event_name = "message"
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw_data)
                        except Exception:
                            payload = {"data": raw_data}
                        yield event_name, payload
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip() or "message"
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    payload = json.loads(raw_data)
                except Exception:
                    payload = {"data": raw_data}
                yield event_name, payload

    def _next_sse_event(self, iterator):
        try:
            return next(iterator)
        except StopIteration:
            return None

    def _multipart_filename(self, value: Any) -> str:
        name = Path(str(value or "upload")).name.replace("\x00", "").strip()
        name = re.sub(r'[\r\n\t"\\]', "_", name) or "upload"
        name = "".join(ch if ch.isprintable() else "_" for ch in name)
        return name[:180] or "upload"

    def _multipart_content_type(self, value: Any) -> str:
        content_type = str(value or "application/octet-stream").strip()
        content_type = re.sub(r"[\r\n]", "", content_type)
        if (
            not content_type
            or len(content_type) > 120
            or not re.fullmatch(
                r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+(?:\s*;\s*[A-Za-z0-9_.-]+=[A-Za-z0-9_.-]+)*",
                content_type,
            )
        ):
            return "application/octet-stream"
        return content_type

    def _content_bytes(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value or b"")

    def _post_multipart(
        self,
        path: str,
        payload: dict[str, Any],
        files: list[dict[str, Any]],
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        boundary = f"zi-rag-{secrets.token_hex(16)}"
        chunks: list[bytes] = []

        def add_part(name: str, data: bytes, filename: str = "", content_type: str = "") -> None:
            disposition = f'form-data; name="{name}"'
            if filename:
                disposition += f'; filename="{self._multipart_filename(filename)}"'
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(f"Content-Disposition: {disposition}\r\n".encode("utf-8"))
            if content_type:
                chunks.append(f"Content-Type: {self._multipart_content_type(content_type)}\r\n".encode("utf-8"))
            chunks.append(b"\r\n")
            chunks.append(data)
            chunks.append(b"\r\n")

        add_part(
            "payload",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
        for item in files:
            add_part(
                "files",
                self._content_bytes(item.get("content") or b""),
                filename=item.get("filename") or "upload",
                content_type=str(item.get("content_type") or "application/octet-stream"),
            )
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        if self.valves.api_key:
            headers["X-API-Key"] = self.valves.api_key
        request = urllib.request.Request(
            self.valves.sidecar_url.rstrip("/") + path,
            data=b"".join(chunks),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_sec or self.valves.timeout_sec,
            ) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RAG sidecar HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RAG sidecar unavailable: {exc}") from exc

    def _sidecar_admin_config(self) -> dict[str, Any]:
        if not getattr(self.valves, "sync_sidecar_admin_config", True):
            return {}
        now = time.monotonic()
        if now - self._admin_config_loaded_at < 5:
            return self._admin_config_cache
        try:
            data = self._get_json("/config", timeout_sec=min(self.valves.timeout_sec, 2))
        except Exception as exc:
            self._set_admin_config_unavailable(str(exc))
            data = {}
        else:
            if isinstance(data, dict):
                self._set_admin_config_available()
            else:
                self._set_admin_config_unavailable("unexpected /config response")
                data = {}
        self._admin_config_cache = data if isinstance(data, dict) else {}
        self._admin_config_loaded_at = now
        return self._admin_config_cache

    async def _maybe_emit_admin_config_warning(self, event_emitter=None) -> None:
        if not event_emitter or not self._admin_config_unavailable:
            return
        now = time.monotonic()
        if now - self._admin_config_warned_at < 60.0:
            return
        self._admin_config_warned_at = now
        reason = self._admin_config_error_message or "unknown error"
        await event_emitter(
            {
                "type": "notification",
                "data": {
                    "type": "warning",
                    "content": (
                        f"ZI_RAG sidecar /config недоступен ({reason}). "
                        "Фильтр работает на собственных valves."
                    ),
                },
            }
        )

    def _setting(self, name: str, default: Any = None) -> Any:
        admin = self._sidecar_admin_config()
        aliases = {
            "min_relevance_score": "score_threshold",
        }
        if name in admin and admin[name] not in (None, ""):
            return admin[name]
        alias = aliases.get(name)
        if alias and alias in admin and admin[alias] not in (None, ""):
            return admin[alias]
        return getattr(self.valves, name, default)

    def _bool_setting(self, name: str, default: bool) -> bool:
        value = self._setting(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on", "enable", "enabled"}
        return bool(value)

    def _int_valve(self, name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self._setting(name, default) or default)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))

    def _float_valve(self, name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(self._setting(name, default) or default)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))

    def _retrieval_top_k(self) -> int:
        return self._int_valve("retrieval_top_k", 70, 1, 500)

    def _min_relevance_score(self) -> float:
        return self._float_valve("min_relevance_score", 0.50, 0.0, 1.0)

    def _deep_top_k(self) -> int:
        return self._int_valve("deep_top_k", 70, 1, 500)

    def _deep_batch_chars(self) -> int:
        return self._int_valve("deep_batch_chars", 10000, 1500, 64000)

    def _deep_max_batches(self) -> int:
        return self._int_valve("deep_max_batches", 10, 1, 50)

    def _deep_timeout_sec(self) -> int:
        return self._int_valve("deep_timeout_sec", 900, 60, 3600)

    def _compliance_timeout_sec(self) -> int:
        return self._int_valve("compliance_timeout_sec", 1200, 60, 7200)

    def _compliance_requirement_top_k(self) -> int:
        return self._int_valve("compliance_requirement_top_k", 24, 1, 100)

    def _compliance_section_chars(self) -> int:
        return self._int_valve("compliance_section_chars", 8000, 1500, 64000)

    def _compliance_max_sections(self) -> int:
        return self._int_valve("compliance_max_sections", 80, 1, 500)

    def _compliance_max_files(self) -> int:
        return self._int_valve("compliance_max_files", 10, 1, 100)

    def _compliance_max_file_mb(self) -> int:
        return self._int_valve("compliance_max_file_mb", 256, 1, 4096)

    def _chat_attachment_max_files(self) -> int:
        return self._int_valve("chat_attachment_max_files", 10, 1, 100)

    def _chat_attachment_max_file_mb(self) -> int:
        return self._int_valve("chat_attachment_max_file_mb", 256, 1, 4096)

    def _chat_attachment_timeout_sec(self) -> int:
        return self._int_valve("chat_attachment_timeout_sec", 900, 60, 3600)

    def _context_budgets(self) -> tuple[int, int, int, int]:
        max_chars = self._int_valve("max_context_chars", 32000, 4000, 128000)
        batch_chars = self._int_valve("context_batch_chars", 10000, 1500, 32000)
        max_batches = self._int_valve("max_context_batches", 3, 1, 16)
        compact_sources = self._int_valve("max_compact_sources", 8, 0, 200)
        batch_chars = min(batch_chars, max_chars)
        return max_chars, batch_chars, max_batches, compact_sources

    def _trim_text(self, text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        trimmed = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
        return (trimmed or text[:max_chars]).rstrip() + "..."

    def _score(self, doc: dict[str, Any]) -> float:
        try:
            return float(doc.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _query_terms(self, query: str) -> list[str]:
        stop_words = {
            "что", "это", "как", "для", "при", "или", "если", "где", "когда",
            "какие", "какой", "какая", "какое", "кто", "чем", "про", "над",
            "под", "без", "есть", "такое", "нужно", "нужна", "нужен",
            "документ", "документы", "регламент", "порядок", "методика",
            "политика", "раздел", "пункт", "what", "where", "when", "which",
            "that", "this", "with", "from", "about", "все", "всё",
        }
        stop_stems = {
            "нмд", "норма", "метод", "докум", "регла", "поряд", "полны",
            "переч", "проан", "списо", "пункт", "котор", "прове", "работ",
            "свой", "наруш",
        }
        terms: list[str] = []
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", (query or "").lower()):
            token = token.replace("ё", "е")
            if token in stop_words:
                continue
            if len(token) == 5 and token[-1] in "аеиоуыэюя":
                token = token[:-1]
            elif len(token) > 5:
                token = token[:5]
            if token in stop_words or token in stop_stems:
                continue
            if token and token not in terms:
                terms.append(token)
        return terms[:12]

    def _query_term_hits(self, terms: list[str], doc: dict[str, Any]) -> int:
        if not terms:
            return 0
        haystack = " ".join(
            str(doc.get(key) or "")
            for key in ("source", "locator", "quote", "text")
        ).lower().replace("ё", "е")
        return sum(1 for term in terms if term in haystack)

    def _dedupe_docs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for doc in docs:
            text = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip().lower()
            key = ("text", text[:1200]) if text else (
                str(doc.get("source") or "").strip().lower(),
                str(doc.get("chunk_id") or doc.get("chunk_no") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def _filter_docs_for_prompt(
        self, query: str, docs: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        deduped = self._dedupe_docs(docs)
        if not deduped:
            return [], {
                "raw": len(docs),
                "deduped": 0,
                "filtered": 0,
                "best_score": 0.0,
                "score_floor": self._min_relevance_score(),
            }

        min_score = self._min_relevance_score()
        margin = self._float_valve("adaptive_score_margin", 0.20, 0.0, 0.5)
        max_docs = self._int_valve("max_prompt_chunks", 24, 1, 200)
        min_hits = self._int_valve("min_query_term_hits", 1, 0, 8)
        terms = self._query_terms(query)

        best_score = max(self._score(doc) for doc in deduped)
        if best_score < min_score:
            return [], {
                "raw": len(docs),
                "deduped": len(deduped),
                "filtered": 0,
                "best_score": best_score,
                "score_floor": min_score,
            }

        score_floor = max(min_score, best_score - margin)
        candidates = [doc for doc in deduped if self._score(doc) >= score_floor]

        required_hits = min_hits
        ordered_by_backfill = False
        if terms and min_hits > 0:
            if len(terms) >= 4 and min_hits <= 1:
                required_hits = 2
            lexical = [
                doc for doc in candidates
                if self._query_term_hits(terms, doc) >= required_hits
            ]
            if len(lexical) < min(3, len(candidates)) and required_hits > 1:
                lexical = [
                    doc for doc in candidates
                    if self._query_term_hits(terms, doc) >= min_hits
                ]
            if lexical:
                candidates = lexical

            backfill_hits = max(required_hits + 2, 4)
            backfill = [
                doc for doc in deduped
                if self._score(doc) >= min_score
                and self._query_term_hits(terms, doc) >= backfill_hits
            ]
            if backfill:
                primary = sorted(
                    candidates,
                    key=lambda doc: (self._query_term_hits(terms, doc), self._score(doc)),
                    reverse=True,
                )
                merged_candidates: list[dict[str, Any]] = list(primary)
                seen: set[tuple[str, str, int]] = set()
                for doc in primary:
                    text = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip().lower()
                    key = (
                        "text" if text else str(doc.get("source") or ""),
                        text[:1200] if text else str(doc.get("chunk_id") or doc.get("chunk_no") or ""),
                        0,
                    )
                    seen.add(key)
                backfill_only: list[dict[str, Any]] = []
                for doc in backfill:
                    text = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip().lower()
                    key = (
                        "text" if text else str(doc.get("source") or ""),
                        text[:1200] if text else str(doc.get("chunk_id") or doc.get("chunk_no") or ""),
                        0,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    backfill_only.append(doc)
                backfill_only.sort(
                    key=lambda doc: (self._query_term_hits(terms, doc), self._score(doc)),
                    reverse=True,
                )
                merged_candidates.extend(backfill_only)
                candidates = merged_candidates
                ordered_by_backfill = True

        if not ordered_by_backfill:
            candidates.sort(
                key=lambda doc: (
                    self._query_term_hits(terms, doc),
                    self._score(doc),
                ),
                reverse=True,
            )
        filtered = candidates[:max_docs]
        return filtered, {
            "raw": len(docs),
            "deduped": len(deduped),
            "filtered": len(filtered),
            "best_score": best_score,
            "score_floor": score_floor,
        }

    def _format_doc_block(self, index: int, doc: dict[str, Any]) -> str:
        source = doc.get("source") or "unknown"
        locator = self._locator(doc)
        quote = self._quote(doc)
        text = str(doc.get("text") or "").strip()
        return (
            f"[{index}] Документ: {source} · score={self._score(doc):.3f}\n"
            f"Локатор: {locator}\n"
            f"Цитата: «{quote}»\n"
            f"Фрагмент:\n{text}"
        )

    def _compact_item_line(self, index: int, doc: dict[str, Any]) -> str:
        source = doc.get("source") or "unknown"
        locator = self._locator(doc)
        quote = self._quote(doc, max_chars=320)
        return (
            f"[{index}] Документ: {source} · score={self._score(doc):.3f}; "
            f"Локатор: {locator}; Цитата: «{quote}»"
        )

    def _pack_docs_into_batches(
        self, docs: list[dict[str, Any]]
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
        max_chars, batch_chars, max_batches, _ = self._context_budgets()
        batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []
        current_chars = 0
        committed_chars = 0
        included_items: list[dict[str, Any]] = []
        overflow_items: list[dict[str, Any]] = []

        def flush_current() -> None:
            nonlocal current_batch, current_chars, committed_chars
            if not current_batch:
                return
            batches.append(current_batch)
            committed_chars += current_chars
            current_batch = []
            current_chars = 0

        for index, raw_doc in enumerate(docs, start=1):
            doc = dict(raw_doc)
            original_text = str(doc.get("text") or "").strip()
            block = self._format_doc_block(index, doc)

            if len(block) > batch_chars:
                overhead_doc = dict(doc)
                overhead_doc["text"] = ""
                overhead = len(self._format_doc_block(index, overhead_doc))
                text_budget = max(300, batch_chars - overhead - 120)
                doc["text"] = (
                    self._trim_text(original_text, text_budget)
                    + "\n[Фрагмент усечён по бюджету пачки.]"
                )
                block = self._format_doc_block(index, doc)

            item = {"index": index, "doc": doc, "block": block}
            block_chars = len(block) + 2

            if current_batch and current_chars + block_chars > batch_chars:
                flush_current()

            budget_exhausted = committed_chars + current_chars + block_chars > max_chars
            batch_limit_exhausted = len(batches) >= max_batches
            if budget_exhausted or batch_limit_exhausted:
                overflow_items.append({"index": index, "doc": raw_doc})
                continue

            current_batch.append(item)
            current_chars += block_chars
            included_items.append(item)

        if current_batch:
            if (
                len(batches) < max_batches
                and committed_chars + current_chars <= max_chars
            ):
                flush_current()
            else:
                overflow_items.extend(
                    {"index": item["index"], "doc": item["doc"]} for item in current_batch
                )

        return batches, included_items, overflow_items

    def _format_knowledge(
        self, docs: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        batches, included_items, overflow_items = self._pack_docs_into_batches(docs)
        _, _, _, compact_limit = self._context_budgets()
        compact_items = overflow_items[:compact_limit]
        included_count = len(included_items)
        overflow_count = len(overflow_items)

        lines = [
            f"Найдено RAG-фрагментов: {len(docs)}.",
            (
                "Полнотекстово включено: "
                f"{included_count} фрагментов в {len(batches)} пачках."
            ),
        ]
        if overflow_count:
            lines.append(
                "Чтобы OpenWebUI не срезал контекст, остальные фрагменты не "
                "развёрнуты полностью: ниже дан компактный список с локаторами и цитатами."
            )

        for batch_index, batch in enumerate(batches, start=1):
            lines.append(f"\n=== Пачка {batch_index}/{len(batches)} ===")
            lines.extend(item["block"] for item in batch)

        if compact_items:
            lines.append(
                f"\n=== Компактная пачка остальных источников "
                f"({len(compact_items)} из {overflow_count}) ==="
            )
            lines.extend(
                self._compact_item_line(item["index"], item["doc"])
                for item in compact_items
            )
            if overflow_count > len(compact_items):
                lines.append(
                    f"... ещё {overflow_count - len(compact_items)} фрагментов "
                    "не включены из-за бюджета контекста."
                )

        source_docs = [item["doc"] for item in included_items]
        source_docs.extend(item["doc"] for item in compact_items)
        stats = {
            "retrieved": len(docs),
            "included": included_count,
            "batches": len(batches),
            "compact": len(compact_items),
            "overflow": overflow_count,
        }
        return "\n\n".join(lines).strip(), source_docs, stats

    def _format_source_lines(self, docs: list[dict[str, Any]], limit: int | None = None) -> str:
        lines = []
        seen = set()
        for doc in docs:
            if limit is not None and len(lines) >= limit:
                break
            source = doc.get("source") or "unknown"
            locator = self._locator(doc)
            quote = self._quote(doc, max_chars=260)
            label = f"{source}, {locator}"
            if label in seen:
                continue
            seen.add(label)
            lines.append(
                f"{len(lines) + 1}. Документ: {source}; "
                f"Локатор: {locator}; Цитата: «{quote}»"
            )
        return "\n".join(lines)

    def _source_policy(self, docs: list[dict[str, Any]]) -> str:
        if not self._bool_setting("include_sources", True):
            return ""
        _, _, _, compact_limit = self._context_budgets()
        source_lines = self._format_source_lines(docs, limit=compact_limit or None)
        if not source_lines:
            return ""
        return (
            "Каждый фактический вывод подкрепляй ссылкой на документ, локатор "
            "(пункт, абзац, страницу, таблицу или строку) и короткую цитату. "
            "В конце ответа обязательно сгенерируй список источников внутри HTML-спойлера "
            "строго такого вида:\n\n"
            "<details>\n"
            "<summary>Источники</summary>\n\n"
            "1. Документ: ...; Локатор: ...; Цитата: «...»\n"
            "</details>\n\n"
            "Не пиши список источников вне этого блока. Не добавляй в источники ничего, "
            "чего нет в RAG-контексте. Используй для спойлера эти источники:\n"
            f"{source_lines}"
        )

    def _inject_context(self, user_message: dict[str, Any], context: str) -> None:
        prefix = (
            "Контекст RAG для ответа:\n"
            f"{context}\n\n"
            "Вопрос пользователя:\n"
        )
        content = user_message.get("content", "")
        if isinstance(content, str):
            user_message["content"] = f"{prefix}{content}"
            return
        if isinstance(content, list):
            user_message["content"] = [{"type": "text", "text": prefix}, *content]
            return
        user_message["content"] = f"{prefix}{content}"

    def _set_message_text(self, message: dict[str, Any], text: str) -> None:
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = text
            return
        if isinstance(content, list):
            replaced = False
            updated = []
            for item in content:
                if (
                    not replaced
                    and isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    new_item = dict(item)
                    new_item["text"] = text
                    updated.append(new_item)
                    replaced = True
                else:
                    updated.append(item)
            if not replaced:
                updated.insert(0, {"type": "text", "text": text})
            message["content"] = updated
            return
        message["content"] = text

    def _deep_marker(self, query: str) -> tuple[bool, str]:
        match = re.match(r"^\s*/deep(?:\s+|$)", query or "", re.I)
        if not match:
            return False, query
        cleaned = (query or "")[match.end():].strip()
        return True, cleaned or query

    def _check_marker(self, query: str) -> tuple[bool, str]:
        match = re.match(r"^\s*/check(?:\s+|$)", query or "", re.I)
        if not match:
            return False, query
        cleaned = (query or "")[match.end():].strip()
        return True, cleaned or "проверь приложенные документы на соответствие НМД"

    def _extract_check_index_override(self, query: str) -> tuple[list[str], str]:
        index_ids: list[str] = []

        def replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            index_ids.extend(self._parse_ids(raw))
            return " "

        cleaned = re.sub(
            r"(?:^|\s)(?:index|индекс):([0-9A-Za-zА-Яа-яЁё_.:,/-]+)",
            replace,
            query or "",
            flags=re.I,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return index_ids, cleaned or query

    def _deep_trigger_phrases(self) -> list[str]:
        value = self._setting("deep_trigger_phrases", self.valves.deep_trigger_phrases)
        if isinstance(value, list):
            phrases = [str(item).strip().lower() for item in value if str(item).strip()]
        else:
            phrases = [
                item.strip().lower()
                for item in re.split(r"[\n,]+", str(value or ""))
                if item.strip()
            ]
        return phrases or [
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

    def _is_complex_query(self, query: str) -> bool:
        value = (query or "").lower()
        patterns = self._deep_trigger_phrases()
        return any(pattern in value for pattern in patterns)

    def _compliance_trigger_phrases(self) -> list[str]:
        value = self._setting("compliance_trigger_phrases", self.valves.compliance_trigger_phrases)
        if isinstance(value, list):
            phrases = [str(item).strip().lower() for item in value if str(item).strip()]
        else:
            phrases = [
                item.strip().lower()
                for item in re.split(r"[\n,]+", str(value or ""))
                if item.strip()
            ]
        return phrases or [
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

    def _is_compliance_query(self, query: str) -> bool:
        value = (query or "").lower()
        return any(pattern in value for pattern in self._compliance_trigger_phrases())

    def _merge_file_items(self, *file_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        for file_list in file_lists:
            for item in file_list or []:
                if not isinstance(item, dict):
                    continue
                key = json.dumps(
                    {
                        "type": item.get("type"),
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "url": item.get("url"),
                    },
                    sort_keys=True,
                    default=str,
                )
                if key in seen:
                    continue
                seen.add(key)
                files.append(item)
        return files

    def _request_file_items(self, body: dict, metadata: Optional[dict]) -> list[dict[str, Any]]:
        metadata = metadata or body.get("metadata") or {}
        user_message = metadata.get("user_message") or {}
        return [
            item
            for item in self._merge_file_items(
                body.get("files") or [],
                metadata.get("files") or [],
                user_message.get("files") or [],
            )
            if item.get("id")
            and item.get("type") in {"file", "doc", None, ""}
            and not str(item.get("content_type") or "").startswith("image/")
            and not item.get("collection_name")
            and not item.get("collection_names")
        ]

    def _filter_file_list(self, files: Any, processed_ids: set[str]) -> list[dict[str, Any]]:
        kept = []
        for item in files or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "")
            item_type = item.get("type")
            if (
                item_id in processed_ids
                and item_type in {"file", "doc", None, ""}
                and not str(item.get("content_type") or "").startswith("image/")
                and not item.get("collection_name")
                and not item.get("collection_names")
            ):
                continue
            kept.append(item)
        return kept

    def _set_or_remove_files(self, owner: dict[str, Any], processed_ids: set[str]) -> None:
        if "files" not in owner:
            return
        kept = self._filter_file_list(owner.get("files"), processed_ids)
        if kept:
            owner["files"] = kept
        else:
            owner.pop("files", None)

    def _remove_processed_request_files(
        self,
        body: dict,
        metadata: Optional[dict] = None,
        processed_file_ids: set[str] | None = None,
    ) -> None:
        processed_ids = {str(item) for item in (processed_file_ids or set()) if str(item)}
        if not processed_ids:
            return
        self._set_or_remove_files(body, processed_ids)
        metadata_items = [body.get("metadata")]
        if metadata is not None and metadata is not body.get("metadata"):
            metadata_items.append(metadata)
        for metadata_item in metadata_items:
            if not isinstance(metadata_item, dict):
                continue
            self._set_or_remove_files(metadata_item, processed_ids)
            user_message = metadata_item.get("user_message")
            if isinstance(user_message, dict):
                self._set_or_remove_files(user_message, processed_ids)
        for message in body.get("messages") or []:
            if isinstance(message, dict):
                self._set_or_remove_files(message, processed_ids)

    async def _openwebui_file_payloads(
        self,
        file_items: list[dict[str, Any]],
        user: Optional[dict],
        *,
        max_file_mb: int | None = None,
        error_label: str = "Compliance Check",
    ) -> list[dict[str, Any]]:
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage
        except Exception as exc:
            raise RuntimeError(f"OpenWebUI file storage is unavailable: {exc}") from exc

        user = user or {}
        user_id = str(user.get("id") or "")
        role = str(user.get("role") or "")
        payloads: list[dict[str, Any]] = []
        max_bytes = int(max_file_mb or self._compliance_max_file_mb()) * 1024 * 1024
        for item in file_items:
            file_id = str(item.get("id") or "")
            file_object = await Files.get_file_by_id(file_id)
            if not file_object:
                continue
            if role != "admin" and str(getattr(file_object, "user_id", "")) != user_id:
                continue
            meta = getattr(file_object, "meta", None) or {}
            if isinstance(meta, BaseModel):
                meta = self._model_values(meta)
            if not isinstance(meta, dict):
                meta = {}
            filename = (
                str(item.get("name") or "")
                or str(meta.get("name") or "")
                or str(getattr(file_object, "filename", "") or "")
                or file_id
            )
            content_type = (
                str(item.get("content_type") or "")
                or str(meta.get("content_type") or "")
                or "application/octet-stream"
            )
            content = b""
            path = str(getattr(file_object, "path", "") or "")
            if path:
                resolved = await asyncio.to_thread(Storage.get_file, path)
                resolved_path = Path(resolved)
                try:
                    size = resolved_path.stat().st_size
                except OSError:
                    size = 0
                if size > max_bytes:
                    raise RuntimeError(f"Файл слишком большой для {error_label}: {filename}")
                content = await asyncio.to_thread(resolved_path.read_bytes)
            if not content:
                data = getattr(file_object, "data", None) or {}
                if isinstance(data, dict) and data.get("content"):
                    content = str(data.get("content") or "").encode("utf-8")
                    if "." not in Path(filename).name:
                        filename = f"{filename}.txt"
            if content:
                if len(content) > max_bytes:
                    raise RuntimeError(f"Файл слишком большой для {error_label}: {filename}")
                payloads.append(
                    {
                        "filename": filename,
                        "content_type": content_type,
                        "content": content,
                        "external_id": file_id,
                        "metadata": {
                            **item,
                            "id": file_id,
                            "name": filename,
                            "content_type": content_type,
                        },
                    }
                )
        return payloads

    def _sanitize_index_id(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", str(value or "").strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
        return cleaned[:80]

    def _chat_attachment_scope(self, body: dict, metadata: Optional[dict], user: Optional[dict]) -> tuple[str, bool]:
        metadata = metadata or body.get("metadata") or {}
        chat_id = str(metadata.get("chat_id") or "").strip()
        if chat_id:
            return chat_id, True
        for key in ("session_id", "message_id", "user_message_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value, False
        return "", False

    def _chat_attachment_index_id(self, scope_id: str) -> str:
        raw_prefix = str(self._setting("chat_attachment_index_prefix", "owui_chat_") or "owui_chat_").strip()
        prefix = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", raw_prefix)
        prefix = re.sub(r"_+", "_", prefix)[:40] or "owui_chat_"
        scope = self._sanitize_index_id(scope_id)
        return self._sanitize_index_id(f"{prefix}{scope}")

    def _chat_attachment_index_ids(self, body: dict, metadata: Optional[dict], user: Optional[dict]) -> list[str]:
        if not self._bool_setting("chat_attachments_enabled", True):
            return []
        scope_id, _stable = self._chat_attachment_scope(body, metadata, user)
        if not scope_id:
            return []
        index_id = self._chat_attachment_index_id(scope_id)
        return [index_id] if index_id else []

    async def _index_chat_attachments(
        self,
        body: dict,
        metadata: Optional[dict],
        user: Optional[dict],
        file_items: list[dict[str, Any]],
        event_emitter=None,
    ) -> list[str]:
        extra_index_ids = self._chat_attachment_index_ids(body, metadata, user)
        if not self._bool_setting("chat_attachments_enabled", True) or not file_items:
            return extra_index_ids

        max_files = self._chat_attachment_max_files()
        if len(file_items) > max_files:
            if event_emitter:
                await event_emitter(
                    {
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": f"Слишком много вложений для индексации ZI_RAG: максимум {max_files}",
                        },
                    }
                )
            return extra_index_ids

        scope_id, stable_scope = self._chat_attachment_scope(body, metadata, user)
        if not scope_id:
            return extra_index_ids
        metadata = metadata or body.get("metadata") or {}
        if event_emitter:
            description = f"Enhanced RAG: индексирую вложения ({len(file_items)})"
            if not stable_scope:
                description += "; chat_id отсутствует, используется временный scope"
            await self._emit_status(event_emitter, description, done=False)
        try:
            upload_payloads = await self._openwebui_file_payloads(
                file_items,
                user,
                max_file_mb=self._chat_attachment_max_file_mb(),
                error_label="индексации вложений ZI_RAG",
            )
        except Exception as exc:
            if event_emitter:
                await event_emitter(
                    {
                        "type": "notification",
                        "data": {"type": "error", "content": str(exc)},
                    }
                )
            return extra_index_ids
        if not upload_payloads:
            return extra_index_ids

        payload = {
            "chat_id": str(metadata.get("chat_id") or ""),
            "session_id": str(metadata.get("session_id") or ""),
            "message_id": str(metadata.get("message_id") or metadata.get("user_message_id") or ""),
            "user_id": str((user or {}).get("id") or ""),
            "scope_id": scope_id,
            "files": [
                {
                    "id": str(item.get("external_id") or ""),
                    "name": str(item.get("filename") or "upload"),
                    "content_type": str(item.get("content_type") or ""),
                }
                for item in upload_payloads
            ],
        }
        try:
            response = await asyncio.to_thread(
                self._post_multipart,
                "/chat-attachments/index",
                payload,
                upload_payloads,
                self._chat_attachment_timeout_sec(),
            )
        except Exception as exc:
            if event_emitter:
                await event_emitter(
                    {
                        "type": "notification",
                        "data": {"type": "error", "content": str(exc)},
                    }
                )
            return extra_index_ids

        index_id = str(response.get("index_id") or "").strip()
        if index_id and index_id not in extra_index_ids:
            extra_index_ids.append(index_id)
        processed_ids = {
            str(item.get("external_id") or "")
            for item in upload_payloads
            if str(item.get("external_id") or "")
        }
        self._remove_processed_request_files(body, metadata, processed_ids)
        failed = response.get("failed") or []
        skipped = response.get("skipped") or []
        indexed = len(response.get("indexed_document_ids") or [])
        if event_emitter:
            await self._emit_status(
                event_emitter,
                (
                    "Enhanced RAG: вложения проиндексированы "
                    f"(new={indexed}, skipped={len(skipped)}, failed={len(failed)})"
                ),
                done=False,
            )
        return extra_index_ids

    def _serializable_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for message in messages[-8:]:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            result.append({"role": role, "content": self._message_text(message)})
        return result

    def _inject_deep_answer(self, user_message: dict[str, Any], answer: str) -> None:
        context = (
            "Multi-pass RAG уже выполнил анализ и подготовил итоговый ответ. "
            "Верни ответ ниже без изменений, не добавляя новые факты и источники.\n\n"
            f"{answer}"
        )
        self._inject_context(user_message, context)

    async def _inject_retrieval_context(
        self,
        *,
        query: str,
        user_message: dict[str, Any],
        index_ids: list[str],
        extra_index_ids: list[str],
        key: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        retrieval_top_k = top_k if top_k is not None else self._retrieval_top_k()
        payload = {
            "query": query,
            "index_ids": index_ids,
            "extra_index_ids": extra_index_ids,
            "top_k": retrieval_top_k,
            "score_threshold": self._min_relevance_score(),
        }
        response = await asyncio.to_thread(self._post_json, "/retrieve", payload)
        raw_docs = response.get("results") or []
        docs, filter_stats = self._filter_docs_for_prompt(query, raw_docs)
        if docs:
            knowledge, source_docs, stats = self._format_knowledge(docs)
            stats.update(filter_stats)
            self._store_sources(key, source_docs)
            source_policy = self._source_policy(source_docs)
            context_template = str(
                self._setting("context_template", self.valves.context_template)
                or self.valves.context_template
            )
            if "{knowledge}" not in context_template:
                context_template = f"{context_template.rstrip()}\n\n{{knowledge}}"
            context = context_template.format(
                knowledge=knowledge,
                source_policy=source_policy,
            )
            if source_policy:
                context = f"{source_policy}\n\n{context}"
            self._inject_context(user_message, context)
            return stats

        self._store_sources(key, [])
        return {
            "retrieved": len(raw_docs),
            "included": 0,
            "batches": 0,
            "compact": 0,
            "overflow": 0,
            **filter_stats,
        }

    def _replace_last_assistant(self, body: dict, answer: str) -> None:
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") == "assistant":
                message["content"] = answer
                return
        messages.append({"role": "assistant", "content": answer})
        body["messages"] = messages

    async def _emit_status(self, event_emitter, description: str, *, done: bool = False) -> None:
        if not event_emitter:
            return
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": self._trim_text(description, 900),
                    "done": done,
                },
            }
        )

    def _deep_progress_description(self, event: dict[str, Any]) -> str:
        stage = str(event.get("stage") or "")
        message = str(event.get("message") or "").strip()
        batch = event.get("batch")
        total = event.get("total_batches")
        note = str(event.get("note_excerpt") or "").strip()
        if stage == "batch_start" and batch and total:
            return f"Enhanced RAG: анализ пачки {batch}/{total}"
        if stage == "batch_done" and batch and total:
            detail = note or message
            return f"Enhanced RAG: пачка {batch}/{total} готова - {detail}"
        if stage == "synthesis":
            return "Enhanced RAG: финальный синтез"
        if stage == "done":
            return f"Enhanced RAG: {message or 'multi-pass готов'}"
        return f"Enhanced RAG: {message or stage or 'multi-pass анализ'}"

    async def _run_deep_analysis(
        self,
        payload: dict[str, Any],
        timeout_sec: int,
        event_emitter=None,
    ) -> dict[str, Any]:
        if not event_emitter:
            return await asyncio.to_thread(self._post_json, "/analyze", payload, timeout_sec)

        try:
            job = await asyncio.to_thread(
                self._post_json,
                "/analyze/jobs",
                payload,
                min(timeout_sec, 30),
            )
        except Exception:
            return await asyncio.to_thread(self._post_json, "/analyze", payload, timeout_sec)

        job_id = str(job.get("id") or "")
        if not job_id:
            return await asyncio.to_thread(self._post_json, "/analyze", payload, timeout_sec)

        async def cancel_job() -> None:
            try:
                await asyncio.shield(
                    asyncio.to_thread(self._post_json, f"/analyze/jobs/{job_id}/cancel", {}, 5)
                )
            except Exception:
                pass

        await self._emit_status(event_emitter, "Enhanced RAG: multi-pass анализ запущен", done=False)
        seen_events = 0
        deadline = time.monotonic() + max(1, timeout_sec)

        async def consume_sse_stream() -> dict[str, Any]:
            nonlocal seen_events
            iterator = self._iter_sse_events(f"/analyze/jobs/{job_id}/events", max(2, timeout_sec))
            while True:
                item = await asyncio.to_thread(self._next_sse_event, iterator)
                if item is None:
                    raise RuntimeError("Multi-pass analysis stream ended before completion")
                event_name, stream_job = item
                if event_name == "error":
                    raise RuntimeError(str(stream_job.get("error") or "Multi-pass analysis stream failed"))
                events = stream_job.get("events") or []
                for event in events[seen_events:]:
                    await self._emit_status(
                        event_emitter,
                        self._deep_progress_description(event),
                        done=False,
                    )
                seen_events = len(events)

                status = str(stream_job.get("status") or "")
                if status == "completed":
                    return dict(stream_job.get("result") or {})
                if status in {"failed", "canceled"}:
                    raise RuntimeError(
                        str(stream_job.get("error") or stream_job.get("message") or "Multi-pass analysis failed")
                    )
                if time.monotonic() >= deadline:
                    await cancel_job()
                    raise RuntimeError("Multi-pass analysis timed out")

        try:
            try:
                return await consume_sse_stream()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            while True:
                events = job.get("events") or []
                for event in events[seen_events:]:
                    await self._emit_status(
                        event_emitter,
                        self._deep_progress_description(event),
                        done=False,
                    )
                seen_events = len(events)

                status = str(job.get("status") or "")
                if status == "completed":
                    return dict(job.get("result") or {})
                if status in {"failed", "canceled"}:
                    raise RuntimeError(str(job.get("error") or job.get("message") or "Multi-pass analysis failed"))
                if time.monotonic() >= deadline:
                    await cancel_job()
                    raise RuntimeError("Multi-pass analysis timed out")

                await asyncio.sleep(1.0)
                poll_timeout = max(2, min(15, int(deadline - time.monotonic()) or 2))
                job = await asyncio.to_thread(self._get_json, f"/analyze/jobs/{job_id}", poll_timeout)
        except asyncio.CancelledError:
            await cancel_job()
            raise

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        __user_valves__: Optional[UserValves] = None,
        __request__=None,
    ) -> dict:
        self._cleanup_request_state()
        user_valves = self._user_valves(__user__, __user_valves__)
        self._sidecar_admin_config()
        await self._maybe_emit_admin_config_warning(__event_emitter__)
        if not getattr(user_valves, "rag_enabled", True):
            return body
        if not self._bool_setting("rag_enabled", True):
            return body

        messages = body.get("messages") or []
        last_user = self._last_user_message(messages)
        query = self._message_text(last_user or {}).strip()
        if not query:
            return body

        check_marker, clean_check_query = self._check_marker(query)
        if check_marker:
            query = clean_check_query
            self._set_message_text(last_user, query)
        check_index_ids, cleaned_query = self._extract_check_index_override(query)
        if check_index_ids and cleaned_query != query:
            query = cleaned_query
            self._set_message_text(last_user, query)

        file_items = self._request_file_items(body, __metadata__)
        extra_index_ids = await self._index_chat_attachments(
            body,
            __metadata__,
            __user__,
            file_items,
            __event_emitter__,
        )
        compliance_enabled = (
            bool(getattr(user_valves, "compliance_enabled", True))
            and self._bool_setting("compliance_enabled", True)
        )
        compliance_auto = self._bool_setting("compliance_auto_enabled", True)
        compliance_triggered = check_marker or (
            compliance_enabled
            and compliance_auto
            and bool(file_items)
            and self._is_compliance_query(query)
        )
        if compliance_triggered:
            if not compliance_enabled:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": "Compliance Check выключен"},
                        }
                    )
                return body
            if not file_items:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": "Для /check прикрепите файл к сообщению"},
                        }
                    )
                return body
            max_files = self._compliance_max_files()
            if len(file_items) > max_files:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "error",
                                "content": f"Слишком много вложений для Compliance Check: максимум {max_files}",
                            },
                        }
                    )
                return body
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Compliance Check: чтение {len(file_items)} вложений",
                            "done": False,
                        },
                    }
                )
            try:
                upload_payloads = await self._openwebui_file_payloads(file_items, __user__)
            except Exception as exc:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": str(exc)},
                        }
                    )
                return body
            if not upload_payloads:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": "Не удалось прочитать вложенные файлы"},
                        }
                    )
                return body

            allow_user_indexes = self._bool_setting("compliance_allow_user_index_override", True)
            compliance_index_ids: list[str] = []
            if allow_user_indexes:
                compliance_index_ids = check_index_ids or self._parse_ids(
                    getattr(user_valves, "compliance_index_ids", "")
                )
            payload = {
                "query": query,
                "messages": self._serializable_messages(messages),
                "nmd_index_ids": compliance_index_ids,
                "generation_model": str(body.get("model") or ""),
                "top_k": self._compliance_requirement_top_k(),
                "score_threshold": self._min_relevance_score(),
                "section_chars": self._compliance_section_chars(),
                "max_sections": self._compliance_max_sections(),
            }
            try:
                response = await asyncio.to_thread(
                    self._post_multipart,
                    "/compliance/analyze",
                    payload,
                    upload_payloads,
                    self._compliance_timeout_sec(),
                )
            except Exception as exc:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": str(exc)},
                        }
                    )
                return body
            answer = str(response.get("answer") or "").strip()
            if answer:
                key = self._request_key(body, __metadata__, __user__)
                self._store_deep_answer(key, answer)
                self._inject_deep_answer(last_user, answer)
                self._remove_processed_request_files(
                    body,
                    __metadata__,
                    {str(item.get("id") or "") for item in file_items},
                )
            stats = response.get("stats") or {}
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                "Compliance Check: готов "
                                f"({stats.get('sections', 0)} секций, "
                                f"{stats.get('matrix_rows', 0)} строк матрицы)"
                            ),
                            "done": True,
                        },
                    }
                )
            return body

        deep_marker, clean_query = self._deep_marker(query)
        if deep_marker:
            query = clean_query
            self._set_message_text(last_user, query)
        deep_auto = (
            bool(getattr(user_valves, "deep_analysis_enabled", False))
            or self._bool_setting("deep_analysis_enabled", False)
        )
        deep_force = self._bool_setting("deep_force_all", False)
        deep_enabled = deep_marker or deep_force or (deep_auto and self._is_complex_query(query))

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            "Enhanced RAG: multi-pass анализ"
                            if deep_enabled
                            else "Enhanced RAG: поиск контекста"
                        ),
                        "done": False,
                    },
                }
        )

        index_ids = self._parse_ids(user_valves.index_ids)
        if not index_ids:
            default_ids = self._setting("default_index_ids", [])
            if isinstance(default_ids, list):
                index_ids = [str(item).strip() for item in default_ids if str(item).strip()]
        key = self._request_key(body, __metadata__, __user__)

        if deep_enabled:
            mode = "answer" if (
                bool(getattr(user_valves, "deep_final_answer", True))
                and self._bool_setting("deep_final_answer", True)
            ) else "context"
            payload = {
                "query": query,
                "messages": self._serializable_messages(messages),
                "index_ids": index_ids,
                "extra_index_ids": extra_index_ids,
                "generation_model": str(body.get("model") or ""),
                "mode": mode,
                "top_k": self._deep_top_k(),
                "score_threshold": self._min_relevance_score(),
                "batch_chars": self._deep_batch_chars(),
                "max_batches": self._deep_max_batches(),
            }
            try:
                response = await self._run_deep_analysis(
                    payload,
                    self._deep_timeout_sec(),
                    __event_emitter__,
                )
            except Exception as exc:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": str(exc)},
                        }
                    )
                try:
                    stats = await self._inject_retrieval_context(
                        query=query,
                        user_message=last_user,
                        index_ids=index_ids,
                        extra_index_ids=extra_index_ids,
                        key=key,
                        top_k=self._deep_top_k(),
                    )
                except Exception as fallback_exc:
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {
                                    "type": "error",
                                    "content": (
                                        "Deep RAG не сработал, fallback retrieval тоже не удался: "
                                        f"{fallback_exc}"
                                    ),
                                },
                            }
                        )
                    return body
                if __event_emitter__:
                    await self._emit_status(
                        __event_emitter__,
                        (
                            "Enhanced RAG: multi-pass недоступен, вставлен обычный RAG-контекст "
                            f"({stats.get('included', 0)} фрагментов)"
                        ),
                        done=True,
                    )
                return body

            stats = response.get("stats") or {}
            if mode == "answer":
                answer = str(response.get("answer") or "").strip()
                if answer:
                    self._store_deep_answer(key, answer)
                    self._inject_deep_answer(last_user, answer)
            else:
                context = str(response.get("context") or "").strip()
                if context:
                    self._inject_context(last_user, context)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                "Enhanced RAG: multi-pass готов "
                                f"({stats.get('batches', 0)} пачек, "
                                f"{stats.get('filtered', 0)} фрагментов)"
                            ),
                            "done": True,
                        },
                    }
                )
            return body

        retrieval_top_k = self._retrieval_top_k()
        try:
            stats = await self._inject_retrieval_context(
                query=query,
                user_message=last_user,
                index_ids=index_ids,
                extra_index_ids=extra_index_ids,
                key=key,
                top_k=retrieval_top_k,
            )
        except Exception as exc:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "notification",
                        "data": {"type": "error", "content": str(exc)},
                    }
                )
            return body

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            "Enhanced RAG: найдено "
                            f"{stats.get('raw', stats['retrieved'])}, после фильтра "
                            f"{stats.get('filtered', stats['retrieved'])}, включено {stats['included']} "
                            f"в {stats['batches']} пачках, компактно {stats['compact']}"
                        ),
                        "done": True,
                    },
                }
            )
        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        self._cleanup_request_state()
        key = self._request_key(body, __metadata__, __user__)
        self._sources_by_key.pop(key, None)
        answer = self._pop_deep_answer(key)
        if answer:
            self._replace_last_assistant(body, answer)
        return body
