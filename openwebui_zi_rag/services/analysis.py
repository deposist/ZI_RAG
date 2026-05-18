from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

try:
    from fastapi import HTTPException
except Exception:  # pragma: no cover - mirrors server import tolerance
    class HTTPException(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

from ..config import SidecarConfig
from ..indexing.extraction import extract_text, is_supported_file
from ..indexing.service import RagService
from ..ollama_client import OllamaClient, make_generation_client
from ..text_utils import compact_dialog_context, filter_analysis_docs, trim_text
from .prompting import (
    analysis_context,
    batch_prompt,
    checked_source_details,
    clamp_int,
    compliance_final_prompt,
    compliance_section_prompt,
    final_prompt,
    matrix_markdown,
    pack_analysis_batches,
    parse_matrix_rows,
    safe_upload_name,
    source_details,
    split_checked_sections,
)


class AnalysisCancelled(RuntimeError):
    pass


def make_ollama_client(
    cfg: SidecarConfig,
    *,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    stream_idle_timeout: float | None = None,
) -> Any:
    return make_generation_client(
        cfg,
        request_timeout=request_timeout,
        connect_timeout=connect_timeout,
        stream_idle_timeout=stream_idle_timeout,
        ollama_client_cls=OllamaClient,
    )


def available_model_names(client: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in client.list_models():
        name = item.get("name") or item.get("model")
        if not name:
            continue
        value = str(name)
        if value not in seen:
            names.append(value)
            seen.add(value)
    return names


def generation_model_aliases(candidate: str) -> list[str]:
    aliases = [candidate]
    if "/" in candidate:
        aliases.append(candidate.rsplit("/", 1)[-1])
    if candidate.startswith("ollama:"):
        aliases.append(candidate.split(":", 1)[1])
    return aliases


def resolve_generation_model(candidates: list[str], client: Any) -> str:
    try:
        available = available_model_names(client)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load Ollama models: {exc}") from exc
    available_set = set(available)
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        for alias in generation_model_aliases(candidate):
            if alias in available:
                return alias
            if alias in available_set:
                return alias
    available_text = ", ".join(available) if available else "no models available"
    raise HTTPException(
        status_code=409,
        detail=f"Generation model is not configured. Select one of: {available_text}",
    )


def progress_event(
    progress: Callable[[dict[str, Any]], None] | None,
    stage: str,
    message: str,
    **extra: Any,
) -> None:
    if progress is None:
        return
    event = {
        "stage": stage,
        "message": message,
        "ts": round(time.time(), 3),
    }
    event.update({key: value for key, value in extra.items() if value is not None})
    progress(event)


def progress_note_excerpt(note: str, max_chars: int = 700) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", " ", str(note or ""))
    text = re.sub(r"(?is)<think>.*$", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return trim_text(text, max_chars)


def raise_if_analysis_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled("Multi-pass analysis canceled")


def run_multi_pass_analysis(
    payload: Any,
    *,
    cfg: SidecarConfig,
    service: RagService,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        return {
            "answer": "",
            "context": "",
            "sources": [],
            "batch_notes": [],
            "stats": {"raw": 0, "filtered": 0, "batches": 0, "dropped": 0},
            "model": "",
        }

    progress_events: list[dict[str, Any]] = []
    external_progress = progress

    def capture_progress(event: dict[str, Any]) -> None:
        progress_events.append(dict(event))
        if external_progress is not None:
            external_progress(event)

    progress = capture_progress

    mode = (payload.mode or "answer").strip().lower()
    if mode not in {"answer", "context"}:
        mode = "answer"

    top_k = clamp_int(payload.top_k, cfg.deep_top_k, 1, 500)
    score_threshold = float(
        payload.score_threshold
        if payload.score_threshold is not None
        else cfg.score_threshold
    )
    batch_chars = clamp_int(payload.batch_chars, cfg.deep_batch_chars, 1500, 64000)
    max_batches = clamp_int(payload.max_batches, cfg.deep_max_batches, 1, 50)
    batch_tokens = clamp_int(cfg.deep_batch_max_tokens, 1024, 128, 4096)
    final_tokens = clamp_int(cfg.deep_final_max_tokens, 2048, 256, 8192)

    raise_if_analysis_cancelled(cancel_check)
    progress_event(
        progress,
        "retrieval",
        f"Поиск релевантных фрагментов: top_k={top_k}, threshold={score_threshold:.2f}",
        done=False,
    )
    retrieval = service.retrieve(
        query,
        index_ids=payload.index_ids or None,
        extra_index_ids=payload.extra_index_ids or None,
        embedding_model=payload.embedding_model or None,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    raise_if_analysis_cancelled(cancel_check)
    raw_docs = retrieval.get("results") or []
    docs, filter_stats = filter_analysis_docs(
        query,
        raw_docs,
        min_score=score_threshold,
        margin=float(getattr(cfg, "adaptive_score_margin", 0.20) or 0.20),
    )
    batches, dropped = pack_analysis_batches(
        docs,
        batch_chars=batch_chars,
        max_batches=max_batches,
    )
    stats = {
        **filter_stats,
        "batches": len(batches),
        "dropped": dropped,
        "top_k": top_k,
        "score_threshold": score_threshold,
    }
    progress_event(
        progress,
        "packing",
        (
            f"Найдено {stats.get('raw', 0)}, после фильтра {stats.get('filtered', 0)}, "
            f"упаковано в {stats.get('batches', 0)} пачек, пропущено {stats.get('dropped', 0)}"
        ),
        stats=stats,
        done=False,
    )

    raise_if_analysis_cancelled(cancel_check)
    if not batches:
        context = analysis_context(query, [], "", stats)
        answer = (
            "В базе знаний не найдено релевантных фрагментов для multi-pass анализа."
            if mode == "answer"
            else ""
        )
        progress_event(progress, "done", "Релевантные фрагменты для multi-pass анализа не найдены", done=True)
        return {
            "answer": answer,
            "context": context,
            "sources": [],
            "batch_notes": [],
            "stats": stats,
            "model": "",
            "progress": progress_events,
        }

    client = make_ollama_client(cfg, request_timeout=cfg.deep_timeout_sec)
    model = resolve_generation_model(
        [
            payload.generation_model,
            cfg.deep_generation_model,
        ],
        client,
    )
    raise_if_analysis_cancelled(cancel_check)
    progress_event(
        progress,
        "model",
        f"Модель анализа: {model}",
        model=model,
        done=False,
    )
    dialog_context = compact_dialog_context(payload.messages)
    system_batch = (
        "Ты аккуратно извлекаешь факты из документов для RAG. "
        "Используй только предоставленный текст, сохраняй документы, локаторы и короткие цитаты."
    )
    batch_notes: list[str] = []
    for batch_no, docs_batch in enumerate(batches, start=1):
        raise_if_analysis_cancelled(cancel_check)
        progress_event(
            progress,
            "batch_start",
            f"Пачка {batch_no}/{len(batches)}: извлечение фактов из {len(docs_batch)} фрагментов",
            batch=batch_no,
            total_batches=len(batches),
            docs=len(docs_batch),
            done=False,
        )
        note = client.chat(
            model,
            [
                {"role": "system", "content": system_batch},
                {
                    "role": "user",
                    "content": batch_prompt(
                        query,
                        docs_batch,
                        batch_no=batch_no,
                        total_batches=len(batches),
                        dialog_context=dialog_context,
                    ),
                },
            ],
            temperature=0.1,
            num_predict=batch_tokens,
            cancel_check=cancel_check,
        )
        raise_if_analysis_cancelled(cancel_check)
        if note:
            batch_notes.append(note)
        progress_event(
            progress,
            "batch_done",
            (
                f"Пачка {batch_no}/{len(batches)} готова: "
                f"{progress_note_excerpt(note) or 'существенные факты не извлечены'}"
            ),
            batch=batch_no,
            total_batches=len(batches),
            note_excerpt=progress_note_excerpt(note),
            done=False,
        )

    source_block = source_details(docs)
    context = analysis_context(query, batch_notes, source_block, stats)
    answer = ""
    if mode == "answer":
        raise_if_analysis_cancelled(cancel_check)
        progress_event(
            progress,
            "synthesis",
            f"Финальный синтез по {len(batch_notes)} промежуточным извлечениям",
            done=False,
        )
        answer = client.chat(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Ты собираешь финальный ответ по multi-pass RAG-извлечениям. "
                        "Не добавляй внешние знания и обязательно сохраняй источники."
                    ),
                },
                {
                    "role": "user",
                    "content": final_prompt(
                        query,
                        batch_notes,
                        dialog_context=dialog_context,
                        sources=source_block,
                    ),
                },
            ],
            temperature=0.1,
            num_predict=final_tokens,
            cancel_check=cancel_check,
        )
        raise_if_analysis_cancelled(cancel_check)
        if "<details>" not in answer.lower():
            answer = f"{answer.rstrip()}\n\n{source_block}"
        progress_event(progress, "answer_done", "Финальный ответ multi-pass готов", done=False)
    else:
        progress_event(progress, "context_done", "Deep-контекст подготовлен для модели OpenWebUI", done=False)

    progress_event(
        progress,
        "done",
        (
            f"Multi-pass завершён: {stats.get('batches', 0)} пачек, "
            f"{len(batch_notes)} извлечений"
        ),
        stats=stats,
        done=True,
    )
    return {
        "answer": answer,
        "context": context,
        "sources": docs,
        "batch_notes": batch_notes,
        "stats": stats,
        "model": model,
        "progress": progress_events,
    }


def compliance_timeout_error() -> HTTPException:
    return HTTPException(status_code=504, detail="Compliance analysis timed out")


def compliance_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise compliance_timeout_error()
    return remaining


def set_compliance_client_timeout(client: Any, cfg: SidecarConfig, deadline: float) -> None:
    remaining = max(1.0, min(float(cfg.request_timeout_sec or 120), compliance_remaining(deadline)))
    for attr in ("timeout", "request_timeout", "stream_idle_timeout"):
        if hasattr(client, attr):
            setattr(client, attr, remaining)


def run_compliance_analysis(
    payload: Any,
    file_payloads: list[dict[str, Any]],
    *,
    cfg: SidecarConfig,
    service: RagService,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + max(1, int(cfg.compliance_timeout_sec or 1200))
    query = payload.query.strip()
    if not query:
        query = "Проверить приложенные документы на соответствие НМД"

    index_ids = payload.nmd_index_ids or cfg.compliance_index_ids
    top_k = clamp_int(payload.top_k, cfg.compliance_requirement_top_k, 1, 100)
    score_threshold = float(
        payload.score_threshold
        if payload.score_threshold is not None
        else cfg.score_threshold
    )
    section_chars = clamp_int(payload.section_chars, cfg.compliance_section_chars, 1500, 64000)
    max_sections = clamp_int(payload.max_sections, cfg.compliance_max_sections, 1, 500)

    checked_files: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    dropped_sections = 0
    with tempfile.TemporaryDirectory(prefix="zi-rag-compliance-") as tmp:
        tmp_root = Path(tmp)
        for file_index, item in enumerate(file_payloads, start=1):
            compliance_remaining(deadline)
            filename = safe_upload_name(str(item.get("filename") or f"file-{file_index}"))
            target = tmp_root / filename
            if target.exists():
                target = tmp_root / f"{target.stem}_{file_index}{target.suffix}"
            target.write_bytes(bytes(item.get("content") or b""))
            file_info = {
                "filename": filename,
                "content_type": item.get("content_type") or "",
                "size": target.stat().st_size,
                "text_chars": 0,
                "sections": 0,
                "error": "",
            }
            try:
                if not is_supported_file(target):
                    raise ValueError(f"Unsupported file extension: {target.suffix}")
                compliance_remaining(deadline)
                text = extract_text(target, cfg)
                compliance_remaining(deadline)
                if not text.strip():
                    raise ValueError("Document text is empty after extraction")
                file_info["text_chars"] = len(text)
                remaining_before = max_sections - len(sections)
                file_sections, dropped = split_checked_sections(
                    filename,
                    text,
                    section_chars=section_chars,
                    remaining=remaining_before,
                )
                file_info["sections"] = len(file_sections)
                sections.extend(file_sections)
                dropped_sections += dropped
                if len(file_sections) >= remaining_before:
                    dropped_sections += max(0, len(file_payloads) - file_index)
            except HTTPException:
                raise
            except Exception as exc:
                file_info["error"] = str(exc)
            checked_files.append(file_info)
            if len(sections) >= max_sections:
                break

    if not sections:
        answer = (
            "Не удалось извлечь текст из приложенных файлов для проверки соответствия НМД.\n\n"
            + checked_source_details(checked_files, [])
        )
        return {
            "answer": answer,
            "matrix": [],
            "checked_files": checked_files,
            "sources": [],
            "stats": {
                "files": len(file_payloads),
                "sections": 0,
                "dropped_sections": dropped_sections,
                "requirement_hits": 0,
                "llm_passes": 0,
            },
            "model": "",
        }

    client = make_ollama_client(cfg, request_timeout=cfg.compliance_timeout_sec)
    set_compliance_client_timeout(client, cfg, deadline)
    model = resolve_generation_model(
        [
            payload.generation_model,
            cfg.compliance_generation_model,
            cfg.deep_generation_model,
        ],
        client,
    )
    compliance_remaining(deadline)
    dialog_context = compact_dialog_context(payload.messages)
    batch_tokens = clamp_int(cfg.deep_batch_max_tokens, 1024, 128, 4096)
    final_tokens = clamp_int(cfg.deep_final_max_tokens, 2048, 256, 8192)

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    nmd_docs: list[dict[str, Any]] = []
    system = (
        "Ты эксперт по проверке документов на соответствие НМД. "
        "Работай только по предоставленным фрагментам и сохраняй цитаты."
    )

    for section in sections:
        compliance_remaining(deadline)
        section_query = (
            f"{query}\n\n"
            f"Проверяемый файл: {section['file']}; локатор: {section['locator']}\n"
            f"{trim_text(section['text'], 2400)}"
        )
        compliance_remaining(deadline)
        retrieval = service.retrieve(
            section_query,
            index_ids=index_ids or None,
            embedding_model=payload.embedding_model or None,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        compliance_remaining(deadline)
        raw_requirements = retrieval.get("results") or []
        requirements, _ = filter_analysis_docs(
            section_query,
                raw_requirements,
                min_score=score_threshold,
                margin=float(getattr(cfg, "adaptive_score_margin", 0.20) or 0.20),
            )
        requirements = requirements[:top_k]
        nmd_docs.extend(requirements)
        if not requirements:
            rows.append(
                {
                    "requirement": "",
                    "nmd_source": "",
                    "nmd_locator": "",
                    "nmd_quote": "",
                    "checked_file": section["file"],
                    "checked_locator": section["locator"],
                    "checked_quote": section["quote"],
                    "status": "требует ручной проверки",
                    "risk": "Релевантные требования НМД не найдены по этой секции.",
                    "recommendation": "Проверьте выбор индекса НМД и формулировку запроса.",
                }
            )
            continue
        set_compliance_client_timeout(client, cfg, deadline)
        note = client.chat(
            model,
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": compliance_section_prompt(
                        query,
                        section,
                        requirements,
                        dialog_context=dialog_context,
                    ),
                },
            ],
            temperature=0.1,
            num_predict=batch_tokens,
        )
        compliance_remaining(deadline)
        notes.append(note)
        rows.extend(parse_matrix_rows(note, section))

    compliance_remaining(deadline)
    nmd_source_block = source_details(nmd_docs, limit=80)
    checked_source_block = checked_source_details(checked_files, sections)
    set_compliance_client_timeout(client, cfg, deadline)
    answer = client.chat(
        model,
        [
            {
                "role": "system",
                "content": (
                    "Ты формируешь итоговый акт проверки соответствия НМД. "
                    "Ответ должен быть точным, структурированным и пригодным для аудита."
                ),
            },
            {
                "role": "user",
                "content": compliance_final_prompt(
                    query,
                    rows,
                    notes,
                    checked_sources=checked_source_block,
                    nmd_sources=nmd_source_block,
                    dialog_context=dialog_context,
                ),
            },
        ],
        temperature=0.1,
        num_predict=final_tokens,
    )
    compliance_remaining(deadline)
    if "<details>" not in answer.lower():
        answer = (
            f"{answer.rstrip()}\n\n"
            "## Матрица соответствия\n\n"
            f"{matrix_markdown(rows)}\n\n"
            "<details>\n<summary>Источники</summary>\n\n"
            f"{checked_source_block}\n\n{nmd_source_block}\n"
            "</details>"
        )

    unique_nmd: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for doc in nmd_docs:
        key = f"{doc.get('source')}|{doc.get('locator')}|{doc.get('quote')}"
        if key in seen_sources:
            continue
        seen_sources.add(key)
        unique_nmd.append(doc)

    return {
        "answer": answer,
        "matrix": rows,
        "checked_files": checked_files,
        "sources": [
            {"type": "checked", "items": sections[:120]},
            {"type": "nmd", "items": unique_nmd[:120]},
        ],
        "stats": {
            "files": len(file_payloads),
            "checked_files": len(checked_files),
            "sections": len(sections),
            "dropped_sections": dropped_sections,
            "requirement_hits": len(nmd_docs),
            "matrix_rows": len(rows),
            "llm_passes": len(notes) + 1,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "nmd_index_ids": index_ids,
        },
        "model": model,
    }


__all__ = [
    "AnalysisCancelled",
    "OllamaClient",
    "extract_text",
    "is_supported_file",
    "filter_analysis_docs",
    "make_ollama_client",
    "available_model_names",
    "generation_model_aliases",
    "resolve_generation_model",
    "progress_event",
    "progress_note_excerpt",
    "raise_if_analysis_cancelled",
    "compliance_remaining",
    "compliance_timeout_error",
    "set_compliance_client_timeout",
    "run_multi_pass_analysis",
    "run_compliance_analysis",
]
