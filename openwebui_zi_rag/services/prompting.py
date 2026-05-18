from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..text_utils import clean_quote_text, trim_text


def _clean_quote(text: str, max_chars: int) -> str:
    return trim_text(clean_quote_text(text), max_chars)


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def format_source_line(index: int, doc: dict[str, Any], *, quote_chars: int = 260) -> str:
    source = str(doc.get("source") or doc.get("source_path") or "unknown")
    locator = str(doc.get("locator") or f"chunk {doc.get('chunk_no', '?')}")
    quote = _clean_quote(doc.get("quote") or doc.get("text") or "", quote_chars)
    score = float(doc.get("score") or 0.0)
    return (
        f"{index}. **{source}** ({locator}, score={score:.3f})\n"
        f"   > {quote}"
    )


def source_details(docs: list[dict[str, Any]], *, limit: int = 40) -> str:
    lines = ["<details>", "<summary>Источники</summary>", ""]
    for doc in docs[:limit]:
        lines.append(format_source_line(len(lines) + 1, doc))
        lines.append("")
    if len(docs) > limit:
        lines.append(f"... ещё {len(docs) - limit} источников скрыто лимитом.")
    lines.append("</details>")
    return "\n".join(lines)


def format_batch_doc(index: int, doc: dict[str, Any], *, max_text_chars: int) -> str:
    source = str(doc.get("source") or doc.get("source_path") or "unknown")
    locator = str(doc.get("locator") or f"chunk {doc.get('chunk_no', '?')}")
    quote = _clean_quote(doc.get("quote") or "", 500)
    text = trim_text(doc.get("text") or quote, max_text_chars)
    return (
        f"[{index}] Источник: {source}\n"
        f"Локатор: {locator}\n"
        f"Score: {float(doc.get('score') or 0.0):.3f}\n"
        f"Цитата: {quote}\n"
        f"Текст:\n{text}"
    )


def pack_analysis_batches(
    docs: list[dict[str, Any]],
    *,
    batch_chars: int,
    max_batches: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for doc in docs:
        block = format_batch_doc(1, doc, max_text_chars=batch_chars)
        block_chars = len(block)
        if current and current_chars + block_chars > batch_chars:
            batches.append(current)
            current = []
            current_chars = 0
            if len(batches) >= max_batches:
                break
        current.append(doc)
        current_chars += block_chars
    if current and len(batches) < max_batches:
        batches.append(current)
    packed_count = sum(len(batch) for batch in batches)
    return batches, max(0, len(docs) - packed_count)


def batch_prompt(question: str, docs: list[dict[str, Any]], *, batch_no: int, total_batches: int, dialog_context: str) -> str:
    docs_text = "\n\n".join(
        format_batch_doc(index, doc, max_text_chars=3500)
        for index, doc in enumerate(docs, start=1)
    )
    dialog_block = f"Контекст последних сообщений:\n{dialog_context}\n\n" if dialog_context else ""
    return (
        f"{dialog_block}"
        f"Вопрос пользователя: {question}\n\n"
        f"Ниже дана пачка документов {batch_no}/{total_batches}. "
        "Извлеки только подтверждённые факты, требования, числа, условия, ограничения и противоречия, "
        "которые помогают ответить на вопрос. Не добавляй внешние знания и догадки. "
        "Для каждого существенного факта указывай документ, локатор и короткую цитату. "
        "Если прямого ответа в пачке нет, кратко напиши, что прямой ответ не найден.\n\n"
        "Формат:\n"
        "1. Краткий вывод по пачке.\n"
        "2. Факты/требования с источниками.\n"
        "3. Неясности или противоречия.\n\n"
        f"Документы пачки:\n{docs_text}"
    )


def final_prompt(question: str, notes: list[str], *, dialog_context: str, sources: str) -> str:
    notes_text = "\n\n".join(
        f"[Пачка {index}]\n{note.strip()}"
        for index, note in enumerate(notes, start=1)
        if note.strip()
    )
    dialog_block = f"Контекст последних сообщений:\n{dialog_context}\n\n" if dialog_context else ""
    return (
        f"{dialog_block}"
        f"Вопрос пользователя: {question}\n\n"
        "Ниже даны промежуточные извлечения из пачек документов. "
        "Собери единый полный ответ только по этим извлечениям. Не добавляй внешние знания. "
        "Не теряй факты из поздних пачек, убирай повторы, противоречия отмечай явно. "
        "Каждый фактический вывод подкрепляй документом, локатором и короткой цитатой. "
        "В конце обязательно добавь HTML-спойлер источников ровно в формате <details><summary>Источники</summary>.\n\n"
        f"Промежуточные извлечения:\n{notes_text or 'Извлечения отсутствуют.'}\n\n"
        f"Разрешённые источники для спойлера:\n{sources}"
    )


def analysis_context(question: str, notes: list[str], sources: str, stats: dict[str, Any]) -> str:
    notes_text = "\n\n".join(
        f"[Пачка {index}]\n{note.strip()}"
        for index, note in enumerate(notes, start=1)
        if note.strip()
    )
    return (
        "Deep RAG multi-pass context.\n"
        f"Вопрос: {question}\n"
        f"Статистика: найдено={stats.get('raw', 0)}, после фильтра={stats.get('filtered', 0)}, "
        f"пачек={stats.get('batches', 0)}, пропущено={stats.get('dropped', 0)}.\n\n"
        "Используй только эти промежуточные извлечения. Не добавляй внешние знания. "
        "В конце ответа добавь источники в HTML-спойлере.\n\n"
        f"{notes_text or 'Релевантные извлечения отсутствуют.'}\n\n"
        f"{sources}"
    )


def safe_upload_name(filename: str) -> str:
    name = Path(filename or "upload").name.replace("\x00", "").strip()
    name = re.sub(r"[\r\n\t]", "_", name)
    if name in {"", ".", ".."}:
        name = "upload"
    return name[:180] or "upload"


def locators_from_text(text: str, *, limit: int = 3) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\[([^\]]{1,160})\]", text or ""):
        value = re.sub(r"\s+", " ", match.group(1)).strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def split_checked_sections(
    filename: str,
    text: str,
    *,
    section_chars: int,
    remaining: int,
) -> tuple[list[dict[str, Any]], int]:
    cleaned = re.sub(r"\r\n?", "\n", text or "").strip()
    if not cleaned or remaining <= 0:
        return [], 0
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned]
    sections: list[dict[str, Any]] = []
    current: list[str] = []
    current_chars = 0
    paragraph_no = 0
    dropped = 0

    def flush() -> bool:
        nonlocal current, current_chars, paragraph_no, dropped
        if not current:
            return True
        if len(sections) >= remaining:
            dropped += 1
            current = []
            current_chars = 0
            return False
        paragraph_no += 1
        section_text = "\n\n".join(current).strip()
        locators = locators_from_text(section_text)
        locator = "; ".join(locators) or f"секция {paragraph_no}"
        sections.append(
            {
                "file": filename,
                "locator": locator,
                "text": section_text,
                "quote": _clean_quote(section_text, 360),
            }
        )
        current = []
        current_chars = 0
        return True

    for paragraph in paragraphs:
        if current and current_chars + len(paragraph) + 2 > section_chars:
            if not flush():
                dropped += 1
                continue
        if len(paragraph) > section_chars:
            for start in range(0, len(paragraph), section_chars):
                if current and not flush():
                    dropped += 1
                    continue
                current = [paragraph[start:start + section_chars]]
                current_chars = len(current[0])
                if not flush():
                    dropped += 1
                    break
            continue
        current.append(paragraph)
        current_chars += len(paragraph) + 2
    flush()
    if len(sections) >= remaining:
        dropped += max(0, len(paragraphs) - sum(1 for _ in sections))
    return sections[:remaining], max(0, dropped)


def format_requirement_doc(index: int, doc: dict[str, Any]) -> str:
    source = str(doc.get("source") or "NMD")
    locator = str(doc.get("locator") or f"chunk {doc.get('chunk_no', '?')}")
    quote = _clean_quote(doc.get("quote") or doc.get("text") or "", 700)
    return (
        f"[Требование {index}] {source} / {locator} / score={float(doc.get('score') or 0.0):.3f}\n"
        f"{quote}"
    )


def checked_source_details(files: list[dict[str, Any]], sections: list[dict[str, Any]]) -> str:
    lines = ["### Проверяемые файлы"]
    for item in files:
        status = f"; ошибка: {item['error']}" if item.get("error") else ""
        lines.append(
            f"- {item.get('filename')} ({item.get('size', 0)} bytes, "
            f"символов={item.get('text_chars', 0)}, секций={item.get('sections', 0)}{status})"
        )
    if sections:
        lines.append("\n### Проверяемые секции")
        for index, section in enumerate(sections[:80], start=1):
            lines.append(
                f"{index}. **{section['file']}** ({section['locator']})\n"
                f"   > {section['quote']}"
            )
    return "\n".join(lines)


def compliance_section_prompt(
    query: str,
    section: dict[str, Any],
    requirements: list[dict[str, Any]],
    *,
    dialog_context: str,
) -> str:
    req_text = "\n\n".join(
        format_requirement_doc(index, doc)
        for index, doc in enumerate(requirements, start=1)
    )
    dialog_block = f"Контекст последних сообщений:\n{dialog_context}\n\n" if dialog_context else ""
    return (
        f"{dialog_block}"
        f"Задача проверки: {query}\n\n"
        f"Проверяемый файл: {section['file']}\n"
        f"Локатор: {section['locator']}\n"
        f"Фрагмент проверяемого документа:\n{trim_text(section['text'], 3500)}\n\n"
        f"Релевантные требования НМД:\n{req_text}\n\n"
        "Сравни фрагмент с требованиями. Верни JSON-массив объектов матрицы с полями "
        "requirement, nmd_source, nmd_locator, nmd_quote, "
        "checked_file, checked_locator, checked_quote, status, risk, recommendation. "
        "status используй из: соответствует, не соответствует, требует ручной проверки."
    )


def parse_matrix_rows(value: str, section: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(row: Any) -> None:
        if not isinstance(row, dict):
            return
        rows.append(
            {
                "requirement": str(row.get("requirement") or ""),
                "nmd_source": str(row.get("nmd_source") or ""),
                "nmd_locator": str(row.get("nmd_locator") or ""),
                "nmd_quote": _clean_quote(row.get("nmd_quote") or "", 360),
                "checked_file": str(row.get("checked_file") or section.get("file") or ""),
                "checked_locator": str(row.get("checked_locator") or section.get("locator") or ""),
                "checked_quote": _clean_quote(row.get("checked_quote") or section.get("quote") or "", 360),
                "status": str(row.get("status") or "требует ручной проверки"),
                "risk": str(row.get("risk") or ""),
                "recommendation": str(row.get("recommendation") or ""),
            }
        )

    text = str(value or "").strip()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        for item in loaded:
            add_row(item)
    elif isinstance(loaded, dict):
        add_row(loaded)
    if rows:
        return rows

    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            add_row(json.loads(line))
        except json.JSONDecodeError:
            continue
    if rows:
        return rows
    return [
        {
            "requirement": "",
            "nmd_source": "",
            "nmd_locator": "",
            "nmd_quote": "",
            "checked_file": section.get("file", ""),
            "checked_locator": section.get("locator", ""),
            "checked_quote": section.get("quote", ""),
            "status": "требует ручной проверки",
            "risk": trim_text(value, 500),
            "recommendation": "Проверьте вывод LLM вручную.",
        }
    ]


def matrix_markdown(rows: list[dict[str, Any]], *, limit: int = 80) -> str:
    if not rows:
        return "_Матрица пуста._"
    header = (
        "| # | Статус | Требование | Проверяемый фрагмент | Риск | Рекомендация |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for index, row in enumerate(rows[:limit], start=1):
        req = _clean_quote(row.get("nmd_quote") or row.get("requirement") or "", 180).replace("|", "\\|")
        checked = _clean_quote(row.get("checked_quote") or "", 180).replace("|", "\\|")
        risk = trim_text(row.get("risk") or "", 160).replace("|", "\\|")
        rec = trim_text(row.get("recommendation") or "", 160).replace("|", "\\|")
        lines.append(f"| {index} | {row.get('status', '')} | {req} | {checked} | {risk} | {rec} |")
    if len(rows) > limit:
        lines.append(f"\n_Показано {limit} из {len(rows)} строк матрицы._")
    return "\n".join(lines)


def compliance_final_prompt(
    query: str,
    rows: list[dict[str, Any]],
    notes: list[str],
    *,
    checked_sources: str,
    nmd_sources: str,
    dialog_context: str,
) -> str:
    dialog_block = f"Контекст последних сообщений:\n{dialog_context}\n\n" if dialog_context else ""
    return (
        f"{dialog_block}"
        f"Задача: {query}\n\n"
        f"Матрица предварительных выводов:\n{matrix_markdown(rows)}\n\n"
        f"Сырьевые заметки LLM:\n{trim_text(chr(10).join(notes), 6000)}\n\n"
        "Сформируй итоговый акт проверки: краткое резюме, найденные несоответствия, "
        "что соответствует, что требует ручной проверки, рекомендации. "
        "Не добавляй внешние требования. В конце добавь HTML-спойлер источников.\n\n"
        f"Источники проверяемых файлов:\n{checked_sources}\n\n"
        f"Источники НМД:\n{nmd_sources}"
    )
