from __future__ import annotations

import re
from typing import Any


STOP_WORDS = {
    "что", "это", "как", "для", "при", "или", "если", "где", "когда",
    "какие", "какой", "какая", "какое", "кто", "чем", "про", "над",
    "под", "без", "есть", "такое", "нужно", "нужна", "нужен",
    "документ", "документы", "регламент", "порядок", "методика",
    "политика", "раздел", "пункт", "what", "where", "when", "which",
    "that", "this", "with", "from", "about", "все", "всё",
}

STOP_STEMS = {
    "нмд", "норма", "метод", "докум", "регла", "поряд", "полны",
    "переч", "проан", "списо", "пункт", "котор", "прове", "работ",
    "свой", "наруш",
}


def score(doc: dict[str, Any]) -> float:
    try:
        return float(doc.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def trim_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return (trimmed or text[:max_chars]).rstrip() + "..."


def clean_quote_text(text: str) -> str:
    quote = re.sub(r"\[[^\]]+\]", " ", str(text or ""))
    quote = re.sub(r"(?:\s*\|\s*(?:-|–|—|v|V|x|X|✓|✔)?\s*){3,}", " ", quote)
    quote = re.sub(r"\s+\|(?=\s*[.;,]|$)", " ", quote)
    quote = re.sub(r"\|\s+", "| ", quote)
    quote = re.sub(r"\s+", " ", quote).strip()
    return quote.strip(" |")


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", (query or "").lower()):
        token = token.replace("ё", "е")
        if token in STOP_WORDS:
            continue
        if len(token) == 5 and token[-1] in "аеиоуыэюя":
            token = token[:-1]
        elif len(token) > 5:
            token = token[:5]
        if token in STOP_WORDS or token in STOP_STEMS:
            continue
        if token and token not in terms:
            terms.append(token)
    return terms[:12]


def query_term_hits(terms: list[str], doc: dict[str, Any]) -> int:
    if not terms:
        return 0
    haystack = " ".join(
        str(doc.get(key) or "")
        for key in ("source", "locator", "quote", "text")
    ).lower().replace("ё", "е")
    return sum(1 for term in terms if term in haystack)


def dedupe_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _doc_seen_key(doc: dict[str, Any]) -> tuple[str, str, int]:
    text = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip().lower()
    if text:
        return ("text", text[:1200], 0)
    return (
        str(doc.get("source") or ""),
        str(doc.get("chunk_id") or doc.get("chunk_no") or ""),
        0,
    )


def _merge_backfill(
    candidates: list[dict[str, Any]],
    backfill: list[dict[str, Any]],
    terms: list[str],
) -> list[dict[str, Any]]:
    primary = sorted(
        candidates,
        key=lambda doc: (query_term_hits(terms, doc), score(doc)),
        reverse=True,
    )
    merged: list[dict[str, Any]] = list(primary)
    seen = {_doc_seen_key(doc) for doc in primary}
    backfill_only: list[dict[str, Any]] = []
    for doc in backfill:
        key = _doc_seen_key(doc)
        if key in seen:
            continue
        seen.add(key)
        backfill_only.append(doc)
    backfill_only.sort(
        key=lambda doc: (query_term_hits(terms, doc), score(doc)),
        reverse=True,
    )
    merged.extend(backfill_only)
    return merged


def filter_docs_for_prompt(
    query: str,
    docs: list[dict[str, Any]],
    *,
    min_score: float,
    margin: float,
    max_docs: int,
    min_hits: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped = dedupe_docs(docs)
    if not deduped:
        return [], {
            "raw": len(docs),
            "deduped": 0,
            "filtered": 0,
            "best_score": 0.0,
            "score_floor": min_score,
        }

    terms = query_terms(query)
    best_score = max(score(doc) for doc in deduped)
    if best_score < min_score:
        return [], {
            "raw": len(docs),
            "deduped": len(deduped),
            "filtered": 0,
            "best_score": best_score,
            "score_floor": min_score,
        }

    score_floor = max(min_score, best_score - margin)
    candidates = [doc for doc in deduped if score(doc) >= score_floor]

    required_hits = min_hits
    ordered_by_backfill = False
    if terms and min_hits > 0:
        if len(terms) >= 4 and min_hits <= 1:
            required_hits = 2
        lexical = [doc for doc in candidates if query_term_hits(terms, doc) >= required_hits]
        if len(lexical) < min(3, len(candidates)) and required_hits > 1:
            lexical = [doc for doc in candidates if query_term_hits(terms, doc) >= min_hits]
        if lexical:
            candidates = lexical

        backfill_hits = max(required_hits + 2, 4)
        backfill = [
            doc for doc in deduped
            if score(doc) >= min_score and query_term_hits(terms, doc) >= backfill_hits
        ]
        if backfill:
            candidates = _merge_backfill(candidates, backfill, terms)
            ordered_by_backfill = True

    if not ordered_by_backfill:
        candidates.sort(
            key=lambda doc: (query_term_hits(terms, doc), score(doc)),
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


def filter_analysis_docs(
    query: str,
    docs: list[dict[str, Any]],
    *,
    min_score: float,
    margin: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_count = len(docs)
    docs = dedupe_docs(docs)
    if not docs:
        return [], {"raw": raw_count, "deduped": 0, "filtered": 0, "best_score": 0.0, "score_floor": min_score}
    best_score = max(score(doc) for doc in docs)
    if best_score < min_score:
        return [], {
            "raw": raw_count,
            "deduped": len(docs),
            "filtered": 0,
            "best_score": best_score,
            "score_floor": min_score,
        }
    terms = query_terms(query)
    score_floor = max(float(min_score), best_score - margin)
    candidates = [doc for doc in docs if score(doc) >= score_floor]
    required_hits = 1
    ordered_by_backfill = False
    if terms:
        required_hits = 2 if len(terms) >= 4 else 1
        lexical = [doc for doc in candidates if query_term_hits(terms, doc) >= required_hits]
        if len(lexical) < min(3, len(candidates)) and required_hits > 1:
            lexical = [doc for doc in candidates if query_term_hits(terms, doc) >= 1]
        if lexical:
            candidates = lexical
        backfill_hits = max(required_hits + 2, 4)
        backfill = [
            doc for doc in docs
            if score(doc) >= min_score and query_term_hits(terms, doc) >= backfill_hits
        ]
        if backfill:
            candidates = _merge_backfill(candidates, backfill, terms)
            ordered_by_backfill = True
    if not ordered_by_backfill:
        candidates.sort(
            key=lambda doc: (query_term_hits(terms, doc), score(doc)),
            reverse=True,
        )
    return candidates, {
        "raw": raw_count,
        "deduped": len(docs),
        "filtered": len(candidates),
        "best_score": best_score,
        "score_floor": score_floor,
    }


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def compact_dialog_context(messages: list[dict[str, Any]], *, max_messages: int = 4) -> str:
    lines: list[str] = []
    for message in messages[-max_messages:]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = trim_text(message_text(message), 500)
        if not text:
            continue
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)
