from __future__ import annotations

import re

WHITESPACE_RE = re.compile(r"\s+")
LOCATOR_PREFIX_RE = re.compile(r"^((?:\[[^\]]+\]\s*)+)(.*)$")
SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?…])\s+(?=(?:\[[^\]]+\]\s*)?[A-ZА-ЯЁ0-9])"
)
CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[;:])\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def _join_units(units: list[str]) -> str:
    return " ".join(unit for unit in units if unit)


def _joined_len(units: list[str]) -> int:
    if not units:
        return 0
    return sum(len(unit) for unit in units) + len(units) - 1


def _candidate_len(current_len: int, has_current: bool, unit: str) -> int:
    if not has_current:
        return len(unit)
    return current_len + 1 + len(unit)


def _split_locator_prefix(text: str) -> tuple[str, str]:
    match = LOCATOR_PREFIX_RE.match(text)
    if not match:
        return "", text
    return normalize_whitespace(match.group(1)), normalize_whitespace(match.group(2))


def _split_by_words(text: str, max_chars: int) -> list[str]:
    words = normalize_whitespace(text).split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        candidate_len = _candidate_len(current_len, bool(current), word)
        if current and candidate_len > max_chars:
            parts.append(_join_units(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = candidate_len
    if current:
        parts.append(_join_units(current))
    return parts


def _sentence_units(text: str, max_chars: int) -> list[str]:
    normalized = normalize_whitespace(text)
    sentences = [
        part
        for part in SENTENCE_BOUNDARY_RE.split(normalized)
        if part
    ]
    units: list[str] = []
    for sentence in sentences or [normalized]:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue

        clauses = [
            part
            for part in CLAUSE_BOUNDARY_RE.split(sentence)
            if part
        ]
        if len(clauses) == 1:
            units.extend(_split_by_words(sentence, max_chars))
            continue

        for clause in clauses:
            if len(clause) <= max_chars:
                units.append(clause)
            else:
                units.extend(_split_by_words(clause, max_chars))
    return units


def _tail_overlap(units: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(units) <= 1:
        return []
    selected: list[str] = []
    current_len = 0
    for unit in reversed(units):
        candidate_len = _candidate_len(current_len, bool(selected), unit)
        if selected and candidate_len > overlap_chars:
            break
        if candidate_len <= overlap_chars:
            selected.append(unit)
            current_len = candidate_len
    if len(selected) >= len(units):
        return []
    return list(reversed(selected))


def _pack_units(
    units: list[str],
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> list[str]:
        previous = list(current)
        chunk = _join_units(previous)
        if chunk:
            chunks.append(chunk)
        return previous

    def replace_with_overlap(previous: list[str], next_unit: str) -> tuple[list[str], int]:
        overlap = _tail_overlap(previous, overlap_chars)
        if overlap and _candidate_len(_joined_len(overlap), True, next_unit) <= max_chars:
            return overlap, _joined_len(overlap)
        return [], 0

    for unit in units:
        unit = normalize_whitespace(unit)
        if not unit:
            continue

        if current and current_len >= target_chars:
            previous = flush()
            current, current_len = replace_with_overlap(previous, unit)

        candidate_len = _candidate_len(current_len, bool(current), unit)
        if current and candidate_len > max_chars:
            previous = flush()
            current, current_len = replace_with_overlap(previous, unit)

        if not current:
            current.append(unit)
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += 1 + len(unit)

    chunk = _join_units(current)
    if chunk:
        chunks.append(chunk)
    return chunks


def _paragraph_pieces(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    prefix, body = _split_locator_prefix(paragraph)
    if not body:
        return [prefix] if prefix else []

    max_chars = max(chunk_size, int(chunk_size * 1.35))
    body_max_chars = max(80, max_chars - len(prefix) - (1 if prefix else 0))
    body_target_chars = max(80, chunk_size - len(prefix) - (1 if prefix else 0))
    body_overlap_chars = max(0, min(chunk_overlap, body_target_chars // 2))

    units = _sentence_units(body, body_max_chars)
    bodies = _pack_units(units, body_target_chars, body_max_chars, body_overlap_chars)
    return [
        _join_units([prefix, item]) if prefix else item
        for item in bodies
        if item.strip()
    ]


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        chunk_size = 1200
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    max_chars = max(chunk_size, int(chunk_size * 1.35))
    pieces: list[str] = []
    for paragraph in re.split(r"\n{2,}", text or ""):
        paragraph = normalize_whitespace(paragraph)
        if paragraph:
            pieces.extend(_paragraph_pieces(paragraph, chunk_size, chunk_overlap))

    chunks = _pack_units(pieces, chunk_size, max_chars, 0)
    return [chunk for chunk in chunks if chunk.strip()]
