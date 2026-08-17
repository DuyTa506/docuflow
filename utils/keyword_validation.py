"""Validation helpers for keyword refinement output."""

from __future__ import annotations

import re
from typing import Any

from utils.structural_titles import is_structural_title

MAX_KEYWORD_PHRASE_LEN = 120
_MIN_STORED_KEYWORDS = 5

_URL_RE = re.compile(r"https?://|www\.", re.I)
_AFFILIATION_RE = re.compile(
    r"\b(?:university|department|institute|laboratory|faculty|college)\b",
    re.I,
)


def _phrase_in_text(phrase: str, source_text: str) -> bool:
    if not phrase or not source_text:
        return False
    return phrase.casefold() in source_text.casefold()


def validate_keyword_item(
    item: Any,
    *,
    source_text: str,
    reject_structural_titles: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "wrong_type"
    keyword = str(item.get("keyword") or "").strip()
    if not keyword:
        return None, "empty"
    if "\n" in keyword:
        return None, "multiline"
    if len(keyword) > MAX_KEYWORD_PHRASE_LEN:
        return None, "too_long"
    if len(keyword.split()) > 12 or re.search(r"[.!?]\s+\S", keyword):
        return None, "paragraph"
    if _URL_RE.search(keyword):
        return None, "url"
    if _AFFILIATION_RE.search(keyword) and len(keyword) > 80:
        return None, "affiliation"
    if reject_structural_titles and is_structural_title(keyword, body=keyword):
        return None, "structural_title"

    try:
        weight = float(item.get("weight", 1.0))
    except (TypeError, ValueError):
        return None, "bad_weight"
    if not (0.0 <= weight <= 1.0):
        return None, "bad_weight"
    if weight < 0.5:
        return None, "low_weight"

    if not _phrase_in_text(keyword, source_text):
        return None, "not_grounded"

    display = str(item.get("display") or "").strip() or keyword
    return {"keyword": keyword, "display": display, "weight": weight}, None


def validate_keyword_batch(
    items: list[Any],
    *,
    source_text: str,
    pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: dict[str, int] = {}

    for item in items or []:
        row, reason = validate_keyword_item(item, source_text=source_text)
        if row is None:
            rejected[reason or "rejected"] = rejected.get(reason or "rejected", 0) + 1
            continue
        key = row["keyword"].casefold()
        if key in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        seen.add(key)
        validated.append(row)
        if len(validated) >= pool_size:
            break

    min_required = min(_MIN_STORED_KEYWORDS, max(1, pool_size // 4))
    diagnostics = {
        "requested": pool_size,
        "parsed": len(items or []),
        "valid": len(validated),
        "min_required": min_required,
        "rejected": rejected,
    }
    return validated, diagnostics
