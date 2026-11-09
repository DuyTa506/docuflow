"""Validation helpers for keyword refinement output."""

from __future__ import annotations

import math
import re
from functools import lru_cache
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


@lru_cache(maxsize=2)
def _words_of(source_text: str) -> frozenset:
    return frozenset(re.findall(r"\w+", source_text.casefold()))


def _grounded(keyword: str, source_text: str) -> bool:
    """The phrase is in the text, or it names a concept the text discusses: every one of
    its words occurs there as a whole word. A term no word of which the document uses is
    still rejected."""
    if _phrase_in_text(keyword, source_text):
        return True
    words = [w for w in re.findall(r"\w+", keyword.casefold()) if len(w) >= 3]
    return bool(words) and all(w in _words_of(source_text) for w in words)


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
    if reject_structural_titles and is_structural_title(keyword, body=keyword, explicit_only=True):
        return None, "structural_title"

    try:
        weight = float(item.get("weight", 1.0))
    except (TypeError, ValueError):
        return None, "bad_weight"
    if not (0.0 <= weight <= 1.0):
        return None, "bad_weight"
    if weight < 0.5:
        return None, "low_weight"

    if not _grounded(keyword, source_text):
        return None, "not_grounded"

    display = str(item.get("display") or "").strip() or keyword
    kept = {"keyword": keyword, "display": display, "weight": weight}
    # The pieces `format_keyword_display` assembles the §2.3 line from.
    for field in ("vi", "en"):
        value = str(item.get(field) or "").strip()
        if value:
            kept[field] = value
    return kept, None


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


# A term used this often saturates the frequency score; log-scaled so the first
# few mentions matter more than the thousandth.
_FREQUENCY_SATURATION = 1000
# How much of the final weight is frequency. A larger rerank pool with a bigger
# share promoted bare physics words («velocity», «mass») over the book's subjects.
_FREQUENCY_SHARE = 0.5


def rank_by_frequency(items: list[dict], source_text: str) -> list[dict]:
    """Blend the model's weight with how often the document uses each term, best first.

    The re-rank model rates a chapter-title phrase as highly as a subject the whole
    book is about, so §2.3 filled with section names. Half of the weight now comes
    from the term's occurrence count in the full text. A term the counter cannot find
    (hyphenation, width folding) gets a neutral frequency rather than the bottom.
    """
    haystack = source_text.casefold()
    ranked = []
    for item in items:
        keyword = str(item["keyword"]).casefold()
        count = haystack.count(keyword)
        if not count:
            # A concept named in other wording: its rarest word, discounted.
            words = [w for w in re.findall(r"\w+", keyword) if len(w) >= 3]
            count = min((haystack.count(w) for w in words), default=0) // 4
        frequency = (
            min(1.0, math.log10(1 + count) / math.log10(1 + _FREQUENCY_SATURATION))
            if count
            else 0.5
        )
        blended = (1 - _FREQUENCY_SHARE) * float(item.get("weight", 1.0))
        ranked.append({**item, "weight": round(blended + _FREQUENCY_SHARE * frequency, 4)})
    return sorted(ranked, key=lambda item: item["weight"], reverse=True)
