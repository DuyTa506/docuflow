"""Shared heading-vs-body classifier for tree build and keyword intake."""

from __future__ import annotations

import re

from utils.heading_patterns import (
    ANCHOR_TITLE_MAX_CHARS,
    NON_ANCHOR_LABELS,
    TITLE_IS_BODY_MIN_CHARS,
    extract_markdown_level,
    extract_numbered_section_level,
)

_HEADINGISH_LABELS = frozenset({"title", "sub_title", "subtitle", "heading", "section_heading"})
_URL_RE = re.compile(r"https?://|www\.", re.I)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|fig(?:ure)?|bảng|hình|рис\.?|таблица)\s*[\d.:]",
    re.I,
)
_AFFILIATION_RE = re.compile(
    r"\b(?:university|department|institute|laboratory|faculty|college)\b",
    re.I,
)
_SENTENCE_END_RE = re.compile(r"[.!?]\s+\S")


def is_structural_title(
    title: str | None,
    *,
    label: str | None = None,
    body: str | None = None,
) -> bool:
    """Return True when ``title`` is plausibly a section heading, not body text."""
    candidate = (title or "").strip()
    if not candidate:
        return False
    if "\n" in candidate:
        return False
    if len(candidate) > ANCHOR_TITLE_MAX_CHARS:
        return False

    normalized_label = (label or "").lower()
    if normalized_label in NON_ANCHOR_LABELS:
        return False

    own_body = (body or "").strip()
    if own_body and candidate == own_body and len(candidate) > TITLE_IS_BODY_MIN_CHARS:
        return False

    if _TABLE_CAPTION_RE.match(candidate):
        return False
    if _URL_RE.search(candidate) and len(candidate) > 60:
        return False
    if _EMAIL_RE.search(candidate):
        return False
    if _AFFILIATION_RE.search(candidate) and len(candidate) > 120:
        return False
    if extract_markdown_level(candidate) is not None:
        return True
    if extract_numbered_section_level(candidate) is not None:
        return True
    from core.spatial.zone_classifier import match_chapter_heading

    if match_chapter_heading(candidate):
        return True

    # A short title can be imported without a layout label in legacy trees.
    # Do not treat sentence-like prose as that kind of title.
    if len(candidate.split()) > 12 or _SENTENCE_END_RE.search(candidate):
        return False

    if normalized_label in _HEADINGISH_LABELS and len(candidate) <= ANCHOR_TITLE_MAX_CHARS:
        if candidate.count(". ") > 1:
            return False
        if candidate.endswith(".") and len(candidate.split()) > 8:
            return False
        return True

    # Imported PageIndex trees do not reliably retain the original label. A
    # compact title-cased line is safe enough as an explicit structural signal.
    words = [word for word in re.findall(r"\w+", candidate) if word]
    return bool(words and any(word[:1].isupper() for word in words))
