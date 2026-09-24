"""Lightweight heading pattern helpers without spatial pipeline imports."""

from __future__ import annotations

import re
from typing import Optional

ANCHOR_TITLE_MAX_CHARS = 150
TITLE_IS_BODY_MIN_CHARS = 80
NON_ANCHOR_LABELS = frozenset(
    {
        "paragraph",
        "table",
        "figure",
        "image",
        "chart",
        "graph",
        "picture",
        "equation",
        "formula",
        "isolate_formula",
        "caption",
        "footnote",
        "page_header",
        "page_footer",
        "header",
        "footer",
    }
)

_NUMBERED_SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+)*|[A-Z]\.\d+(?:\.\d+)*)(?:\.)?\s+\S"
)


def extract_markdown_level(text: str) -> Optional[int]:
    match = re.match(r"^(#{1,6})\s+", (text or "").strip())
    if match:
        return len(match.group(1)) - 1
    return None


def extract_numbered_section_level(text: str, *, max_len: int = 150) -> Optional[int]:
    stripped = (text or "").strip()
    if len(stripped) > max_len:
        return None
    match = _NUMBERED_SECTION_RE.match(stripped)
    if not match:
        return None
    return match.group(1).count(".") + 1
