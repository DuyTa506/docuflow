"""Detect and batch table-of-contents text for translation.

A full TOC page is often one layout element (30+ leader-dot lines). Sending
that as a single LLM unit frequently truncates or leaves Russian; translating
small line batches is reliable and still preserves leader dots / page numbers.
"""

from __future__ import annotations

import re

# Same pattern as utils/chapter_units.py — short line ending in a page number.
_TOC_TAIL_RE = re.compile(r"(?:\.{2,}|\s)\d{1,4}\s*$")
_TOC_LINE_MAX_CHARS = 160
_TOC_MIN_LINES = 6
_TOC_MIN_HIT_RATIO = 0.55
# Soft ceiling so a 40-line TOC becomes ~4 LLM calls, not 40.
_DEFAULT_BATCH_LINES = 10


def looks_like_toc_text(text: str) -> bool:
    """True when most non-empty lines look like TOC / index entries."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < _TOC_MIN_LINES:
        return False
    hits = sum(
        1
        for ln in lines
        if len(ln) <= _TOC_LINE_MAX_CHARS and _TOC_TAIL_RE.search(ln)
    )
    return hits >= max(_TOC_MIN_LINES, int(_TOC_MIN_HIT_RATIO * len(lines)))


def split_toc_line_batches(text: str, *, lines_per_batch: int = _DEFAULT_BATCH_LINES) -> list[str]:
    """Split TOC text into ordered batches of lines (single newlines preserved)."""
    raw_lines = (text or "").splitlines()
    if not raw_lines:
        return []
    n = max(1, int(lines_per_batch))
    batches: list[str] = []
    for i in range(0, len(raw_lines), n):
        chunk = "\n".join(raw_lines[i : i + n]).strip()
        if chunk:
            batches.append(chunk)
    return batches or ([text.strip()] if text and text.strip() else [])
