"""Merge fine-grained PDF overlay paragraphs before LLM translation."""

from __future__ import annotations

import re
from typing import List, Tuple, TypeVar

T = TypeVar("T")

_FORMULA_ONLY = re.compile(r"^\{v\d+\}$")


def merge_overlay_paragraphs(
    texts: List[str],
    meta: List[T],
    *,
    max_chars: int = 800,
    gap_ratio: float = 2.5,
) -> Tuple[List[str], List[T]]:
    """Join adjacent short paragraphs on the same vertical band."""
    if not texts or len(texts) != len(meta):
        return texts, meta

    def _is_narrow(m: T) -> bool:
        width = float(getattr(m, "x1", 0) - getattr(m, "x0", 0))
        size = float(getattr(m, "size", 12) or 12)
        return width > 0 and width < size * 2.0

    out_texts: List[str] = []
    out_meta: List[T] = []
    cur_text = texts[0]
    cur_meta = meta[0]

    for i in range(1, len(texts)):
        nxt_text = texts[i]
        nxt_meta = meta[i]
        if _FORMULA_ONLY.match(cur_text) or _FORMULA_ONLY.match(nxt_text):
            out_texts.append(cur_text)
            out_meta.append(cur_meta)
            cur_text, cur_meta = nxt_text, nxt_meta
            continue
        # Chart Y-axis / stacked single-glyph columns must stay unmerged —
        # joining them into one narrow box causes vertical character stacks.
        if _is_narrow(cur_meta) or _is_narrow(nxt_meta):
            out_texts.append(cur_text)
            out_meta.append(cur_meta)
            cur_text, cur_meta = nxt_text, nxt_meta
            continue

        size = getattr(cur_meta, "size", 12) or 12
        gap = abs(getattr(nxt_meta, "y0", 0) - getattr(cur_meta, "y1", 0))
        same_band = gap <= size * gap_ratio
        combined = len(cur_text) + len(nxt_text) + 1

        if same_band and combined <= max_chars:
            joiner = "\n" if getattr(cur_meta, "brk", False) else " "
            cur_text = f"{cur_text}{joiner}{nxt_text}"
            cur_meta.x0 = min(cur_meta.x0, nxt_meta.x0)
            cur_meta.x1 = max(cur_meta.x1, nxt_meta.x1)
            cur_meta.y0 = min(cur_meta.y0, nxt_meta.y0)
            cur_meta.y1 = max(cur_meta.y1, nxt_meta.y1)
            cur_meta.brk = getattr(cur_meta, "brk", False) or getattr(nxt_meta, "brk", False)
        else:
            out_texts.append(cur_text)
            out_meta.append(cur_meta)
            cur_text, cur_meta = nxt_text, nxt_meta

    out_texts.append(cur_text)
    out_meta.append(cur_meta)
    return out_texts, out_meta
