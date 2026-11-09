"""Translate the cell text of an HTML or markdown table, keeping its structure.

Tables used to pass through untranslated like images (E2E: 46/46 tables of
Digital IC Design stayed English). Only the text between tags / pipes is sent
to the model — markup, spans and purely numeric cells never are.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, List, Optional

_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
_PIPE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
# A cell is worth translating when it holds a real word: part codes such as
# "50B10-5D" or values such as ".650" / "348,000" are kept verbatim.
_WORD_RE = re.compile(r"[^\W\d_]{2,}")

CellTranslator = Callable[[List[str]], Awaitable[List[str]]]


def _needs_translation(text: str) -> bool:
    return not text.startswith("<") and bool(_WORD_RE.search(text))


async def _translate_segments(segments: List[str], translate_cells: CellTranslator) -> List[str]:
    """Translate the translatable entries of ``segments`` in one call, in place."""
    targets = [i for i, seg in enumerate(segments) if _needs_translation(seg)]
    if not targets:
        return segments
    stripped = [segments[i].strip() for i in targets]
    translated = await translate_cells(stripped)
    out = list(segments)
    for i, source, new in zip(targets, stripped, translated):
        # Keep the surrounding whitespace so markup layout is unchanged.
        out[i] = segments[i].replace(source, new, 1) if new else segments[i]
    return out


async def translate_table_text(text: str, translate_cells: CellTranslator) -> Optional[str]:
    """Return ``text`` with its cells translated, or None when it is not a table."""
    if "<table" in text.lower():
        return "".join(await _translate_segments(_TAG_SPLIT_RE.split(text), translate_cells))

    lines = text.split("\n")
    if not any(line.lstrip().startswith("|") for line in lines):
        return None
    cells: List[str] = []
    layout: List[Optional[List[int]]] = []
    for line in lines:
        if not line.lstrip().startswith("|") or _PIPE_SEPARATOR_RE.match(line):
            layout.append(None)
            continue
        parts = line.split("|")
        layout.append(list(range(len(cells), len(cells) + len(parts))))
        cells.extend(parts)
    cells = await _translate_segments(cells, translate_cells)
    return "\n".join(
        line if idx is None else "|".join(cells[i] for i in idx) for line, idx in zip(lines, layout)
    )
