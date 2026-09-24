"""Document glossary for translation: source term → established Vietnamese term.

Built from the digest's bilingual keywords (``DocumentKeyword.display`` is
"Tiếng Việt (source term)"), so the same term reads the same in every unit.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

_DISPLAY_RE = re.compile(r"^\s*(?P<vi>[^()]+?)\s*\((?P<src>[^()]+)\)\s*$")
GLOSSARY_PROMPT_MAX_TERMS = 20


def parse_glossary(displays: Iterable[Optional[str]]) -> Dict[str, str]:
    glossary: Dict[str, str] = {}
    for display in displays:
        matched = _DISPLAY_RE.match(display or "")
        if not matched:
            continue
        src, vi = matched.group("src").strip(), matched.group("vi").strip()
        if src and vi and src.casefold() != vi.casefold():
            glossary[src] = vi
    return glossary


def glossary_clause(glossary: Dict[str, str], text: str) -> str:
    """Prompt lines for the glossary terms that occur in ``text`` (or "")."""
    if not glossary or not text:
        return ""
    folded = text.casefold()
    hits = [(src, vi) for src, vi in glossary.items() if src.casefold() in folded]
    if not hits:
        return ""
    lines = "\n".join(f"- {src} → {vi}" for src, vi in hits[:GLOSSARY_PROMPT_MAX_TERMS])
    return f"GLOSSARY (use exactly these translations for these terms):\n{lines}\n"
