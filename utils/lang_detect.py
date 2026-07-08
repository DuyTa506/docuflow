"""Auto-detect a document's source language from its extracted text."""
from __future__ import annotations

from config.settings import normalize_lang_code

_SAMPLE_CHARS = 4000


def detect_source_language(text: str | None, *, fallback: str = "en") -> str:
    """Detect the language of extracted text, falling back on empty/ambiguous input.

    Runs on a leading sample rather than the full text — detection accuracy
    plateaus after a few hundred characters, and full-document scans are
    wasted work on large (700+ page) books.
    """
    sample = (text or "").strip()[:_SAMPLE_CHARS]
    if not sample:
        return normalize_lang_code(fallback)

    from langdetect import LangDetectException, detect

    try:
        return normalize_lang_code(detect(sample))
    except LangDetectException:
        return normalize_lang_code(fallback)
