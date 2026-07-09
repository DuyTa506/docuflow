"""Auto-detect a document's source language from its extracted text."""
from __future__ import annotations

from config.settings import normalize_lang_code

_SAMPLE_CHARS = 6000
_SPAN_CHARS = 1800


def sample_representative_text(pages: list[str], span_chars: int = _SPAN_CHARS) -> str:
    """Build a representative text sample from the first, middle, and last
    non-empty pages instead of only the document's leading pages.

    A single leading sample can misdetect language when a document's front
    matter (cover page, library stamp, registration boilerplate) is in a
    different language than the body — confirmed on a 761-page book with
    Vietnamese front matter and a Russian body, which detected as "vi" from
    the first ~4000 chars alone.
    """
    non_empty = [p for p in pages if p and p.strip()]
    if not non_empty:
        return ""
    n = len(non_empty)
    indices = sorted({0, n // 2, n - 1})
    return "\n\n".join(non_empty[i][:span_chars] for i in indices)


def detect_source_language(text: str | None, *, fallback: str = "en") -> str:
    """Detect the language of extracted text, falling back on empty/ambiguous input.

    Runs on a bounded sample rather than the full text — detection accuracy
    plateaus after a few thousand characters, and full-document scans are
    wasted work on large (700+ page) books. Callers with per-page text should
    build a spread sample via `sample_representative_text()` first — this
    function's own leading-chars truncation is a safety cap, not a
    representative-sampling strategy on its own.
    """
    sample = (text or "").strip()[:_SAMPLE_CHARS]
    if not sample:
        return normalize_lang_code(fallback)

    from langdetect import LangDetectException, detect

    try:
        return normalize_lang_code(detect(sample))
    except LangDetectException:
        return normalize_lang_code(fallback)
