"""Auto-detect a document's source language from its extracted text."""

from __future__ import annotations

import re

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


_MARKUP_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]*\$"  # LaTeX math
    r"|<[^>]+>"  # HTML tags (OCR tables)
    r"|https?://\S+"
    r"|[#*_`|>\-=~\[\]()]+",  # markdown syntax
    re.S,
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_MIN_LETTERS = 20
_MIN_CONFIDENCE = 0.8


def _strip_markup(text: str) -> str:
    return _MARKUP_RE.sub(" ", text)


def _script_language(letters: list[str]) -> str | None:
    """Decide by writing system when one non-Latin script dominates.

    langdetect is unreliable on short CJK pages mixed with pinyin/Latin names
    (DOC_009 cover detected as "vi" at 0.99), while script share is not.
    """
    total = len(letters)
    joined = "".join(letters)
    cjk = len(_CJK_RE.findall(joined)) + len(_KANA_RE.findall(joined))
    if cjk / total >= 0.3:
        return (
            "ja" if _KANA_RE.search(joined) and len(_KANA_RE.findall(joined)) / cjk > 0.1 else "zh"
        )
    if len(_CYRILLIC_RE.findall(joined)) / total >= 0.5:
        return "ru"
    return None


def detect_source_language(text: str | None, *, fallback: str = "en") -> str:
    """Detect the language of extracted text, falling back on empty/ambiguous input.

    Runs on a bounded sample rather than the full text — detection accuracy
    plateaus after a few thousand characters, and full-document scans are
    wasted work on large (700+ page) books. Callers with per-page text should
    build a spread sample via `sample_representative_text()` first — this
    function's own leading-chars truncation is a safety cap, not a
    representative-sampling strategy on its own.

    Markup (HTML tables, LaTeX, markdown) is stripped first; a dominant CJK or
    Cyrillic script decides directly; otherwise langdetect must be at least
    ``_MIN_CONFIDENCE`` sure, else ``fallback`` (the user's choice) wins.
    """
    sample = _strip_markup((text or "").strip()[:_SAMPLE_CHARS])
    letters = [ch for ch in sample if ch.isalpha()]
    if len(letters) < _MIN_LETTERS:
        return normalize_lang_code(fallback)

    by_script = _script_language(letters)
    if by_script:
        return by_script

    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    try:
        best = detect_langs(sample)[0]
    except (LangDetectException, IndexError):
        return normalize_lang_code(fallback)
    if best.prob < _MIN_CONFIDENCE:
        return normalize_lang_code(fallback)
    return normalize_lang_code(best.lang)
