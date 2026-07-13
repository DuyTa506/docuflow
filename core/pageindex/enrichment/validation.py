"""Translation output validation.

Cheap, dependency-free checks that catch the failure modes a local LLM
actually produces on long documents: empty completions, slot-limit
truncation, degenerate repetition loops, and answers left in the source
language. Callers retry on failure and degrade to source text after
exhausting retries — never hard-fail a whole document on one bad unit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from config.settings import normalize_lang_code
from utils.lang_detect import detect_source_language

# Below this many chars, language detection is too noisy (titles, labels).
_LANG_CHECK_MIN_CHARS = 200
# Length-ratio sanity only applies to real paragraphs, not short strings.
_RATIO_CHECK_MIN_CHARS = 80
_RATIO_BOUNDS = (0.25, 4.0)
_DEGENERATE_NGRAM = 4
_DEGENERATE_MIN_WORDS = 24
_DEGENERATE_THRESHOLD = 0.3


@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None


def _is_degenerate(text: str) -> bool:
    """True for repetition loops: either one n-gram dominates, or the n-gram
    vocabulary is tiny relative to the text (a K-word cycle spreads its
    frequency over K rotations, so the max-frequency check alone misses it)."""
    words = text.split()
    if len(words) < _DEGENERATE_MIN_WORDS:
        return False
    grams = [
        tuple(words[i : i + _DEGENERATE_NGRAM]) for i in range(len(words) - _DEGENERATE_NGRAM + 1)
    ]
    counts = Counter(grams)
    if counts.most_common(1)[0][1] / len(grams) > _DEGENERATE_THRESHOLD:
        return True
    unique_ratio = len(counts) / len(grams)
    return unique_ratio < 0.15


def validate_translation(
    source: str,
    output: str,
    target_lang: str,
    finish_reason: Optional[str] = None,
) -> ValidationResult:
    src = (source or "").strip()
    out = (output or "").strip()

    if not src:
        return ValidationResult(True)
    if not out:
        return ValidationResult(False, "empty")
    if finish_reason == "length":
        return ValidationResult(False, "truncated")
    if _is_degenerate(out):
        return ValidationResult(False, "degenerate")

    if len(src) > _RATIO_CHECK_MIN_CHARS:
        ratio = len(out) / len(src)
        lo, hi = _RATIO_BOUNDS
        if not (lo <= ratio <= hi):
            return ValidationResult(False, "length_ratio")

    if len(out) > _LANG_CHECK_MIN_CHARS:
        target = normalize_lang_code(target_lang)
        detected = detect_source_language(out, fallback=target)
        if detected != target:
            return ValidationResult(False, "wrong_language")

    return ValidationResult(True)
