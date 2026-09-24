"""OCR text normalization regressions from the 2026-09 E2E run."""

from services.normalization_service import NormalizationService


def _norm(text, lang="en"):
    return NormalizationService().normalize(text, lang)


def test_soft_hyphen_line_break_rejoins_the_word():
    """Ru_Designing captions kept the text layer's «си\\u00ad стемы»."""
    assert _norm("микропроцессорные си­ стемы", "ru") == "микропроцессорные системы"
    assert _norm("си­\nстемы", "ru") == "системы"


def test_markdown_rules_survive():
    text = "Intro\n\n---\n\n***\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = _norm(text)
    assert "\n---\n" in out
    assert "\n***\n" in out
    assert "|---|---|" in out


def test_inline_punctuation_noise_still_collapses():
    assert _norm("Result!!!! is ready") == "Result! is ready"
