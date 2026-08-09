"""Chapter headings had no textual rescue, so geometry alone decided structure.

`_NUMBERED_SECTION_RE` in spatial_tree_builder matches "1.1.2 Foo" but not
"Глава 1" / "Chương 1" / "Phụ lục A". Numbered *subsections* therefore received
correct deep levels while their unnumbered parent chapters kept a noisy
percentile level — frequently a shallower one — promoting subsections to the
top level and orphaning the real chapters.

zone_classifier already owned the repo's heading vocabulary (Vietnamese and
English), but it was only consumed for reading order. These matchers make it
usable for structure, and add the Russian forms the test corpus needs.
"""

from core.spatial.zone_classifier import (
    SECTION_PATTERNS,
    match_chapter_heading,
    parse_section_ordinal,
)


class TestMatchChapterHeading:
    def test_matches_russian_chapter_and_appendix(self):
        assert match_chapter_heading("Глава 1. Введение")[:3] == ("chapter", "1", 1)
        assert match_chapter_heading("ГЛАВА 12")[:3] == ("chapter", "12", 12)
        assert match_chapter_heading("Приложение А")[0] == "appendix"
        assert match_chapter_heading("Часть II")[:3] == ("part", "II", 2)

    def test_matches_vietnamese_and_english(self):
        assert match_chapter_heading("Chương 3: Tầng logic số")[:3] == ("chapter", "3", 3)
        assert match_chapter_heading("Phụ lục B")[0] == "appendix"
        assert match_chapter_heading("Chapter 7 — Assembly")[:3] == ("chapter", "7", 7)
        assert match_chapter_heading("Appendix C")[0] == "appendix"
        assert match_chapter_heading("Part II")[:3] == ("part", "II", 2)

    def test_appendix_ordinal_is_optional(self):
        """Russian books routinely label a single appendix with no ordinal."""
        matched = match_chapter_heading("Приложение")
        assert matched is not None and matched[0] == "appendix"
        assert match_chapter_heading("Глава") is None, "a chapter without an ordinal is prose"

    def test_rejects_body_mention_and_overlong_text(self):
        assert match_chapter_heading("как показано в Главе 2, конвейер работает иначе") is None
        assert match_chapter_heading("Глава 1. " + "очень длинный текст " * 20) is None
        assert match_chapter_heading("") is None
        assert match_chapter_heading(None) is None


class TestConfigurableVocabulary:
    """Adding a language must be a data change in config/chapter_vocab.json."""

    def test_shipped_config_covers_the_documented_languages(self):
        from core.spatial.zone_classifier import CHAPTER_VOCAB

        assert {"en", "vi", "ru", "fr", "de", "es", "zh", "ja"} <= set(CHAPTER_VOCAB)

    def test_languages_beyond_the_hardcoded_three(self):
        assert match_chapter_heading("Chapitre 4")[:3] == ("chapter", "4", 4)
        assert match_chapter_heading("Kapitel 2")[:3] == ("chapter", "2", 2)
        assert match_chapter_heading("Capítulo 5")[:3] == ("chapter", "5", 5)
        assert match_chapter_heading("Annexe A")[0] == "appendix"

    def test_suffix_ordinal_languages_use_regex_patterns(self):
        """Chinese/Japanese put the ordinal inside the marker: 第1章."""
        assert match_chapter_heading("第1章 计算机体系结构")[:3] == ("chapter", "1", 1)
        assert match_chapter_heading("第十二章")[:3] == ("chapter", "十二", 12)
        assert match_chapter_heading("附录 A")[0] == "appendix"

    def test_language_restriction(self):
        assert match_chapter_heading("Глава 1", languages=["ru"]) is not None
        assert match_chapter_heading("Глава 1", languages=["en", "vi"]) is None


class TestParseSectionOrdinal:
    def test_accepts_both_dotted_and_bare_numbering(self):
        assert parse_section_ordinal("16. Title") == (16,)
        assert parse_section_ordinal("16 Title") == (16,)
        assert parse_section_ordinal("1.2 Background") == (1, 2)
        assert parse_section_ordinal("1.2. Background") == (1, 2)

    def test_rejects_non_headings(self):
        assert parse_section_ordinal("Introduction") is None
        assert parse_section_ordinal("2013 was a good year for computer architecture books") is None
        assert parse_section_ordinal("") is None


def test_section_patterns_still_match_legacy_inputs():
    """SECTION_PATTERNS is rebuilt from the vocabulary — old consumers must hold."""
    import re

    for text in ("1. Introduction", "1.2. Methods", "Chapter 4", "Chương 2"):
        assert any(re.match(p, text) for p in SECTION_PATTERNS), text
    # Russian is new; it was missing from the vocabulary entirely.
    assert any(re.match(p, "Глава 5") for p in SECTION_PATTERNS)


def test_section_patterns_legacy_quirk_is_unchanged():
    """`1.2.3 Methods` never matched, despite the comment in zone_classifier.

    The numeric pattern requires a dot before the whitespace, so a three-level
    heading without a trailing dot falls through. Pre-existing behaviour, left
    alone here so zone classification does not shift for every document;
    `parse_section_ordinal` handles both forms for the structural path.
    """
    import re

    assert not any(re.match(p, "1.2.3 Methods") for p in SECTION_PATTERNS)
    assert parse_section_ordinal("1.2.3 Methods") == (1, 2, 3)
