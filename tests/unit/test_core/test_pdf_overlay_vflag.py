"""Regression tests for the PDF overlay's formula-detection heuristic
(vflag/is_formula_font_or_char).

Confirmed root cause: the original ported heuristic over-triggers "formula
mode" (untranslated, spliced back in via offset math) on ordinary
italic/bold/embedded fonts and on Vietnamese NFD-decomposed accented
characters -- causing ghosting and character-level "nhảy chữ" on general
(non-STEM) documents. This must still correctly flag genuine LaTeX/math
fonts and formula characters (the original STEM use case)."""

import unicodedata

from core.pdf_overlay.converter import is_formula_font_or_char


class TestFontNameFalsePositivesFixed:
    def test_ordinary_italic_font_is_not_a_formula(self):
        assert is_formula_font_or_char("TimesNewRomanPS-ItalicMT", "a") is False

    def test_calibri_italic_is_not_a_formula(self):
        assert is_formula_font_or_char("Calibri-Italic", "x") is False

    def test_symbol_style_font_name_is_not_a_formula(self):
        assert is_formula_font_or_char("ArialSymbolMT", "x") is False

    def test_monospace_code_font_is_not_a_formula(self):
        assert is_formula_font_or_char("Consolas-Mono", "x") is False


class TestFontNameGenuineLatexStillDetected:
    def test_computer_modern_font_is_formula(self):
        assert is_formula_font_or_char("CMSY10", "x") is True

    def test_tex_prefixed_font_is_formula(self):
        assert is_formula_font_or_char("TeX-cmex10", "x") is True

    def test_subset_prefix_is_stripped_before_matching(self):
        # "ABCDEF+CMSY10" -> font name truncated to "CMSY10" before matching.
        assert is_formula_font_or_char("ABCDEF+CMSY10", "x") is True


class TestCidCharAlwaysFormula:
    def test_cid_placeholder_char_is_formula(self):
        assert is_formula_font_or_char("AnyFont", "(cid:123)") is True


class TestCharCategoryFalsePositivesFixed:
    def test_vietnamese_nfd_combining_diacritic_is_not_a_formula(self):
        # "a" + COMBINING ACUTE ACCENT (U+0301), NFD-decomposed Vietnamese "á".
        combining_accent = unicodedata.normalize("NFD", "á")[1]
        assert unicodedata.category(combining_accent) == "Mn"
        assert is_formula_font_or_char("ArialMT", combining_accent) is False

    def test_modifier_symbol_char_is_not_a_formula(self):
        # U+02DC SMALL TILDE is category "Sk" (modifier symbol).
        assert unicodedata.category("˜") == "Sk"
        assert is_formula_font_or_char("ArialMT", "˜") is False


class TestCharCategoryGenuineMathStillDetected:
    def test_math_symbol_char_is_formula(self):
        # U+2211 N-ARY SUMMATION is category "Sm" (math symbol).
        assert unicodedata.category("∑") == "Sm"
        assert is_formula_font_or_char("ArialMT", "∑") is True

    def test_greek_letter_char_is_formula(self):
        assert is_formula_font_or_char("ArialMT", "α") is True  # alpha


class TestOverrideRegexesStillHonored:
    def test_explicit_vfont_override_still_applies(self):
        assert is_formula_font_or_char("MyCustomFont", "x", vfont=r"MyCustom.*") is True

    def test_explicit_vchar_override_still_applies(self):
        assert is_formula_font_or_char("ArialMT", "Q", vchar=r"Q") is True
