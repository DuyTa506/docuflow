"""Unit tests for preview text helpers."""

from utils.preview_text import preview_flat_text, preview_translated_elements


class TestPreviewFlatText:
    def test_no_preview_returns_full(self):
        text = "a" * 10000
        out, truncated = preview_flat_text(text, None)
        assert out == text
        assert truncated is False

    def test_preview_truncates_long_text(self):
        text = "x" * 10000
        out, truncated = preview_flat_text(text, 2, chars_per_page=100)
        assert truncated is True
        assert len(out) < len(text)


class TestPreviewTranslatedElements:
    def test_filters_by_page_number(self):
        elements = [
            {"page_number": 1, "text_content": "Page one"},
            {"page_number": 2, "text_content": "Page two"},
            {"page_number": 3, "text_content": "Page three"},
        ]
        out, truncated = preview_translated_elements(elements, 2)
        assert "Page three" not in out
        assert truncated is True
