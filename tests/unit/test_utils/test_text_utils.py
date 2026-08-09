"""
Unit tests for utils.text_utils module.
"""

import pytest

from utils.text_utils import clean_grounding_format


class TestCleanGroundingFormat:
    """Tests for clean_grounding_format function."""

    def test_remove_grounding_tags(self):
        """Test removing grounding tags from text."""
        text = "Before <|ref|>title<|/ref|><|det|>[[10,20,30,40]]<|/det|> After"

        result = clean_grounding_format(text)

        assert "<|ref|>" not in result
        assert "<|/ref|>" not in result
        assert "<|det|>" not in result
        assert "<|/det|>" not in result

    def test_keep_images_placeholder(self):
        """Test keeping image placeholders when requested."""
        text = "Text <|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|> more text"

        result = clean_grounding_format(text, keep_images=True)

        assert "[Figure 1]" in result or "Figure" in result

    def test_remove_images_default(self):
        """Test removing images by default."""
        text = "Text <|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|> more text"

        result = clean_grounding_format(text, keep_images=False)

        # Image reference should be removed
        assert "image<|/ref|>" not in result

    def test_multiple_grounding_tags(self):
        """Test removing multiple grounding tags."""
        text = """
        <|ref|>title<|/ref|><|det|>[[1,1,1,1]]<|/det|>
        <|ref|>text<|/ref|><|det|>[[2,2,2,2]]<|/det|>
        """

        result = clean_grounding_format(text)

        assert "<|ref|>" not in result
        assert "<|det|>" not in result

    def test_empty_text(self):
        """Test with empty string."""
        result = clean_grounding_format("")

        assert result == ""

    def test_text_without_tags(self):
        """Test text without any grounding tags."""
        text = "Just plain markdown text"

        result = clean_grounding_format(text)

        assert result == text.strip()
