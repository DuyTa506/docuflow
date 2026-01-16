"""
Unit tests for utils.text_utils module.
"""
import pytest
from utils.text_utils import (
    clean_grounding_format,
    extract_text_from_grounding,
    strip_markdown_headers
)


class TestCleanGroundingFormat:
    """Tests for clean_grounding_format function."""
    
    def test_remove_grounding_tags(self):
        """Test removing grounding tags from text."""
        text = "Before <|ref|>title<|/ref|><|det|>[[10,20,30,40]]<|/det|> After"
        
        result = clean_grounding_format(text)
        
        assert '<|ref|>' not in result
        assert '<|/ref|>' not in result
        assert '<|det|>' not in result
        assert '<|/det|>' not in result
    
    def test_keep_images_placeholder(self):
        """Test keeping image placeholders when requested."""
        text = "Text <|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|> more text"
        
        result = clean_grounding_format(text, keep_images=True)
        
        assert '[Figure 1]' in result or 'Figure' in result
    
    def test_remove_images_default(self):
        """Test removing images by default."""
        text = "Text <|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|> more text"
        
        result = clean_grounding_format(text, keep_images=False)
        
        # Image reference should be removed
        assert 'image<|/ref|>' not in result
    
    def test_multiple_grounding_tags(self):
        """Test removing multiple grounding tags."""
        text = """
        <|ref|>title<|/ref|><|det|>[[1,1,1,1]]<|/det|>
        <|ref|>text<|/ref|><|det|>[[2,2,2,2]]<|/det|>
        """
        
        result = clean_grounding_format(text)
        
        assert '<|ref|>' not in result
        assert '<|det|>' not in result
    
    def test_empty_text(self):
        """Test with empty string."""
        result = clean_grounding_format("")
        
        assert result == ""
    
    def test_text_without_tags(self):
        """Test text without any grounding tags."""
        text = "Just plain markdown text"
        
        result = clean_grounding_format(text)
        
        assert result == text.strip()


class TestExtractTextFromGrounding:
    """Tests for extract_text_from_grounding function."""
    
    def test_extract_plain_text(self):
        """Test extracting text without grounding tags."""
        text = "Some text <|ref|>label<|/ref|><|det|>[[1,2,3,4]]<|/det|> more text"
        
        result = extract_text_from_grounding(text)
        
        assert 'Some text' in result
        assert 'more text' in result
        assert '<|ref|>' not in result
    
    def test_extract_from_empty(self):
        """Test extracting from empty string."""
        result = extract_text_from_grounding("")
        
        assert result == ""


class TestStripMarkdownHeaders:
    """Tests for strip_markdown_headers function."""
    
    def test_strip_single_hash(self):
        """Test stripping single # header."""
        text = "# Title"
        
        result = strip_markdown_headers(text)
        
        assert result == "Title"
    
    def test_strip_multiple_hashes(self):
        """Test stripping multiple ## headers."""
        text = "## Subtitle"
        
        result = strip_markdown_headers(text)
        
        assert result == "Subtitle"
    
    def test_strip_multiline(self):
        """Test stripping headers from multiple lines."""
        text = "# Title\n## Subtitle\nPlain text"
        
        result = strip_markdown_headers(text)
        
        assert "# " not in result
        assert "## " not in result
        assert "Title" in result
        assert "Subtitle" in result
        assert "Plain text" in result
    
    def test_preserve_plain_text(self):
        """Test plain text without headers is preserved."""
        text = "Just plain text"
        
        result = strip_markdown_headers(text)
        
        assert result == text
