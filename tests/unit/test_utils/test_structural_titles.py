"""Tests for structural title classification."""

from utils.structural_titles import is_structural_title


def test_numbered_heading_is_structural():
    assert is_structural_title("1. Introduction", label="title")


def test_paragraph_body_rejected():
    paragraph = (
        "Large language models have revolutionized natural language processing by enabling "
        "unprecedented scale in training data and parameter count across many domains."
    )
    assert not is_structural_title(paragraph, label="title", body=paragraph)


def test_affiliation_block_rejected():
    text = (
        "John Smith, Department of Computer Science, Example University, "
        "john@example.edu, https://example.edu/~jsmith"
    )
    assert not is_structural_title(text, label="title")


def test_table_caption_rejected():
    assert not is_structural_title("Table 1: Model performance on Spider", label="caption")


def test_short_chapter_label_accepted():
    assert is_structural_title("Abstract", label="title")
