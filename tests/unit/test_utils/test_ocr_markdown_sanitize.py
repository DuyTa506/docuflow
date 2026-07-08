"""Tests for OCR markdown sanitization before DOCX export."""

from utils.file_download import build_docx_bytes_from_content
from utils.ocr_markdown import normalize_ocr_markdown, sanitize_xml_text


def test_sanitize_xml_text_strips_control_chars():
    raw = "Hello\x00world\x07test"
    assert sanitize_xml_text(raw) == "Helloworldtest"


def test_normalize_ocr_markdown_strips_null_bytes():
    assert "\x00" not in normalize_ocr_markdown("Page\x00 one\n\n---\n\nPage two")


def test_build_docx_bytes_with_control_chars():
    content = "Line one\x00\nLine two\x1F\n| a | b |\n|---|---|\n| 1 | 2 |"
    body = build_docx_bytes_from_content(content, title="Test\x00 doc", structured=True)
    assert body[:2] == b"PK"
