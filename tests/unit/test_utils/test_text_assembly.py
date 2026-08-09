"""Tests for OCR text assembly from pages."""

from types import SimpleNamespace

from utils.text_assembly import assemble_ocr_from_pages


def test_assemble_ocr_from_pages_orders_and_formats():
    pages = [
        SimpleNamespace(page_number=2, markdown_content="Page two"),
        SimpleNamespace(page_number=1, markdown_content="Page one"),
    ]
    out = assemble_ocr_from_pages(pages)
    assert "# Page 1" in out
    assert "Page one" in out
    assert out.index("Page one") < out.index("Page two")
