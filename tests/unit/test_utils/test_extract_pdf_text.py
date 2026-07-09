"""Tests for PDF text extraction used by overlay DOCX export."""

from __future__ import annotations

import fitz

from utils.file_download import extract_pdf_text


def test_extract_pdf_text_joins_pages():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello overlay")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Second page")
    pdf_bytes = doc.tobytes()
    doc.close()

    text = extract_pdf_text(pdf_bytes)
    assert "Hello overlay" in text
    assert "Second page" in text


def test_extract_pdf_text_strips_nul_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello overlay")
    pdf_bytes = doc.tobytes()
    doc.close()

    from unittest.mock import patch

    with patch("fitz.Page.get_text", return_value="Hello\x00overlay"):
        text = extract_pdf_text(pdf_bytes)

    assert "\x00" not in text
    assert text == "Hellooverlay"
