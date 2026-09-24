"""Export-DPI background MinIO cache + parallel page fragment render."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fitz
import pytest

from core.pdf_render.renderer import render_document_pdf
from utils.layout_pdf import get_or_render_export_backgrounds
from utils.storage_keys import export_bg_key, export_bg_prefix


def test_export_bg_key_includes_dpi_max_quality():
    key = export_bg_key("DOC_1", 3, dpi=150, max_size=3000, quality=85)
    assert key == "documents/DOC_1/export_bg/d150_m3000_q85/p0003.jpg"
    assert export_bg_prefix("DOC_1") == "documents/DOC_1/export_bg/"


def test_get_or_render_cache_hit_skips_pixmap():
    storage = MagicMock()
    storage.exists.return_value = True
    storage.get_bytes.return_value = b"cached-jpeg"

    with (
        patch("services.object_storage.get_object_storage", return_value=storage),
        patch("utils.image_utils.render_pdf_page_to_jpeg_bytes") as mock_render,
        patch("config.settings.settings") as mock_settings,
    ):
        mock_settings.layout_pdf_export_dpi = 150
        mock_settings.layout_pdf_export_max_size = 3000
        mock_settings.layout_pdf_export_jpeg_quality = 85
        mock_settings.layout_pdf_render_workers = 1
        # original path None — full cache hit must still work
        out = get_or_render_export_backgrounds("DOC_1", None, [1, 2])

    assert out == {1: b"cached-jpeg", 2: b"cached-jpeg"}
    mock_render.assert_not_called()
    storage.put_bytes.assert_not_called()


def test_get_or_render_cache_hit_with_path_still_skips_render():
    storage = MagicMock()
    storage.exists.return_value = True
    storage.get_bytes.return_value = b"cached-jpeg"

    with (
        patch("services.object_storage.get_object_storage", return_value=storage),
        patch("utils.image_utils.render_pdf_page_to_jpeg_bytes") as mock_render,
        patch("config.settings.settings") as mock_settings,
    ):
        mock_settings.layout_pdf_export_dpi = 150
        mock_settings.layout_pdf_export_max_size = 3000
        mock_settings.layout_pdf_export_jpeg_quality = 85
        mock_settings.layout_pdf_render_workers = 1
        out = get_or_render_export_backgrounds("DOC_1", "/fake.pdf", [1, 2])

    assert out == {1: b"cached-jpeg", 2: b"cached-jpeg"}
    mock_render.assert_not_called()

def test_get_or_render_cache_miss_puts_bytes():
    storage = MagicMock()
    storage.exists.return_value = False

    with (
        patch("services.object_storage.get_object_storage", return_value=storage),
        patch(
            "utils.image_utils.render_pdf_page_to_jpeg_bytes", return_value=b"fresh-jpeg"
        ) as mock_render,
        patch("config.settings.settings") as mock_settings,
    ):
        mock_settings.layout_pdf_export_dpi = 150
        mock_settings.layout_pdf_export_max_size = 3000
        mock_settings.layout_pdf_export_jpeg_quality = 85
        mock_settings.layout_pdf_render_workers = 1
        out = get_or_render_export_backgrounds("DOC_1", "/fake.pdf", [1])

    assert out == {1: b"fresh-jpeg"}
    mock_render.assert_called_once()
    storage.put_bytes.assert_called_once()
    key = storage.put_bytes.call_args[0][0]
    assert "d150_m3000_q85" in key
    assert key.endswith("p0001.jpg")


def _blank_bg(w=200, h=280) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    pix = page.get_pixmap()
    data = pix.tobytes("jpeg")
    doc.close()
    return data


@pytest.mark.parametrize("workers", [1, 4])
def test_render_document_pdf_workers_same_page_count(workers, monkeypatch):
    from config.settings import settings as app_settings

    monkeypatch.setattr(app_settings, "layout_pdf_render_workers", workers)

    pages = [
        SimpleNamespace(
            page_number=i,
            image_width=200,
            image_height=280,
            page_type="scanned",
            image_key=None,
        )
        for i in (1, 2, 3)
    ]
    elements = [
        {
            "page_number": pn,
            "label": "text",
            "text": f"hello page {pn}",
            "bbox": {"x1": 10, "y1": 10, "x2": 180, "y2": 40},
        }
        for pn in (1, 2, 3)
    ]
    bgs = {1: _blank_bg(), 2: _blank_bg(), 3: _blank_bg()}
    result = render_document_pdf(
        pages=pages,
        elements=elements,
        pdf_mode="facsimile",
        text_kind="ocr",
        page_backgrounds=bgs,
    )
    doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
    try:
        assert doc.page_count == 3
    finally:
        doc.close()
    assert result.continuation_pages == 0
    assert result.pdf_mode == "facsimile"
