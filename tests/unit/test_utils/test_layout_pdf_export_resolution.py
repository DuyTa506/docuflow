"""Regression tests: exported layout PDFs must not reuse the OCR model's
low-res (1344px-capped) page image as the background -- that image is tuned
for the vision model's tiling behavior, not for human zooming."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import fitz

from utils.layout_pdf import build_layout_pdf_bytes, render_export_backgrounds


def _blank_page_png_bytes(w=100, h=100) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    pix = page.get_pixmap()
    data = pix.tobytes("png")
    doc.close()
    return data


class TestRenderExportBackgrounds:
    def test_no_original_path_returns_empty(self):
        assert render_export_backgrounds(None, [1, 2, 3]) == {}

    def test_renders_each_requested_page(self):
        fake_b64 = "not-real-base64-but-decodable"
        import base64

        encoded = base64.b64encode(b"fake-jpeg-bytes").decode()
        with patch(
            "utils.image_utils.render_pdf_page_to_base64", return_value=encoded
        ):
            result = render_export_backgrounds("/fake/path.pdf", [1, 2])

        assert result == {1: b"fake-jpeg-bytes", 2: b"fake-jpeg-bytes"}

    def test_uses_settings_dpi_and_max_size(self):
        import base64

        encoded = base64.b64encode(b"x").decode()
        with patch(
            "utils.image_utils.render_pdf_page_to_base64", return_value=encoded
        ) as mock_render:
            render_export_backgrounds("/fake/path.pdf", [1])

        _, kwargs = mock_render.call_args
        from config.settings import settings

        assert kwargs["target_dpi"] == settings.layout_pdf_export_dpi
        assert kwargs["max_size"] == settings.layout_pdf_export_max_size
        # The whole point: this must not be the OCR model's small cap.
        assert kwargs["max_size"] > 1344

    def test_per_page_failure_is_skipped_not_raised(self):
        def _boom(*a, **k):
            raise RuntimeError("render failed")

        with patch("utils.image_utils.render_pdf_page_to_base64", side_effect=_boom):
            result = render_export_backgrounds("/fake/path.pdf", [1, 2])

        assert result == {}


class TestBuildLayoutPdfBytesBackgroundOverride:
    def test_page_backgrounds_override_used_instead_of_storage_fetch(self):
        page_w, page_h = 200.0, 300.0
        pages = [
            SimpleNamespace(page_number=1, image_width=int(page_w), image_height=int(page_h), image_key="stale-low-res-key")
        ]
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "hello",
                "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 40},
            }
        ]
        high_res_bytes = _blank_page_png_bytes()

        with patch("services.object_storage.get_object_storage") as mock_storage:
            pdf_bytes = build_layout_pdf_bytes(
                elements,
                pages,
                document_id="DOC_TEST",
                page_background=True,
                page_backgrounds={1: high_res_bytes},
            )
            # storage.get_bytes must never be called — the override takes precedence.
            mock_storage.return_value.get_bytes.assert_not_called()

        assert pdf_bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert doc.page_count == 1
        doc.close()

    def test_no_override_falls_back_to_storage_fetch(self):
        page_w, page_h = 200.0, 300.0
        pages = [
            SimpleNamespace(page_number=1, image_width=int(page_w), image_height=int(page_h), image_key="some-key")
        ]
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "hello",
                "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 40},
            }
        ]
        with patch("services.object_storage.get_object_storage") as mock_storage:
            mock_storage.return_value.get_bytes.return_value = _blank_page_png_bytes()
            build_layout_pdf_bytes(
                elements,
                pages,
                document_id="DOC_TEST",
                page_background=True,
            )
            mock_storage.return_value.get_bytes.assert_called_once_with("some-key")
