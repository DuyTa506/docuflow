"""Tests for translation PDF overlay (mask + replace mode)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fitz
import pytest

from utils.layout_pdf import build_layout_pdf_bytes


def _page(w=595, h=842):
    return SimpleNamespace(page_number=1, image_width=w, image_height=h, image_key=None)


class TestLayoutPdfReplaceMode:
    def test_replace_mode_draws_mask_before_textbox(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Translated paragraph.",
                "bbox": {"x1": 50, "y1": 100, "x2": 400, "y2": 130},
            }
        ]
        with patch.object(fitz.Page, "draw_rect", wraps=lambda self, *a, **k: None) as mock_draw:
            with patch.object(fitz.Page, "insert_textbox", return_value=1) as mock_tb:
                build_layout_pdf_bytes(
                    elements,
                    [_page()],
                    page_background=False,
                    text_overlay="replace",
                )
                assert mock_draw.called
                assert mock_tb.called

    def test_long_text_replace_expands_rect(self):
        short = "Short."
        long_vn = (
            "Tài liệu bổ sung này cung cấp các chi tiết bổ sung cho báo cáo kỹ thuật "
            "với nội dung dài hơn nhiều so với vùng bbox gốc."
        )
        bbox = {"x1": 50, "y1": 100, "x2": 250, "y2": 120}
        pages = [_page()]

        pdf_short = build_layout_pdf_bytes(
            [{"page_number": 1, "label": "text", "text_content": short, "bbox": bbox}],
            pages,
            page_background=False,
            text_overlay="skip",
        )
        pdf_long_skip = build_layout_pdf_bytes(
            [{"page_number": 1, "label": "text", "text_content": long_vn, "bbox": bbox}],
            pages,
            page_background=False,
            text_overlay="skip",
        )
        pdf_long_replace = build_layout_pdf_bytes(
            [{"page_number": 1, "label": "text", "text_content": long_vn, "bbox": bbox}],
            pages,
            page_background=False,
            text_overlay="replace",
        )

        def _len_text(data: bytes) -> int:
            doc = fitz.open(stream=data, filetype="pdf")
            try:
                return len(doc[0].get_text())
            finally:
                doc.close()

        assert _len_text(pdf_long_replace) >= _len_text(pdf_long_skip)
        assert _len_text(pdf_long_replace) > _len_text(pdf_short)

    def test_skip_mode_omits_body_on_background(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Should not appear in text layer.",
                "bbox": {"x1": 50, "y1": 100, "x2": 400, "y2": 130},
            }
        ]
        fake_bg = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        with patch("services.object_storage.get_object_storage") as mock_storage:
            mock_storage.return_value.get_bytes.return_value = fake_bg
            pdf_bytes = build_layout_pdf_bytes(
                elements,
                [SimpleNamespace(page_number=1, image_width=595, image_height=842, image_key="pages/1.jpg")],
                page_background=True,
                text_overlay="skip",
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert "Should not appear" not in doc[0].get_text()
        finally:
            doc.close()

    def test_replace_mode_draws_body_on_background(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Translated visible text.",
                "bbox": {"x1": 50, "y1": 100, "x2": 400, "y2": 130},
            }
        ]
        fake_bg = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        with patch("services.object_storage.get_object_storage") as mock_storage:
            mock_storage.return_value.get_bytes.return_value = fake_bg
            pdf_bytes = build_layout_pdf_bytes(
                elements,
                [SimpleNamespace(page_number=1, image_width=595, image_height=842, image_key="pages/1.jpg")],
                page_background=True,
                text_overlay="replace",
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert "Translated visible" in doc[0].get_text()
        finally:
            doc.close()

    def test_replace_skips_text_inside_figure_bbox(self):
        """Text whose bbox sits inside a figure must not be redrawn on top."""
        elements = [
            {
                "page_number": 1,
                "label": "figure",
                "text_content": "",
                "bbox": {"x1": 50, "y1": 100, "x2": 400, "y2": 400},
            },
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Axis label inside figure",
                "bbox": {"x1": 60, "y1": 150, "x2": 120, "y2": 350},
            },
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Caption outside figure",
                "bbox": {"x1": 50, "y1": 420, "x2": 400, "y2": 450},
            },
        ]
        pdf_bytes = build_layout_pdf_bytes(
            elements,
            [_page()],
            page_background=False,
            text_overlay="replace",
        )
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = doc[0].get_text()
            assert "Axis label inside figure" not in text
            assert "Caption outside figure" in text
        finally:
            doc.close()

    def test_narrow_textbox_skipped_in_replace(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Stacked axis characters",
                "bbox": {"x1": 50, "y1": 100, "x2": 58, "y2": 300},
            }
        ]
        pdf_bytes = build_layout_pdf_bytes(
            elements,
            [_page()],
            page_background=False,
            text_overlay="replace",
        )
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert "Stacked axis" not in doc[0].get_text()
        finally:
            doc.close()
