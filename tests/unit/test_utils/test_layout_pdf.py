"""Tests for layout-faithful PDF export."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from utils.layout_pdf import build_layout_pdf_bytes
from utils.table_grid import build_table_grid, table_text_to_cell_rows

PDF_PATH = Path(__file__).resolve().parents[3] / "2511.19575v2.pdf"


class TestTableGridShared:
    def test_html_table_parses(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        rows = table_text_to_cell_rows(html)
        assert rows is not None
        n_rows, n_cols, placements = build_table_grid(rows)
        assert n_rows == 2
        assert n_cols == 2
        assert len(placements) == 4


class TestLayoutPdfSynthetic:
    def test_two_column_page_is_single_pdf_page(self):
        page_w, page_h = 595.0, 842.0
        pages = [
            SimpleNamespace(
                page_number=1,
                image_width=int(page_w),
                image_height=int(page_h),
                image_key=None,
            )
        ]
        elements = [
            {
                "page_number": 1,
                "label": "title",
                "text_content": "Paper Title",
                "bbox": {"x1": 100, "y1": 40, "x2": 495, "y2": 70},
            },
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Left column body text.",
                "bbox": {"x1": 50, "y1": 100, "x2": 280, "y2": 300},
            },
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Right column body text.",
                "bbox": {"x1": 310, "y1": 100, "x2": 540, "y2": 300},
            },
            {
                "page_number": 1,
                "label": "figure",
                "text_content": "Figure 1: Chart.",
                "bbox": {"x1": 80, "y1": 320, "x2": 515, "y2": 520},
            },
        ]
        pdf_bytes = build_layout_pdf_bytes(elements, pages, page_background=False)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert len(doc) == 1
            text = doc[0].get_text()
            assert "Paper Title" in text
            assert "Left column" in text
            assert "Right column" in text
            assert "Figure 1" in text
        finally:
            doc.close()

    def test_two_source_pages_yield_two_pdf_pages(self):
        pages = [
            SimpleNamespace(page_number=1, image_width=595, image_height=842, image_key=None),
            SimpleNamespace(page_number=2, image_width=595, image_height=842, image_key=None),
        ]
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Page one",
                "bbox": {"x1": 50, "y1": 50, "x2": 500, "y2": 100},
            },
            {
                "page_number": 2,
                "label": "text",
                "text_content": "Page two",
                "bbox": {"x1": 50, "y1": 50, "x2": 500, "y2": 100},
            },
        ]
        pdf_bytes = build_layout_pdf_bytes(elements, pages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert len(doc) == 2
            assert "Page one" in doc[0].get_text()
            assert "Page two" in doc[1].get_text()
        finally:
            doc.close()


@pytest.mark.slow
@pytest.mark.skipif(not PDF_PATH.is_file(), reason="2511.19575v2.pdf not in repo root")
class TestLayoutPdfIntegration:
    def test_paper_extract_page_count(self):
        from services.extractors.docling_layout_extractor import DoclingLayoutExtractor

        ext = DoclingLayoutExtractor(str(PDF_PATH))
        ext.convert()
        total = ext.total_pages
        pages = [
            SimpleNamespace(
                page_number=pn,
                image_width=int(ext.page_size(pn)[0]),
                image_height=int(ext.page_size(pn)[1]),
                image_key=None,
            )
            for pn in range(1, total + 1)
        ]
        all_elements = []
        for pn in range(1, total + 1):
            for e in ext.extract_page(pn):
                all_elements.append(
                    {
                        "page_number": pn,
                        "label": e.element_type if e.element_type != "heading" else "title",
                        "text_content": e.text,
                        "bbox": e.bbox,
                        "crop_image_base64": e.image_bytes_b64,
                    }
                )

        pdf_bytes = build_layout_pdf_bytes(all_elements, pages, page_background=False)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert len(doc) == total
            p1 = doc[0].get_text()
            assert "HunyuanOCR" in p1
            assert "Figure 1" in p1 or "Abstract" in p1
            figures = sum(1 for e in all_elements if e["label"] == "figure")
            assert figures >= 30
        finally:
            doc.close()
