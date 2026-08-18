"""End-to-end hybrid renderer: layout, facsimile, two-column."""

from types import SimpleNamespace

import fitz

from core.pdf_render.renderer import render_document_pdf


def _page(w=300, h=200, page_type="text"):
    return SimpleNamespace(
        page_number=1,
        image_width=w,
        image_height=h,
        image_key=None,
        page_type=page_type,
    )


def _source_pdf(text="Hello world") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 80), text, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


class TestLayoutRender:
    def test_replaces_source_text(self):
        src = _source_pdf("Hello world")
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Xin chao the gioi",
                "bbox": {"x1": 30, "y1": 50, "x2": 250, "y2": 100},
            }
        ]
        result = render_document_pdf(
            pages=[_page()],
            elements=elements,
            original_pdf_bytes=src,
            pdf_mode="layout",
            text_kind="translation",
            lang="vi",
        )
        assert result.pdf_bytes[:4] == b"%PDF"
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            text = doc[0].get_text().replace("\xa0", " ")
            assert "Xin chao" in text
            assert "Hello world" not in text
        finally:
            doc.close()

    def test_two_column_regions_stay_apart(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=400)
        page.insert_text((40, 80), "Left original", fontsize=11)
        page.insert_text((320, 80), "Right original", fontsize=11)
        src = doc.tobytes()
        doc.close()
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Cot trai da dich",
                "bbox": {"x1": 40, "y1": 60, "x2": 250, "y2": 120},
            },
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Cot phai da dich",
                "bbox": {"x1": 320, "y1": 60, "x2": 540, "y2": 120},
            },
        ]
        result = render_document_pdf(
            pages=[_page(595, 400)],
            elements=elements,
            original_pdf_bytes=src,
            pdf_mode="layout",
            text_kind="translation",
            lang="vi",
        )
        out = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            text = out[0].get_text().replace("\xa0", " ")
            assert "Cot trai" in text
            assert "Cot phai" in text
            overlaps = [
                i for i in result.quality.issues if i.kind in {"column_overlap", "block_overlap"}
            ]
            assert not overlaps
        finally:
            out.close()


class TestFacsimileRender:
    def test_invisible_text_is_searchable(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Searchable OCR line",
                "bbox": {"x1": 40, "y1": 40, "x2": 260, "y2": 70},
            }
        ]
        result = render_document_pdf(
            pages=[_page(page_type="scanned")],
            elements=elements,
            pdf_mode="facsimile",
            text_kind="ocr",
        )
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            assert "Searchable OCR" in doc[0].get_text().replace("\xa0", " ")
        finally:
            doc.close()
