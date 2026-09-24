"""End-to-end hybrid renderer: layout, facsimile, two-column."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw

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


def _scan_jpeg_with_text(w=400, h=300, text="SOURCE TABLE TEXT", box=(40, 80, 360, 200)) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(80, 80, 80), width=2)
    draw.text((box[0] + 8, box[1] + 8), text, fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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


class TestScanTableTranslation:
    def test_translation_redraws_table_without_source_crop(self):
        """Scan translation must not paste the OCR table crop over inpainted cells."""
        import base64

        page_jpeg = _scan_jpeg_with_text()
        crop = _scan_jpeg_with_text(w=320, h=120, text="SOURCE TABLE TEXT", box=(4, 4, 316, 116))
        html = (
            "<table><tr><td>Flag</td><td>Translated cell</td></tr>"
            "<tr><td>--no-mmap</td><td>Load into RAM</td></tr></table>"
        )
        elements = [
            {
                "page_number": 1,
                "label": "table",
                "text_content": html,
                "bbox": {"x1": 40, "y1": 80, "x2": 360, "y2": 200},
                "crop_image_base64": base64.b64encode(crop).decode(),
            }
        ]
        result = render_document_pdf(
            pages=[_page(400, 300, page_type="scanned")],
            elements=elements,
            pdf_mode="layout",
            text_kind="translation",
            lang="en",
            page_backgrounds={1: page_jpeg},
        )
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            text = doc[0].get_text().replace("\xa0", " ")
            assert "Translated cell" in text or "Load into RAM" in text
            assert "SOURCE TABLE TEXT" not in text
        finally:
            doc.close()

    def test_translation_skips_table_crop_insert(self):
        import base64

        page_jpeg = _scan_jpeg_with_text()
        crop = _scan_jpeg_with_text(w=320, h=120, text="CROP SOURCE", box=(4, 4, 316, 116))
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        elements = [
            {
                "page_number": 1,
                "label": "table",
                "text_content": html,
                "bbox": {"x1": 40, "y1": 80, "x2": 360, "y2": 200},
                "crop_image_base64": base64.b64encode(crop).decode(),
            }
        ]
        with patch("core.pdf_render.renderer._insert_image") as mock_insert:
            render_document_pdf(
                pages=[_page(400, 300, page_type="scanned")],
                elements=elements,
                pdf_mode="layout",
                text_kind="translation",
                lang="en",
                page_backgrounds={1: page_jpeg},
            )
            # Page background uses insert_image on the fitz page directly;
            # table crop path goes through _insert_image helper — must not fire.
            assert mock_insert.call_count == 0


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


class TestInlineMathInLayoutPdf:
    def test_no_raw_dollar_latex_in_output(self):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Chu kỳ lấy mẫu $T_{sample} = T_S / N$ giây",
                "bbox": {"x1": 30, "y1": 50, "x2": 280, "y2": 120},
            }
        ]
        result = render_document_pdf(
            pages=[_page()],
            elements=elements,
            original_pdf_bytes=_source_pdf("Sampling period"),
            pdf_mode="layout",
            text_kind="translation",
            lang="vi",
        )
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            text = doc[0].get_text().replace("\xa0", " ")
            assert "$" not in text and "{" not in text
            assert "T_sample" in text
        finally:
            doc.close()


class TestOverflowKeepsPageCount:
    """Live regression (E2E): every exported PDF had more pages than its source
    (Digital Control 159 → 170 OCR / 178 translated), so page N of the export
    no longer matched page N of the book. Overflow now goes into a note on the
    same page instead of an appended continuation page."""

    def _render(self, mode):
        elements = [
            {
                "page_number": 1,
                "label": "text",
                "text_content": "Đoạn văn rất dài cần tràn khung. " * 200,
                "bbox": {"x1": 30, "y1": 50, "x2": 120, "y2": 70},
            }
        ]
        return render_document_pdf(
            pages=[_page()],
            elements=elements,
            original_pdf_bytes=_source_pdf("Short"),
            pdf_mode=mode,
            text_kind="translation",
            lang="vi",
        )

    def test_layout_overflow_does_not_add_pages(self):
        result = self._render("layout")
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            assert doc.page_count == 1
            notes = [a.info.get("content", "") for a in doc[0].annots()]
            assert any("tràn khung" in n for n in notes)
        finally:
            doc.close()
        assert result.continuation_pages == 0
        assert any(i["kind"] == "overflow" for i in result.quality.to_dict()["issues"])

    def test_facsimile_overflow_does_not_add_pages(self):
        result = self._render("facsimile")
        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        try:
            assert doc.page_count == 1
        finally:
            doc.close()
