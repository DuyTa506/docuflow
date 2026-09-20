"""Export size: a PDF out must not dwarf the PDF in.

Measured on the 2026-09 E2E corpus: a 6.3 MB, 880-page book exported as a
28× larger 178 MB OCR PDF, and a 29.7 MB study guide came back as a 126 MB
translation. Both rasterised pages the source already carried — the OCR
facsimile re-rendered every page at 150 DPI, and the translation's scan
branch re-rendered at 2× with JPEG quality 95.
"""

from io import BytesIO
from types import SimpleNamespace

import fitz
from PIL import Image

from config.settings import settings
from core.pdf_render.renderer import render_document_pdf


def _page(w=300, h=200, page_type="text", number=1):
    return SimpleNamespace(
        page_number=number,
        image_width=w,
        image_height=h,
        image_key=None,
        page_type=page_type,
    )


def _text_pdf(pages=1, text="Hello world") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 80), f"{text} {i + 1}", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _scan_pdf(w=600, h=800) -> bytes:
    """A page whose only content is one embedded JPEG, like a real scan."""
    img = Image.effect_noise((w, h), 32).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=60)
    doc = fitz.open()
    page = doc.new_page(width=w / 2, height=h / 2)
    page.insert_image(page.rect, stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def _embedded_images(pdf_bytes: bytes, page_index: int = 0):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        return [doc.extract_image(x[0]) for x in page.get_images(full=True)]


class TestOcrFacsimileKeepsTheSourcePage:
    def test_a_native_page_is_copied_not_rasterised(self):
        src = _text_pdf()

        out = render_document_pdf(
            pages=[_page()],
            elements=[
                {
                    "page_number": 1,
                    "label": "text",
                    "text_content": "Hello world 1",
                    "bbox": {"x1": 30, "y1": 60, "x2": 260, "y2": 100},
                }
            ],
            original_pdf_bytes=src,
            pdf_mode="facsimile",
            text_kind="ocr",
        ).pdf_bytes

        # No page-sized raster: the original text stays vector text.
        assert _embedded_images(out) == []
        with fitz.open(stream=out, filetype="pdf") as doc:
            assert doc.page_count == 1
            assert "Hello world 1" in doc[0].get_text("text")

    def test_a_scanned_page_reuses_the_stored_image_bytes(self):
        src = _scan_pdf()
        source_image = _embedded_images(src)[0]

        out = render_document_pdf(
            pages=[_page(w=300, h=400, page_type="scanned")],
            elements=[
                {
                    "page_number": 1,
                    "label": "text",
                    "text_content": "OCR text",
                    "bbox": {"x1": 20, "y1": 20, "x2": 280, "y2": 60},
                }
            ],
            original_pdf_bytes=src,
            pdf_mode="facsimile",
            text_kind="ocr",
        ).pdf_bytes

        images = _embedded_images(out)
        assert len(images) == 1
        # Byte-identical: the scan was copied, not decoded and re-encoded.
        assert images[0]["image"] == source_image["image"]
        # …and the OCR text is still searchable on top of it.
        with fitz.open(stream=out, filetype="pdf") as doc:
            assert "OCR text" in doc[0].get_text("text")

    def test_the_export_stays_close_to_the_source_size(self):
        src = _text_pdf(pages=5)

        out = render_document_pdf(
            pages=[_page(number=i + 1) for i in range(5)],
            elements=[],
            original_pdf_bytes=src,
            pdf_mode="facsimile",
            text_kind="ocr",
        ).pdf_bytes

        assert len(out) < len(src) * 2

    def test_without_a_source_pdf_the_stored_page_image_is_used(self):
        buf = BytesIO()
        Image.new("RGB", (300, 200), "white").save(buf, format="JPEG")

        out = render_document_pdf(
            pages=[_page()],
            elements=[],
            pdf_mode="facsimile",
            text_kind="ocr",
            page_backgrounds={1: buf.getvalue()},
        ).pdf_bytes

        assert len(_embedded_images(out)) == 1


class TestTranslationScanBackground:
    def test_the_scan_is_re_rendered_at_the_configured_dpi(self, monkeypatch):
        monkeypatch.setattr(settings, "layout_pdf_export_dpi", 100)
        src = _scan_pdf(w=1200, h=1600)  # 600x800 pt page

        out = render_document_pdf(
            pages=[_page(w=600, h=800, page_type="scanned")],
            elements=[
                {
                    "page_number": 1,
                    "label": "text",
                    "text_content": "Bản dịch",
                    "bbox": {"x1": 40, "y1": 40, "x2": 560, "y2": 120},
                }
            ],
            original_pdf_bytes=src,
            pdf_mode="layout",
            text_kind="translation",
            lang="vi",
        ).pdf_bytes

        [image] = _embedded_images(out)
        # 600 pt at 100 DPI = 833 px, not the 2× (1200 px) it used to render.
        assert 700 <= image["width"] <= 900


class TestFigureCropsAreStoredAsJpeg:
    """Crops arrive as PNG from part of the extraction path.

    On the 880-page book in the E2E corpus that was 284 RGB PNGs, 86 KB each
    — 23 MB of the 61 MB translation export, for photographs of book pages
    that JPEG stores in a fraction of the space.
    """

    @staticmethod
    def _png_crop(w=400, h=300) -> str:
        import base64

        buf = BytesIO()
        Image.effect_noise((w, h), 48).convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _render(self, crop_b64: str) -> bytes:
        return render_document_pdf(
            pages=[_page(w=600, h=800, page_type="scanned")],
            elements=[
                {
                    "page_number": 1,
                    "label": "image",
                    "text_content": "",
                    "bbox": {"x1": 20, "y1": 20, "x2": 420, "y2": 320},
                    "crop_image_base64": crop_b64,
                }
            ],
            original_pdf_bytes=_scan_pdf(),
            pdf_mode="layout",
            text_kind="translation",
        ).pdf_bytes

    def test_a_png_crop_is_embedded_as_jpeg(self):
        out = self._render(self._png_crop())

        crops = [i for i in _embedded_images(out) if i["width"] == 400 and i["height"] == 300]
        assert crops, "the figure crop is missing from the page"
        assert crops[0]["ext"] == "jpeg"

    def test_the_crop_no_longer_dominates_the_page(self):
        import base64

        crop = self._png_crop()
        png_bytes = len(base64.b64decode(crop))

        out = self._render(crop)

        crops = [i for i in _embedded_images(out) if i["width"] == 400 and i["height"] == 300]
        assert len(crops[0]["image"]) < png_bytes / 2

    def test_a_crop_that_jpeg_cannot_beat_is_left_alone(self):
        import base64

        from PIL import ImageDraw

        # A line drawing: flat areas and hard edges, exactly what PNG stores
        # better than JPEG.
        art = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(art)
        for x in range(20, 380, 30):
            draw.line((x, 20, x, 280), fill="black", width=2)
        buf = BytesIO()
        art.save(buf, format="PNG", optimize=True)
        line_art = base64.b64encode(buf.getvalue()).decode()

        out = self._render(line_art)

        crops = [i for i in _embedded_images(out) if i["width"] == 400 and i["height"] == 300]
        assert crops[0]["ext"] == "png"
