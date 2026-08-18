"""Native redaction keeps images/vectors; scan inpaint is best-effort."""

from io import BytesIO

import fitz
from PIL import Image

from core.pdf_render.cleaner import inpaint_scan_image, redact_native_text
from core.pdf_render.geometry import Rect


def _text_pdf(text="Hello world") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 80), text, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


class TestRedactNativeText:
    def test_removes_glyphs_keeps_page(self):
        doc = fitz.open(stream=_text_pdf(), filetype="pdf")
        page = doc[0]
        assert "Hello" in page.get_text()
        n = redact_native_text(page, [Rect(30, 50, 200, 100)], [])
        assert n >= 1
        leftover = page.get_text()
        assert "Hello" not in leftover
        doc.close()

    def test_keeps_embedded_image(self):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (40, 40), "red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        page.insert_image(fitz.Rect(200, 20, 280, 100), stream=buf.getvalue())
        page.insert_text((40, 80), "Hello world", fontsize=14)
        assert page.get_images()
        redact_native_text(page, [Rect(30, 50, 180, 100)], [])
        assert "Hello" not in page.get_text()
        assert page.get_images()
        doc.close()

    def test_skips_reserved_spans(self):
        doc = fitz.open(stream=_text_pdf("Keep me"), filetype="pdf")
        page = doc[0]
        reserved = [Rect(0, 0, 300, 200)]
        n = redact_native_text(page, [Rect(0, 0, 300, 200)], reserved)
        assert n == 0
        assert "Keep" in page.get_text()
        doc.close()


class TestScanInpaint:
    def test_returns_jpeg_bytes(self):
        img = Image.new("RGB", (200, 100), "white")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        out = inpaint_scan_image(
            buf.getvalue(),
            [Rect(10, 10, 80, 40)],
            [],
            page_w=200,
            page_h=100,
        )
        assert out is not None
        assert out[:2] == b"\xff\xd8"
