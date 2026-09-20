"""Scan page backgrounds: colour only where the page actually has colour.

Translation exports of scanned books are one JPEG per page, so the encoding
decides the file size: 488 scanned pages came to 50 MB. Book scans are
black-on-white, and storing three identical channels per pixel pays for
colour nothing in the page uses.
"""

from io import BytesIO

from PIL import Image, ImageDraw

from utils.image_utils import encode_scan_jpeg


def _grey_page(w=600, h=800) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, h - 40, 24):
        draw.rectangle((40, y, w - 40, y + 10), fill=(30, 30, 30))
    return img


def _colour_page(w=600, h=800) -> Image.Image:
    img = _grey_page(w, h)
    ImageDraw.Draw(img).rectangle((60, 60, 400, 300), fill=(220, 40, 40))
    return img


class TestEncodeScanJpeg:
    def test_a_black_and_white_scan_is_stored_as_grayscale(self):
        data = encode_scan_jpeg(_grey_page(), quality=85)

        with Image.open(BytesIO(data)) as out:
            assert out.mode == "L"

    def test_grayscale_is_smaller_than_the_rgb_encoding(self):
        page = _grey_page()
        buf = BytesIO()
        page.save(buf, format="JPEG", quality=85)

        assert len(encode_scan_jpeg(page, quality=85)) < len(buf.getvalue())

    def test_a_page_with_real_colour_keeps_it(self):
        data = encode_scan_jpeg(_colour_page(), quality=85)

        with Image.open(BytesIO(data)) as out:
            assert out.mode == "RGB"
            # The red block survives the round trip.
            r, g, b = out.convert("RGB").getpixel((200, 200))
            assert r > 150 and g < 100 and b < 100

    def test_jpeg_bytes_are_returned_for_a_grayscale_input(self):
        data = encode_scan_jpeg(_grey_page().convert("L"), quality=85)

        assert data[:2] == b"\xff\xd8"
        with Image.open(BytesIO(data)) as out:
            assert out.mode == "L"
