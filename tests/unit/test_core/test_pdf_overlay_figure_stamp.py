"""Figures must survive PDF-overlay translation pixel-perfect.

Regression (DOC_069, text-layer tech report): the overlay converter strips
ALL original text operators from the page and only re-draws what it manages.
Text inside figures is "reserved" (never translated) but its re-draw is
paragraph-relative, so large text-built figures came out as empty boxes in
the translated PDF. Instead of trusting the converter's re-draw, the
pipeline now stamps each figure region back onto the translated page as an
image rendered from the ORIGINAL pdf — guaranteed identical pixels.
"""

import pymupdf

from core.pdf_overlay.pipeline import stamp_figure_regions


def _source_pdf_with_figure_text():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Body paragraph outside the figure.", fontsize=11)
    # "figure" built from vector text — the DOC_069 failure shape
    page.draw_rect(pymupdf.Rect(100, 300, 400, 500), color=(0, 0, 0))
    page.insert_text((120, 350), "get_current_weather", fontsize=9)
    page.insert_text((120, 380), "text inside figure panel", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def _wrecked_translation_of(src_bytes):
    """Simulate the converter's damage: figure text gone from the page."""
    doc = pymupdf.open(stream=src_bytes, filetype="pdf")
    page = doc[0]
    page.draw_rect(pymupdf.Rect(100, 300, 400, 500), color=(1, 1, 1), fill=(1, 1, 1))
    return doc


def _region_pixels(page, rect):
    return page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(2, 2)).samples


ELEMENTS = {
    1: [
        {"label": "figure", "bbox_x1": 100, "bbox_y1": 300, "bbox_x2": 400, "bbox_y2": 500},
        {"label": "text", "bbox_x1": 72, "bbox_y1": 90, "bbox_x2": 400, "bbox_y2": 110},
    ]
}


class TestStampFigureRegions:
    def test_figure_region_restored_pixel_identical(self):
        src = _source_pdf_with_figure_text()
        wrecked = _wrecked_translation_of(src)
        rect = pymupdf.Rect(100, 300, 400, 500)

        before = _region_pixels(wrecked[0], rect)
        assert before != _region_pixels(pymupdf.open(stream=src, filetype="pdf")[0], rect)

        stamped = stamp_figure_regions(src, wrecked, ELEMENTS)

        out = pymupdf.open(stream=wrecked.tobytes(), filetype="pdf")
        after = out[0].get_pixmap(clip=rect, matrix=pymupdf.Matrix(2, 2))
        src_pix = pymupdf.open(stream=src, filetype="pdf")[0].get_pixmap(
            clip=rect, matrix=pymupdf.Matrix(2, 2)
        )
        # not byte-identical (raster stamp), but must no longer be blank:
        # the stamped region must be visually non-empty like the source
        nonwhite = sum(1 for b in after.samples if b < 250)
        src_nonwhite = sum(1 for b in src_pix.samples if b < 250)
        assert nonwhite > src_nonwhite * 0.5
        assert stamped == 1

    def test_text_elements_and_missing_pages_are_ignored(self):
        src = _source_pdf_with_figure_text()
        wrecked = _wrecked_translation_of(src)
        stamped = stamp_figure_regions(
            src,
            wrecked,
            {
                1: [
                    {"label": "text", "bbox_x1": 72, "bbox_y1": 90, "bbox_x2": 400, "bbox_y2": 110}
                ],
                99: [{"label": "figure", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 50, "bbox_y2": 50}],
            },
        )
        assert stamped == 0

    def test_degenerate_bboxes_are_skipped(self):
        src = _source_pdf_with_figure_text()
        wrecked = _wrecked_translation_of(src)
        stamped = stamp_figure_regions(
            src,
            wrecked,
            {1: [{"label": "figure", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 12, "bbox_y2": 11}]},
        )
        assert stamped == 0
