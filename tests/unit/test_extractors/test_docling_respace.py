"""Docling glued letter-spaced headings: «1.2 WHYDIGITALCONTROL», «2 DIGITAL
CONTROLINPOWERELECTRONICS» — 463 elements across the E2E books. PyMuPDF reads
the same region with its spaces, so the text is re-read when it looks glued."""

import fitz
import pytest

from services.extractors.docling_layout_extractor import DoclingLayoutExtractor, respace_text


@pytest.mark.parametrize(
    "text, words, expected",
    [
        ("1.2 WHYDIGITALCONTROL", ["1.2", "WHY", "DIGITAL", "CONTROL"], "1.2 WHY DIGITAL CONTROL"),
        (
            "INTRODUCTION: DIGITALCONTROL APPLICATION",
            ["INTRODUCTION:", "DIGITAL", "CONTROL", "APPLICATION"],
            "INTRODUCTION: DIGITAL CONTROL APPLICATION",
        ),
        # Different letters → not the same text, keep Docling's.
        ("1.2 WHYDIGITALCONTROL", ["Figure", "3"], "1.2 WHYDIGITALCONTROL"),
        # Nothing glued → untouched even when words are available.
        ("Normal heading", ["Normal", "heading"], "Normal heading"),
    ],
)
def test_respace_text(text, words, expected):
    assert respace_text(text, words) == expected


def test_extractor_rereads_glued_text_from_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 80), "WHY DIGITAL CONTROL", fontsize=12)
    doc.save(path)
    doc.close()

    extractor = DoclingLayoutExtractor(str(path))
    bbox = {"x1": 30, "y1": 60, "x2": 290, "y2": 90}
    assert extractor._respaced("WHYDIGITALCONTROL", 1, bbox) == "WHY DIGITAL CONTROL"
