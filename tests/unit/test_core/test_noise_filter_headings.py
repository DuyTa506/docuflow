"""Regression: the area-based noise filter must not delete section headings.

A one-line heading in a two-column article ("KẾT LUẬN", "1. 引言") covers
about 0.1% of the page, which is where `min_element_area_ratio` cuts. On the
52-document ND4 set the filter removed 593 of 1,334 heading elements — every
heading of the classical Chinese books — so their trees had a few giant nodes
and digest §2.2 found one or two units in a five-section paper.
"""

import pytest

from core.spatial import build_spatial_tree
from core.spatial.filters import apply_all_filters, filter_noise_elements

PAGE = {"width": 1000, "height": 1400}


def _elem(label, text, x1, y1, x2, y2, page=1):
    return {
        "label": label,
        "text_content": text,
        "text_full": text,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "page_number": page,
    }


def _small_heading(label="title", text="KẾT LUẬN"):
    # 90 × 14 px on a 1000 × 1400 page = 0.09% of the page.
    return _elem(label, text, 80, 600, 170, 614)


class TestSmallHeadingsSurvive:
    @pytest.mark.parametrize("label", ["title", "sub_title", "heading", "section_heading"])
    def test_a_small_heading_with_text_is_kept(self, label):
        kept, removed = filter_noise_elements([_small_heading(label)], page_dims=PAGE)
        assert [e["text_content"] for e in kept] == ["KẾT LUẬN"]
        assert removed == []

    def test_a_small_speck_of_body_text_is_still_removed(self):
        speck = _elem("text", "·", 80, 600, 90, 610)
        kept, removed = filter_noise_elements([speck], page_dims=PAGE)
        assert kept == []
        assert removed[0]["filter_reason"] == "too_small"

    def test_an_empty_heading_box_is_still_removed(self):
        kept, removed = filter_noise_elements([_small_heading(text="  ")], page_dims=PAGE)
        assert kept == []
        assert removed[0]["filter_reason"] == "too_small"

    def test_an_oversized_heading_is_still_removed(self):
        backdrop = _elem("title", "COVER", 0, 0, 1000, 1400)
        kept, removed = filter_noise_elements([backdrop], page_dims=PAGE)
        assert kept == []
        assert removed[0]["filter_reason"] == "too_large"


def test_the_heading_reaches_the_tree():
    body = "Nội dung của phần kết luận được trình bày ở đây. " * 20
    elements = [
        _elem("title", "PHƯƠNG PHÁP NGHIÊN CỨU", 80, 100, 400, 124),
        _elem("text", body, 80, 140, 920, 580),
        _small_heading(),
        _elem("text", body, 80, 630, 920, 1300),
        # A full-size element fixes the page dimensions the filter estimates.
        _elem("text", "chân trang", 0, 1380, 1000, 1400),
    ]
    kept, _ = apply_all_filters([dict(e) for e in elements])
    assert "KẾT LUẬN" in [e["text_content"] for e in kept]

    tree = build_spatial_tree([dict(e) for e in elements], use_thinning=True)

    def titles(node):
        yield node.get("title") or ""
        for child in node.get("children", []):
            yield from titles(child)

    assert "KẾT LUẬN" in list(titles(tree))
