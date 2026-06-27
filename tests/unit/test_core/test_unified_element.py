"""Tests for core UnifiedElement layout mapping."""

from core.models import UnifiedElement


def test_table_element_maps_to_table_label():
    elem = UnifiedElement(
        element_type="table",
        text="| a | b |",
        page_number=1,
        order=0,
        source="docx",
    )
    assert elem.to_layout_element_dict()["label"] == "table"


def test_image_element_carries_crop_image_for_spatial_export():
    elem = UnifiedElement(
        element_type="image",
        text="(img_content)[fig1]",
        page_number=2,
        order=3,
        source="docling",
        bbox={"x1": 10, "y1": 20, "x2": 110, "y2": 120},
        image_bytes_b64="abc123",
    )
    d = elem.to_layout_element_dict()
    assert d["label"] == "figure"
    assert d["crop_image"] == "abc123"
    assert d["bbox_x1"] == 10


def test_figure_element_maps_to_figure_label():
    elem = UnifiedElement(
        element_type="figure",
        text="Figure 1: Example chart.",
        page_number=1,
        order=0,
        source="docling_layout",
        bbox={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        image_bytes_b64="imgdata",
    )
    d = elem.to_layout_element_dict()
    assert d["label"] == "figure"
    assert d["crop_image"] == "imgdata"


def test_equation_element_maps_to_equation_label():
    elem = UnifiedElement(
        element_type="equation",
        text="$$E=mc^2$$",
        page_number=5,
        order=2,
        source="docling_layout",
    )
    assert elem.to_layout_element_dict()["label"] == "equation"
