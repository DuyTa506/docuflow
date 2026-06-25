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
