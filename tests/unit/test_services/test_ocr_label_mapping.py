"""OCR label → UnifiedElement mapping."""

from services.extractors.ocr_extractor import ocr_elements_to_unified


class TestOcrLabelMapping:
    def test_equation_labels_map_to_equation_type(self):
        for label in ("equation", "formula", "isolate_formula", "math"):
            elems = ocr_elements_to_unified(
                [{"label": label, "text_full": "E = mc^2", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10}],
                page_number=1,
            )
            assert len(elems) == 1
            assert elems[0].element_type == "equation"

    def test_unknown_label_defaults_to_text(self):
        elems = ocr_elements_to_unified(
            [{"label": "sidebar", "text_full": "note", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10}],
            page_number=1,
        )
        assert elems[0].element_type == "text"
