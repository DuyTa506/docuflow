"""Block merge for OCR layout PDF export."""

from types import SimpleNamespace

from utils.file_download import build_pdf_bytes_from_elements
from utils.translation_blocks import merge_elements_for_layout_export


def _line_payload(page: int, seq: int, y: int, text: str) -> dict:
    return {
        "page_number": page,
        "label": "text",
        "text_content": text,
        "sequence_order": seq,
        "bbox": {"x1": 50, "y1": y, "x2": 500, "y2": y + 20},
    }


class TestMergeElementsForLayoutExport:
    def test_collapses_adjacent_lines_into_one_block(self):
        payloads = [
            _line_payload(1, 0, 100, "Line one."),
            _line_payload(1, 1, 125, "Line two."),
            _line_payload(1, 2, 150, "Line three."),
        ]
        merged = merge_elements_for_layout_export(payloads)
        assert len(merged) == 1
        assert "Line one." in merged[0]["text_content"]
        assert "Line three." in merged[0]["text_content"]
        assert merged[0]["bbox"]["y1"] == 100
        assert merged[0]["bbox"]["y2"] >= 150

    def test_passthrough_table_stays_separate(self):
        payloads = [
            _line_payload(1, 0, 100, "Before table."),
            {
                "page_number": 1,
                "label": "table",
                "text_content": "<table><tr><td>A</td></tr></table>",
                "sequence_order": 1,
                "bbox": {"x1": 50, "y1": 200, "x2": 400, "y2": 350},
                "crop_image_key": "documents/x/crops/0001_0002.jpg",
            },
        ]
        merged = merge_elements_for_layout_export(payloads)
        assert len(merged) == 2
        assert merged[1]["label"] == "table"
        assert merged[1].get("crop_image_key")

    def test_build_pdf_with_merge_blocks_flag(self):
        pages = [SimpleNamespace(page_number=1, image_width=595, image_height=842, image_key=None)]
        orm_elems = [
            SimpleNamespace(
                label="text",
                text_content="Alpha",
                sequence_order=0,
                bbox_x1=50,
                bbox_y1=100,
                bbox_x2=200,
                bbox_y2=120,
                page=SimpleNamespace(page_number=1),
            ),
            SimpleNamespace(
                label="text",
                text_content="Beta",
                sequence_order=1,
                bbox_x1=50,
                bbox_y1=130,
                bbox_x2=200,
                bbox_y2=150,
                page=SimpleNamespace(page_number=1),
            ),
        ]
        pdf_bytes = build_pdf_bytes_from_elements(
            orm_elems, pages, merge_blocks=True, page_background=False, text_overlay="replace"
        )
        assert pdf_bytes[:4] == b"%PDF"
