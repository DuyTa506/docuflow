"""Tests for layout-element block merging before translation."""

from utils.translation_blocks import (
    TranslationBlock,
    is_passthrough_label,
    merge_payloads_to_blocks,
)


def _payload(page, order, label, text, x1=10, y1=20, x2=100, y2=40):
    return {
        "page_number": page,
        "sequence_order": order,
        "label": label,
        "text_content": text,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


class TestPassthroughLabels:
    def test_image_is_passthrough(self):
        assert is_passthrough_label("image")
        assert is_passthrough_label("table")
        assert not is_passthrough_label("text")


class TestMergePayloads:
    def test_image_not_merged_with_text(self):
        payloads = [
            _payload(1, 0, "text", "Hello"),
            _payload(1, 1, "image", "(img)", x1=50, y1=100, x2=150, y2=200),
            _payload(1, 2, "text", "World", y1=250, y2=270),
        ]
        blocks = merge_payloads_to_blocks(payloads)
        assert len(blocks) == 3
        assert blocks[0].text == "Hello"
        assert blocks[1].passthrough and blocks[1].label == "image"
        assert blocks[2].text == "World"

    def test_heading_becomes_separate_block(self):
        payloads = [
            _payload(1, 0, "title", "Introduction", y1=10, y2=30),
            _payload(1, 1, "text", "Body one", y1=40, y2=60),
            _payload(1, 2, "text", "Body two", y1=70, y2=90),
        ]
        blocks = merge_payloads_to_blocks(payloads)
        assert any(b.is_heading for b in blocks)
        heading = next(b for b in blocks if b.is_heading)
        assert heading.text == "Introduction"

    def test_adjacent_textlines_merge_on_same_page(self):
        payloads = [
            _payload(1, 0, "text", "Line one", y1=10, y2=25),
            _payload(1, 1, "text", "Line two", y1=26, y2=41),
        ]
        blocks = merge_payloads_to_blocks(payloads)
        text_blocks = [b for b in blocks if not b.passthrough and not b.is_heading]
        assert len(text_blocks) == 1
        assert "Line one" in text_blocks[0].text
        assert "Line two" in text_blocks[0].text

    def test_table_passthrough(self):
        payloads = [
            _payload(1, 0, "table", "<table><tr><td>A</td></tr></table>"),
        ]
        blocks = merge_payloads_to_blocks(payloads)
        assert len(blocks) == 1
        assert blocks[0].passthrough
        assert blocks[0].label == "table"
