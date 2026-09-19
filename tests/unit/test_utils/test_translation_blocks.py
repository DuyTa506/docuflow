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

    def test_two_columns_stay_separate_blocks(self):
        payloads = [
            _payload(1, 0, "text", "Left A", x1=40, y1=80, x2=250, y2=200),
            _payload(1, 1, "text", "Right A", x1=320, y1=80, x2=540, y2=200),
            _payload(1, 2, "text", "Left B", x1=40, y1=220, x2=250, y2=360),
            _payload(1, 3, "text", "Right B", x1=320, y1=220, x2=540, y2=360),
        ]
        blocks = merge_payloads_to_blocks(payloads)
        texts = {b.text for b in blocks}
        assert any("Left A" in t and "Right A" not in t for t in texts)
        assert any("Right A" in t and "Left A" not in t for t in texts)


class TestReadingOrderRuDesigningPage121:
    """Live regression (E2E Q4, Ru_Designing p.121): OCR order was right, but
    blocks were re-sorted by y across both columns, so right-column «б) вид 1…»
    came before the left column's heading and steps printed 2, 3, 1, 4."""

    ROWS = [
        (0, "sub_title", 166, 252, 313, 270, "4.3.4. Ввод-вывод"),
        (1, "sub_title", 189, 277, 413, 295, "Программируемый ввод-вывод"),
        (2, "text", 165, 302, 465, 422, "В микропроцессоре Z80 реализован режим ввода-вывода."),
        (3, "text", 166, 422, 464, 456, "1. Адрес порта помещается в младшие разряды."),
        (4, "text", 165, 456, 464, 523, "2. На выходах IORQ и RD устанавливается низкий уровень."),
        (5, "text", 165, 523, 465, 609, "3. В целях предоставления устройству времени."),
        (6, "text", 166, 612, 464, 647, "4. ЦП принимает 8-разрядное данное с шины данных."),
        (7, "text", 190, 649, 407, 667, "Прерывание по вводу-выводу"),
        (8, "text", 166, 671, 465, 706, "Процессор Z80 предусматривает несколько режимов:"),
        (9, "text", 166, 706, 465, 791, "1. Немаксируемое прерывание."),
        (10, "text", 166, 791, 465, 843, "2. Максируемое прерывание."),
        (11, "text", 475, 252, 779, 390, "а) вид 0: прерывающее устройство помещает команду."),
        (12, "text", 476, 390, 779, 458, "б) вид 1: этот вид прерывания инициирует рестарт."),
        (13, "text", 476, 458, 780, 667, "в) вид 2: прерывающее устройство посылает байт."),
        (14, "sub_title", 479, 680, 568, 699, "Пример 4.6"),
    ]

    def _blocks(self):
        payloads = [
            _payload(121, o, lab, t, x1=x1, y1=y1, x2=x2, y2=y2)
            for o, lab, x1, y1, x2, y2, t in self.ROWS
        ]
        return merge_payloads_to_blocks(payloads)

    def test_blocks_follow_ocr_reading_order(self):
        joined = "\n".join(b.text for b in self._blocks())
        positions = [joined.index(row[6]) for row in self.ROWS]
        assert positions == sorted(positions)

    def test_list_items_are_not_glued_into_one_block(self):
        texts = [b.text for b in self._blocks()]
        steps = [t for t in texts if t[:3] in ("1. ", "2. ", "3. ", "4. ")]
        assert len(steps) == 6
