"""Tests for PDF overlay white-rect text masking."""

from __future__ import annotations

from core.pdf_overlay.converter import (
    Paragraph,
    cap_render_height_for_next_paragraph,
    gen_op_fill_rect,
    is_narrow_vertical_paragraph,
    next_paragraph_y_same_column,
    starts_new_vertical_band,
)


class TestGenOpFillRect:
    def test_emits_fill_rectangle_ops(self):
        ops = gen_op_fill_rect(10.0, 20.0, 110.0, 40.0)
        assert "1 1 1 rg" in ops
        assert "re f" in ops
        assert "100.000000" in ops or "100." in ops
        assert "20.000000" in ops or "20." in ops

    def test_zero_area_returns_empty(self):
        assert gen_op_fill_rect(10.0, 20.0, 10.0, 30.0) == ""
        assert gen_op_fill_rect(10.0, 20.0, 5.0, 30.0) == ""


class TestCapRenderHeightForNextParagraph:
    def test_no_next_paragraph_unbounded(self):
        assert cap_render_height_for_next_paragraph(100.0, y=500.0, next_y=None) == 100.0

    def test_overflow_capped_at_next_paragraph_start(self):
        """Regression: paragraph translated to more lines than fit in the
        original box must not paint its mask into the next paragraph's
        region — the confirmed ghosting root cause."""
        # Paragraph starts at y=500, natural height=20 (so naturally ends ~480).
        # Next paragraph starts at y=470 — only a 30pt gap available.
        # Translated text wants render_height=100 (way more lines) -- must be capped.
        capped = cap_render_height_for_next_paragraph(100.0, y=500.0, next_y=470.0, pad_y=1.0)
        assert capped < 100.0
        # Mask bottom (y - capped - pad_y) must not cross below next_y + pad_y.
        assert (500.0 - capped - 1.0) >= (470.0 + 1.0) - 1e-9

    def test_no_overflow_leaves_render_height_untouched(self):
        """When there's ample gap before the next paragraph, don't shrink
        below the paragraph's own natural render height."""
        capped = cap_render_height_for_next_paragraph(25.0, y=500.0, next_y=100.0, pad_y=1.0)
        assert capped == 25.0

    def test_caps_even_when_own_height_is_implausible(self):
        """Regression (DOC_073 p30): a paragraph whose char grouping glued in
        far-away text (running header + caption) reports a bogus natural
        height. The mask must still stop above the next paragraph instead of
        letting that bogus height bulldoze already-drawn text."""
        capped = cap_render_height_for_next_paragraph(446.2, y=200.2, next_y=157.9, pad_y=1.0)
        # Mask bottom must stay above the next paragraph's baseline.
        assert (200.2 - capped - 1.0) >= (157.9 + 1.0) - 1e-9

    def test_never_produces_negative_height(self):
        capped = cap_render_height_for_next_paragraph(50.0, y=500.0, next_y=499.0, pad_y=1.0)
        assert capped >= 0.0


class TestNextParagraphYSameColumn:
    def test_skips_other_column(self):
        """Two-column: next in reading order may be the other column —
        cap must use the next paragraph that shares x-overlap."""
        left = Paragraph(y=500, x=50, x0=50, x1=250, y0=480, y1=500, size=10, brk=False)
        right = Paragraph(y=490, x=320, x0=320, x1=520, y0=470, y1=490, size=10, brk=False)
        left_next = Paragraph(y=450, x=50, x0=50, x1=250, y0=430, y1=450, size=10, brk=False)
        assert next_paragraph_y_same_column([left, right, left_next], 0) == 450.0

    def test_none_when_no_same_column_below(self):
        left = Paragraph(y=500, x=50, x0=50, x1=250, y0=480, y1=500, size=10, brk=False)
        right = Paragraph(y=490, x=320, x0=320, x1=520, y0=470, y1=490, size=10, brk=False)
        assert next_paragraph_y_same_column([left, right], 0) is None

    def test_considers_paragraphs_earlier_in_reading_order(self):
        """Regression (DOC_073 p30): a figure caption is emitted *last* in the
        paragraph list but sits *above* the body paragraphs emitted before it.
        Scanning only forward found no neighbour, so its mask painted over the
        already-drawn body text (invisible-but-selectable text)."""
        body_a = Paragraph(y=157.9, x=51, x0=51, x1=420, y0=99.9, y1=167.9, size=10, brk=True)
        body_b = Paragraph(y=88.3, x=51, x0=51, x1=419.8, y0=53.5, y1=98.3, size=10, brk=True)
        caption = Paragraph(y=200.2, x=60, x0=51, x1=411.9, y0=179.8, y1=626.0, size=8.5, brk=True)
        paragraphs = [body_a, body_b, caption]
        assert next_paragraph_y_same_column(paragraphs, 2) == 157.9


class TestStartsNewVerticalBand:
    def test_ordinary_line_wrap_stays_in_paragraph(self):
        # Consecutive lines of a 10pt paragraph drop ~12pt.
        assert starts_new_vertical_band(500.0, 488.0, 10.0) is False

    def test_generous_paragraph_spacing_stays_in_paragraph(self):
        assert starts_new_vertical_band(500.0, 475.0, 10.0) is False

    def test_far_away_text_starts_new_paragraph(self):
        """Regression (DOC_073 p30): a running header at y=626 shared a layout
        class with the figure caption at y=200 and got glued into the same
        paragraph, producing a 446pt-tall paragraph box."""
        assert starts_new_vertical_band(200.2, 626.0, 8.5) is True

    def test_upward_jump_also_splits(self):
        assert starts_new_vertical_band(626.0, 200.2, 8.5) is True


class TestIsNarrowVerticalParagraph:
    def test_axis_label_is_narrow(self):
        para = Paragraph(y=300, x=70, x0=70, x1=78, y0=290, y1=300, size=9, brk=False)
        assert is_narrow_vertical_paragraph(para) is True

    def test_body_text_is_not_narrow(self):
        para = Paragraph(y=300, x=70, x0=70, x1=280, y0=290, y1=300, size=9, brk=False)
        assert is_narrow_vertical_paragraph(para) is False
