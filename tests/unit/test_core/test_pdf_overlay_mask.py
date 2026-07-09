"""Tests for PDF overlay white-rect text masking."""

from __future__ import annotations

from core.pdf_overlay.converter import cap_render_height_for_next_paragraph, gen_op_fill_rect


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
        assert cap_render_height_for_next_paragraph(
            100.0, y=500.0, height=20.0, next_y=None
        ) == 100.0

    def test_overflow_capped_at_next_paragraph_start(self):
        """Regression: paragraph translated to more lines than fit in the
        original box must not paint its mask into the next paragraph's
        region — the confirmed ghosting root cause."""
        # Paragraph starts at y=500, natural height=20 (so naturally ends ~480).
        # Next paragraph starts at y=470 — only a 30pt gap available.
        # Translated text wants render_height=100 (way more lines) -- must be capped.
        capped = cap_render_height_for_next_paragraph(
            100.0, y=500.0, height=20.0, next_y=470.0, pad_y=1.0
        )
        assert capped < 100.0
        # Mask bottom (y - capped - pad_y) must not cross below next_y + pad_y.
        assert (500.0 - capped - 1.0) >= (470.0 + 1.0) - 1e-9

    def test_no_overflow_leaves_render_height_untouched(self):
        """When there's ample gap before the next paragraph, don't shrink
        below the paragraph's own natural render height."""
        capped = cap_render_height_for_next_paragraph(
            25.0, y=500.0, height=20.0, next_y=100.0, pad_y=1.0
        )
        assert capped == 25.0

    def test_never_shrinks_below_own_natural_height(self):
        """Even if the next paragraph is very close, don't cap below the
        paragraph's own pre-existing natural height — that would regress
        already-fine tightly-packed layouts."""
        capped = cap_render_height_for_next_paragraph(
            22.0, y=500.0, height=20.0, next_y=495.0, pad_y=1.0
        )
        assert capped == 22.0
