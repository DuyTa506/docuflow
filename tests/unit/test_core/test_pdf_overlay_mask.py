"""Tests for PDF overlay white-rect text masking."""

from __future__ import annotations

from core.pdf_overlay.converter import gen_op_fill_rect


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
