"""Font-metric wrapping and fitting."""

from core.pdf_render.fonts import fitz_font
from core.pdf_render.geometry import Rect
from core.pdf_render.text_layout import (
    MIN_FONT_PT,
    TABLE_MIN_FONT_PT,
    expand_rect_in_column,
    fit_textbox,
    wrap_words,
)


def test_wrap_words_stays_within_width():
    font = fitz_font("en")
    lines = wrap_words("one two three four five six", font, 11, 60)
    assert len(lines) >= 2
    for line in lines:
        assert font.text_length(line, fontsize=11) <= 61


def test_min_font_floor_is_readable():
    assert MIN_FONT_PT >= 7.0
    assert TABLE_MIN_FONT_PT >= 6.5


def test_fit_overflow_at_min_font():
    font = fitz_font("en")
    rect = Rect(0, 0, 40, 16)
    fitted = fit_textbox(
        "This is a very long paragraph that cannot possibly fit in such a tiny box.",
        rect,
        font,
        min_pt=MIN_FONT_PT,
        max_pt=12,
    )
    assert fitted.fontsize <= MIN_FONT_PT + 0.05
    assert fitted.overflow


def test_expand_rect_stops_at_neighbor():
    rect = Rect(10, 10, 200, 30)
    neighbor = Rect(10, 50, 200, 80)
    expanded = expand_rect_in_column(
        rect, needed_height=80, page_h=200, neighbors=[neighbor], gap=4
    )
    assert expanded.y1 <= 46
