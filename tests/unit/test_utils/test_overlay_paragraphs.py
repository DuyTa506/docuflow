"""Tests for PDF overlay paragraph merging."""

from utils.overlay_paragraphs import merge_overlay_paragraphs


class _P:
    def __init__(self, y0, y1, size=12, brk=False, x0=0, x1=100):
        self.y0, self.y1, self.size, self.brk, self.x0, self.x1 = y0, y1, size, brk, x0, x1


def test_merge_adjacent_short_paragraphs():
    texts = ["Hello", "world", "Next block"]
    meta = [_P(0, 10), _P(11, 20), _P(50, 60)]
    merged_t, merged_m = merge_overlay_paragraphs(texts, meta, max_chars=800)
    assert len(merged_t) == 2
    assert "Hello" in merged_t[0] and "world" in merged_t[0]
    assert merged_t[1] == "Next block"
