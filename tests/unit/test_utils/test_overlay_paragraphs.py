"""Tests for PDF overlay paragraph merging."""

from utils.overlay_paragraphs import merge_overlay_paragraphs


class _P:
    def __init__(self, y0, y1, size=12, brk=False, x0=0, x1=100):
        self.y0, self.y1, self.size, self.brk, self.x0, self.x1 = y0, y1, size, brk, x0, x1


def test_merge_adjacent_short_paragraphs():
    texts = ["Hello", "world", "Next block"]
    # y0=60 puts the third block 40pt from the merged pair's own edge (y1=20),
    # past the 2.5 x 12pt band. It used to be 50, which lands on exactly 30 —
    # the threshold itself, which merges. See the boundary test below.
    meta = [_P(0, 10), _P(11, 20), _P(60, 70)]
    merged_t, merged_m = merge_overlay_paragraphs(texts, meta, max_chars=800)
    assert len(merged_t) == 2
    assert "Hello" in merged_t[0] and "world" in merged_t[0]
    assert merged_t[1] == "Next block"


def test_gap_is_measured_from_the_merged_block_not_its_first_member():
    """Merging unions the bbox, so the next gap is measured from the new edge.

    Here the pair merges to y0=0,y1=20 and the third block sits at y0=50 —
    exactly 2.5 x size away, the inclusive threshold, so all three join. Had
    the gap still been measured from the first member's y1=10 it would read as
    40 and the third block would stay separate.
    """
    meta = [_P(0, 10), _P(11, 20), _P(50, 60)]
    merged_t, merged_m = merge_overlay_paragraphs(
        ["Hello", "world", "Next block"], meta, max_chars=800
    )
    assert merged_t == ["Hello world Next block"]
    assert (merged_m[0].y0, merged_m[0].y1) == (0, 60)
