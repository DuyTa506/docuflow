"""Tiled OCR for pages that still loop or overflow max_tokens after a retry.

A dense page (Ru_Designing p.294: a sparse opcode table) needs more tokens than
the 8192 context leaves (1131 go to the image; 7000 output tokens were still cut
off), and a page that loops on dot leaders loses everything after the loop.
Cutting the page into horizontal bands at white gaps gives each band a shorter
output; the bands' grounding boxes are mapped back to page coordinates and the
raw outputs concatenated, so layout parsing downstream is unchanged.
"""

import re
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from serving.logic import find_cut_row, process_page_api, remap_band_grounding


def _page_with_gap(width=400, height=1000, gap=(480, 560)):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for y in range(20, height - 20, 12):
        if gap[0] <= y <= gap[1]:
            continue
        draw.line([(20, y), (width - 20, y)], fill="black", width=3)
    return img


def test_find_cut_row_lands_in_the_white_gap():
    cut = find_cut_row(_page_with_gap(), 0, 1000)
    assert 480 < cut < 560


def test_find_cut_row_without_a_gap_stays_near_the_middle():
    img = Image.new("RGB", (400, 1000), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, 1000, 4):
        draw.line([(0, y), (400, y)], fill="black", width=2)
    assert 350 <= find_cut_row(img, 0, 1000) <= 650


def test_remap_band_grounding_to_page_coordinates():
    raw = "<|ref|>text<|/ref|><|det|>[[10, 0, 500, 999]]<|/det|>\nhello"
    out = remap_band_grounding(raw, y0=500, y1=1000, page_height=1000)
    box = [int(v) for v in re.search(r"\[\[(.*?)\]\]", out).group(1).split(",")]
    assert box[0] == 10 and box[2] == 500
    assert abs(box[1] - 500) <= 1 and abs(box[3] - 999) <= 1
    assert out.endswith("hello")


def test_remap_band_grounding_handles_several_boxes():
    raw = "<|ref|>image<|/ref|><|det|>[[0, 0, 10, 10], [0, 990, 10, 999]]<|/det|>"
    out = remap_band_grounding(raw, y0=0, y1=500, page_height=1000)
    boxes = re.findall(r"\[(\d+), (\d+), (\d+), (\d+)\]", out)
    assert [int(b[1]) for b in boxes] == [0, 495]


LOOP = " ." * 3000


class _BandClient:
    """Answers each band from a script: (content, finish_reason) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

        async def create(**kwargs):
            self.calls.append(kwargs)
            content, finish = self.script.pop(0)
            choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
            return SimpleNamespace(choices=[choice])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _band(text):
    return f"<|ref|>text<|/ref|><|det|>[[20, 100, 900, 300]]<|/det|>\n{text}"


@pytest.fixture
def page_image(tmp_path):
    path = tmp_path / "page.png"
    _page_with_gap().save(path)
    return path


async def _result(client, path):
    events = [
        ev async for ev in process_page_api(client, str(path), 1, stream_enabled=False, tiled=True)
    ]
    return events[-1]


@pytest.mark.asyncio
async def test_tiled_page_merges_bands_top_to_bottom(page_image):
    client = _BandClient([(_band("TOP BAND TEXT"), "stop"), (_band("BOTTOM BAND TEXT"), "stop")])
    result = (await _result(client, page_image))["result"]

    assert len(client.calls) == 2
    texts = [e.get("text_full") or e.get("text_content") for e in result.layout_elements]
    assert texts == ["TOP BAND TEXT", "BOTTOM BAND TEXT"]
    top, bottom = result.layout_elements
    assert top["bbox_y2"] <= bottom["bbox_y1"]  # bottom band sits below the cut
    assert result.markdown.index("TOP") < result.markdown.index("BOTTOM")


@pytest.mark.asyncio
async def test_looping_band_is_split_again(page_image):
    client = _BandClient(
        [
            (_band("TOP") + LOOP, "length"),  # top half loops → split in two
            (_band("TOP-A"), "stop"),
            (_band("TOP-B"), "stop"),
            (_band("BOTTOM"), "stop"),
        ]
    )
    result = (await _result(client, page_image))["result"]
    texts = [e.get("text_full") or e.get("text_content") for e in result.layout_elements]
    assert texts == ["TOP-A", "TOP-B", "BOTTOM"]


@pytest.mark.asyncio
async def test_band_that_keeps_looping_keeps_its_text_before_the_loop(page_image):
    good = "Address: Vmware_45:85:dc (00:0c:29:45:85:dc) is the target host of this scan."
    client = _BandClient(
        [
            (_band(good) + LOOP, "length"),
            (_band(good) + LOOP, "length"),
            (_band(good) + LOOP, "length"),
            (_band("BOTTOM"), "stop"),
        ]
    )
    result = (await _result(client, page_image))["result"]
    assert "Vmware_45:85:dc" in result.markdown
    assert " . . . ." not in result.markdown
    assert "BOTTOM" in result.markdown


@pytest.mark.asyncio
async def test_tiled_page_with_nothing_usable_is_degenerate(page_image):
    client = _BandClient([("☐" * 3000, "length")] * 6)
    event = await _result(client, page_image)
    assert event["type"] == "error" and event["code"] == "degenerate"


def _block(label, box, text):
    x1, y1, x2, y2 = box
    return f"<|ref|>{label}<|/ref|><|det|>[[{x1}, {y1}, {x2}, {y2}]]<|/det|>\n{text}\n"


def _texts(result):
    return [e.get("text_full") or e.get("text_content") for e in result.layout_elements]


@pytest.mark.asyncio
async def test_two_column_page_is_read_column_by_column(page_image):
    """Ru_Designing p.294: horizontal bands put the left column's last paragraph
    after the right column's table."""
    top = (
        _block("title", (50, 20, 950, 80), "HEADING")
        + _block("text", (50, 200, 480, 900), "L1")
        + _block("text", (520, 200, 950, 900), "R1")
    )
    bottom = _block("text", (50, 50, 480, 900), "L2") + _block("text", (520, 50, 950, 900), "R2")
    client = _BandClient([(top, "stop"), (bottom, "stop")])
    result = (await _result(client, page_image))["result"]

    assert _texts(result) == ["HEADING", "L1", "L2", "R1", "R2"]
    assert result.markdown.index("L2") < result.markdown.index("R1")


@pytest.mark.asyncio
async def test_single_column_page_keeps_band_order(page_image):
    top = _block("text", (50, 100, 950, 900), "A") + _block("image", (50, 920, 400, 990), "")
    bottom = _block("text", (50, 50, 950, 900), "B")
    client = _BandClient([(top, "stop"), (bottom, "stop")])
    result = (await _result(client, page_image))["result"]
    assert [e["label"] for e in result.layout_elements] == ["text", "image", "text"]
    assert result.markdown.index("A") < result.markdown.index("B")


@pytest.mark.asyncio
async def test_table_cut_by_the_band_is_joined(page_image):
    head = "<table><tr><td>Op</td><td>Code</td></tr><tr><td>ADD</td><td>80</td></tr></table>"
    tail = "<table><tr><td>SUB</td><td>90</td></tr></table>"
    top = _block("text", (50, 100, 950, 500), "Intro") + _block("table", (50, 600, 950, 999), head)
    bottom = _block("table", (50, 0, 950, 300), tail) + _block("text", (50, 400, 950, 900), "After")
    client = _BandClient([(top, "stop"), (bottom, "stop")])
    result = (await _result(client, page_image))["result"]

    tables = [e for e in result.layout_elements if e["label"] == "table"]
    assert len(tables) == 1
    assert "ADD" in result.markdown and "SUB" in result.markdown
    assert result.markdown.count("<table>") == 1
    assert _texts(result)[-1] == "After"
