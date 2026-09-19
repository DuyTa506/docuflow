"""Repetition-loop OCR pages: tell a real loop from a repetitive page, salvage the rest.

Live regression (E2E 2026-09, 22 blank pages): the tail check flagged complete
pages whose content repeats — a table of ``<td>↓</td>`` arrows (Digital IC
Design p.157), a VHDL array (p.222), a sparse opcode table (Ru_Designing
p.110). Those finished with ``finish_reason="stop"``. The real loops all ran
to ``max_tokens`` (``"length"``) after a page's worth of good text, which was
thrown away with the loop.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from serving.logic import _is_degenerate, process_page_api, trim_repetition_tail

ARROW_TABLE = (
    "<table><tr><td>Step</td><td>Value</td></tr>"
    + "<tr>"
    + "<td>↓</td>" * 18
    + "</tr>"
    + "<tr><td>1.</td><td>2.</td><td>3.</td><td>2.</td><td>3.</td><td>1.</td><td>3.</td></tr></table>"
)
GOOD_TEXT = (
    "<|ref|>text<|/ref|><|det|>[[65, 346, 737, 381]]<|/det|>\n"
    "Address: Vmware_45:85:dc (00:0c:29:45:85:dc) is the target host of this scan.\n"
)
DOT_LOOP = GOOD_TEXT + "....0" + " ." * 3000  # Penetration Testing p.238


def test_complete_page_with_repetitive_table_is_not_degenerate():
    assert _is_degenerate(ARROW_TABLE, truncated=True)  # old verdict on the tail
    assert not _is_degenerate(ARROW_TABLE, truncated=False)


def test_instruction_echo_is_degenerate_even_when_complete():
    echo = "Use <code> for inline code and <pre> for code blocks. " * 5
    assert _is_degenerate(echo, truncated=False)


def test_multiline_loop_to_max_tokens_is_degenerate():
    """Ru_Designing p.298 / Digital IC p.293: ``:\\n`` and ``---\\n\\n`` loops ran
    to max_tokens but the one-line pattern never matched across newlines."""
    assert _is_degenerate(GOOD_TEXT + ":\n" * 2000)
    assert _is_degenerate(GOOD_TEXT + "---\n\n" * 800)


def test_formula_loop_to_max_tokens_is_degenerate():
    """Ballistics p.238: the LaTeX exemption hid a ``\\mathbf{a}`` loop."""
    page = GOOD_TEXT + r"\[ \mathbf{a}=\frac{\mathrm{d}\mathbf{V}}{\mathrm{d}t}=\mathbf{V}"
    assert _is_degenerate(page + r"\mathbf{a}" * 1200 + r"\mathbf")


def test_truncated_latex_page_without_a_loop_is_not_degenerate():
    page = GOOD_TEXT + r"\[ x_{%d}=\frac{a_{%d}}{b_{%d}} \]" + "\n"
    assert not _is_degenerate("".join(page % (i, i, i) for i in range(40)))


def test_trim_repetition_tail_keeps_the_text_before_the_loop():
    trimmed = trim_repetition_tail(DOT_LOOP)
    assert trimmed.startswith(GOOD_TEXT.strip()[:40])
    assert " . . . ." not in trimmed
    assert not _is_degenerate(trimmed)


def test_trim_repetition_tail_rejects_a_page_that_is_all_loop():
    assert trim_repetition_tail("☐" * 3000) is None


def test_trim_keeps_a_truncated_page_without_a_trailing_loop():
    """Ru_Designing p.294: a dense table cut by max_tokens, not a loop."""
    row = "<tr><td>0</td><td>0</td><td>1</td>" + "<td></td>" * 10 + "</tr>"
    table = "<table><tr><td>Код</td><td>Адрес</td></tr>" + row * 40 + "<tr><td>0</td><td>0</td>"
    kept = trim_repetition_tail(table)
    assert kept is not None and table.startswith(kept)
    assert len(kept) > 0.9 * len(table)


class _FakeClient:
    def __init__(self, content, finish_reason):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason
        )
        response = SimpleNamespace(choices=[choice])

        async def create(**_kwargs):
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def _events(client, image_path, **kwargs):
    return [
        ev
        async for ev in process_page_api(client, str(image_path), 1, stream_enabled=False, **kwargs)
    ]


@pytest.fixture
def page_image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (200, 300), "white").save(path)
    return path


@pytest.mark.asyncio
async def test_repetitive_page_that_finished_is_kept(page_image):
    events = await _events(_FakeClient(ARROW_TABLE, "stop"), page_image)
    assert [e["type"] for e in events][-1] == "result"


@pytest.mark.asyncio
async def test_loop_to_max_tokens_is_degenerate_without_salvage(page_image):
    events = await _events(_FakeClient(DOT_LOOP, "length"), page_image)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "degenerate"


@pytest.mark.asyncio
async def test_loop_to_max_tokens_is_salvaged_on_request(page_image):
    events = await _events(_FakeClient(DOT_LOOP, "length"), page_image, salvage_repetition=True)
    result = events[-1]
    assert result["type"] == "result"
    assert "Vmware_45:85:dc" in result["result"].markdown
    assert " . . . ." not in result["result"].markdown
