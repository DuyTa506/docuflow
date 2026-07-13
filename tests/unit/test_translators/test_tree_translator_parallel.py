"""translate_structure walked the tree serially — one awaited LLM call per
node field. Units across the whole tree are independent, so they must run
with bounded concurrency while landing back on the right nodes.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator


def _structure():
    return [
        {
            "title": f"Root {r}",
            "text": f"body {r}",
            "nodes": [{"title": f"Child {r}.{c}", "text": f"child body {r}.{c}"} for c in range(2)],
        }
        for r in range(3)
    ]


@pytest.mark.asyncio
async def test_units_translate_concurrently_and_map_back_to_nodes():
    translator = StructuredTranslator(llm_client=MagicMock(), source_lang="en", target_lang="vi")
    in_flight = [0]
    peak = [0]

    async def fake_translate(text):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        await asyncio.sleep(0.02)
        in_flight[0] -= 1
        return f"VI:{text}"

    with (
        patch.object(translator, "translate_title", side_effect=fake_translate),
        patch.object(translator, "translate_text", side_effect=fake_translate),
        patch.object(translator, "translate_text_chunked", side_effect=fake_translate),
    ):
        out = await translator.translate_structure(_structure())

    assert peak[0] > 1, "tree units still translate one at a time"
    assert out[0]["title"] == "VI:Root 0"
    assert out[2]["text"] == "VI:body 2"
    assert out[1]["nodes"][1]["title"] == "VI:Child 1.1"
    assert out[1]["nodes"][1]["text"] == "VI:child body 1.1"
    # source structure untouched
    src = _structure()
    assert src[0]["title"] == "Root 0"


@pytest.mark.asyncio
async def test_translate_structure_reports_incremental_progress():
    """Large documents route translation through translate_structure — without
    incremental progress the UI shows 0% for the whole run (hours, for a
    super-long book) instead of ticking up as units complete.
    """
    translator = StructuredTranslator(llm_client=MagicMock(), source_lang="en", target_lang="vi")

    async def fake_translate(text):
        return f"VI:{text}"

    calls = []

    async def on_progress(pct, msg):
        calls.append((pct, msg))

    with (
        patch.object(translator, "translate_title", side_effect=fake_translate),
        patch.object(translator, "translate_text", side_effect=fake_translate),
        patch.object(translator, "translate_text_chunked", side_effect=fake_translate),
    ):
        await translator.translate_structure(_structure(), on_progress=on_progress)

    assert len(calls) > 1, "translate_structure only reported progress once (or never)"
    pcts = [pct for pct, _ in calls]
    assert pcts == sorted(pcts)
    assert pcts[-1] > pcts[0]
