"""Per-run memo cache: repeated strings (page headers/footers appear on every
page of a 700-page book) must hit the LLM once, including when the duplicates
run concurrently.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator


def _translator():
    llm = AsyncMock()
    calls = [0]

    async def completion(prompt, **kwargs):
        calls[0] += 1
        await asyncio.sleep(0.01)
        return ("bản dịch", "stop")

    llm.chat_completion_with_finish_reason = AsyncMock(side_effect=completion)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    t = StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi")
    return t, calls


@pytest.mark.asyncio
async def test_repeated_text_translated_once():
    t, calls = _translator()
    outs = [await t.translate_text("Chapter header — Journal of Testing") for _ in range(50)]
    assert calls[0] == 1
    assert all(o == "bản dịch" for o in outs)


@pytest.mark.asyncio
async def test_concurrent_duplicates_share_one_call():
    t, calls = _translator()
    outs = await asyncio.gather(*(t.translate_text("Running footer page text") for _ in range(20)))
    assert calls[0] == 1
    assert all(o == "bản dịch" for o in outs)


@pytest.mark.asyncio
async def test_distinct_texts_each_call():
    t, calls = _translator()
    await t.translate_text("First unique sentence")
    await t.translate_text("Second unique sentence")
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_long_text_skips_memo():
    t, calls = _translator()
    long_text = "word " * 300  # > memo cutoff
    await t.translate_text(long_text)
    first = calls[0]
    await t.translate_text(long_text)
    assert calls[0] == first * 2  # second run re-calls — no memoization
