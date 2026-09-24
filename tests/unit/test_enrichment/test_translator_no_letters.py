"""Live regression (DOC_002 p.32): a formula fragment `(` went to the LLM as a
text block and the translation printed "Please provide the source text you
would like me to translate." Text without a single letter has nothing to
translate — it is kept as is and never sent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator


def _translator():
    llm = AsyncMock()
    llm.chat_completion_with_finish_reason = AsyncMock(
        return_value=("Please provide the source text you would like me to translate.", "stop")
    )
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi"), llm


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["(", " — ", "12.5 %", "(8.161)", "→ ↓"])
async def test_text_without_letters_is_kept_and_not_sent(text):
    t, llm = _translator()
    assert await t.translate_text(text) == text
    assert await t.translate_title(text) == text
    llm.chat_completion_with_finish_reason.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_with_letters_is_still_translated():
    t, llm = _translator()
    await t.translate_text("(a)")
    llm.chat_completion_with_finish_reason.assert_awaited()
