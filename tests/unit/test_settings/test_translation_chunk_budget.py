"""Translation chunks must budget for BOTH input and output in one model slot:
a translation is roughly the size of its source, so chunking at the full input
budget (~85% of the context window) forces the completion past the slot limit
and the tail of every large chunk is silently truncated (finish_reason=length).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings


def test_translation_chunk_tokens_is_half_the_window_minus_overhead():
    s = Settings(
        ai_model_context_window=16384,
        ai_chunk_ratio=0.85,
        translation_prompt_overhead_tokens=600,
    )
    expected = (s.ai_chunk_tokens - 600) // 2
    assert s.translation_chunk_tokens == expected
    assert s.translation_chunk_tokens < s.ai_input_budget_tokens


def test_translation_chunk_tokens_floor():
    s = Settings(ai_model_context_window=1000, translation_prompt_overhead_tokens=600)
    assert s.translation_chunk_tokens >= 512


@pytest.mark.asyncio
async def test_translate_text_passes_max_tokens_to_llm():
    from core.pageindex.enrichment.translator import StructuredTranslator

    llm = AsyncMock()
    llm.chat_completion_with_finish_reason = AsyncMock(return_value=("đã dịch", "stop"))
    llm.count_tokens = MagicMock(return_value=100)

    translator = StructuredTranslator(
        llm_client=llm, source_lang="en", target_lang="vi", chunk_size=6000
    )
    await translator.translate_text("some text")

    kwargs = llm.chat_completion_with_finish_reason.call_args.kwargs
    assert isinstance(kwargs.get("max_tokens"), int)
    assert kwargs["max_tokens"] >= 256
