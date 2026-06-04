"""Translator uses lang_name() in prompts and skips same-language pairs."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.pageindex.enrichment.translator import StructuredTranslator


@pytest.mark.asyncio
async def test_prompt_uses_full_language_names():
    llm = AsyncMock()
    llm.count_tokens = MagicMock(return_value=10)
    with patch.object(StructuredTranslator, "process_with_retry", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "你好"
        t = StructuredTranslator(llm, source_lang="en", target_lang="zh")
        await t.translate_text("Hello world")
        prompt = mock_llm.call_args[0][0]
        assert "from English to Chinese" in prompt
        assert "from en to zh" not in prompt


@pytest.mark.asyncio
async def test_same_language_returns_source_unchanged():
    llm = AsyncMock()
    t = StructuredTranslator(llm, source_lang="en", target_lang="en")
    out = await t.translate_text("Hello")
    assert out == "Hello"
    llm.chat.assert_not_called() if hasattr(llm, "chat") else None
