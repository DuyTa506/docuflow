"""Tests for translate_text_chunked join behaviour."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator


class TestTranslateTextChunked:
    @pytest.mark.asyncio
    async def test_joins_chunks_with_double_newline(self):
        translator = StructuredTranslator(
            llm_client=MagicMock(),
            source_lang="en",
            target_lang="vi",
            chunk_size=10,
        )
        translator.count_tokens = MagicMock(return_value=100)
        translator.chunk_text = MagicMock(return_value=["part one", "part two"])
        translator.translate_text = AsyncMock(side_effect=lambda t: f"VI:{t}")

        result = await translator.translate_text_chunked("long text")
        assert result == "VI:part one\n\nVI:part two"

    @pytest.mark.asyncio
    async def test_translate_text_auto_chunks_when_over_budget(self):
        translator = StructuredTranslator(
            llm_client=MagicMock(),
            source_lang="en",
            target_lang="vi",
            chunk_size=50,
        )
        translator.count_tokens = MagicMock(return_value=200)
        translator.translate_text_chunked = AsyncMock(return_value="đã chia khối")

        out = await translator.translate_text("a very long source paragraph " * 40)
        assert out == "đã chia khối"
        translator.translate_text_chunked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_toc_text_is_translated_in_line_batches(self):
        translator = StructuredTranslator(
            llm_client=MagicMock(),
            source_lang="ru",
            target_lang="vi",
            chunk_size=8000,
        )
        translator.count_tokens = MagicMock(return_value=100)
        toc = "\n".join(f"Глава {i}. Тема ..... {i * 10}" for i in range(1, 25))
        calls: list[str] = []

        async def fake_impl(text, _split_depth=0):
            calls.append(text)
            return "\n".join(f"VI:{ln}" for ln in text.splitlines())

        translator._translate_text_impl = AsyncMock(side_effect=fake_impl)

        out = await translator.translate_text(toc)
        assert "VI:Глава 1" in out
        assert "VI:Глава 24" in out
        # One fat call would be a single impl invoke; batches → several.
        assert translator._translate_text_impl.await_count >= 2
        assert all(c.count("\n") < 20 for c in calls)
