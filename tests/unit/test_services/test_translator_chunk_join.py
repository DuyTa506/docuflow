"""Tests for translate_text_chunked join behaviour."""

import pytest
from unittest.mock import AsyncMock, MagicMock

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
