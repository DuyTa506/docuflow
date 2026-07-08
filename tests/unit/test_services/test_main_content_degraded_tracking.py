"""_summarize_chapter should report when it fell back to raw-text on LLM failure."""
import pytest
from unittest.mock import AsyncMock


class TestSummarizeChapterDegradedFlag:
    @pytest.mark.asyncio
    async def test_returns_degraded_true_on_llm_failure(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM hiccup"))

        node = {"title": "Chapter One", "content": "x" * 200}
        chapter, degraded = await svc._summarize_chapter(llm, node, 1)

        assert degraded is True
        assert chapter["content"].startswith("x" * 100)

    @pytest.mark.asyncio
    async def test_returns_degraded_false_on_success(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.chat_completion = AsyncMock(return_value="A clean summary.")

        node = {"title": "Chapter One", "content": "x" * 200}
        chapter, degraded = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        assert chapter["content"] == "A clean summary."

    @pytest.mark.asyncio
    async def test_short_content_skips_llm_and_is_not_degraded(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()

        node = {"title": "Chapter One", "content": "too short"}
        chapter, degraded = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        llm.chat_completion.assert_not_called()
