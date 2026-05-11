"""Tests that summarization prompts use the configured output language."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_llm(response="Summary text"):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(return_value=10)
    return llm


class TestChunkSummarizationLanguage:
    """_chunk_summarize prompts must contain the configured output language."""

    @pytest.mark.asyncio
    async def test_vi_language_in_chunk_map_prompt(self):
        from services.summarization_service import SummarizationService
        svc = SummarizationService()
        llm = _make_llm()

        with patch("config.settings.settings.summary_output_lang", "vi"), \
             patch("core.pageindex.enrichment.base.BaseEnricher.chunk_text",
                   return_value=["chunk one"]):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("Vietnamese" in p for p in prompts), \
            "At least one prompt should mention Vietnamese"
        assert all("same language as the source text" not in p for p in prompts), \
            "Old language clause must not appear"

    @pytest.mark.asyncio
    async def test_en_language_in_chunk_map_prompt(self):
        from services.summarization_service import SummarizationService
        svc = SummarizationService()
        llm = _make_llm()

        with patch("config.settings.settings.summary_output_lang", "en"), \
             patch("core.pageindex.enrichment.base.BaseEnricher.chunk_text",
                   return_value=["chunk one"]):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("English" in p for p in prompts)


class TestTreeSummarizationLanguage:
    """_hierarchical_tree_summarize node prompts must contain output language."""

    @pytest.mark.asyncio
    async def test_vi_language_in_synthesis_prompt(self):
        from services.summarization_service import SummarizationService
        svc = SummarizationService()
        llm = _make_llm()

        tree = {
            "title": "Root",
            "content": "root content",
            "children": [
                {"title": "Child", "content": "child content", "children": []}
            ],
        }
        fake_tree_index = MagicMock()
        fake_tree_index.tree_data = tree
        fake_tree_index.id = "TI_001"

        mock_settings = MagicMock()
        mock_settings.summary_output_lang = "vi"

        with patch("services.summarization_service.settings", mock_settings), \
             patch("services.summarization_service.get_db_manager") as mock_dbm:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = fake_tree_index
            mock_dbm.return_value.session.return_value = mock_session

            await svc._hierarchical_tree_summarize("DOC_001", llm, task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("Vietnamese" in p for p in prompts)
        assert all("same language as the source text" not in p for p in prompts)
