"""Tests that summarization prompts always require Vietnamese output."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import pipeline_output_lang_clause


def _make_llm(response="Summary text"):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(return_value=10)
    return llm


class TestChunkSummarizationLanguage:
    @pytest.mark.asyncio
    async def test_vietnamese_in_chunk_map_prompt(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        with patch(
            "core.pageindex.enrichment.base.BaseEnricher.chunk_text",
            return_value=["chunk one"],
        ):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        expected = pipeline_output_lang_clause()
        assert any(expected.strip() in p or "Vietnamese" in p for p in prompts)
        assert all("same language as the source text" not in p for p in prompts)

    @pytest.mark.asyncio
    async def test_vietnamese_even_if_legacy_env_patched(self):
        """SUMMARY_OUTPUT_LANG env no longer changes prompt language."""
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        with (
            patch("config.settings.settings.summary_output_lang", "en"),
            patch(
                "core.pageindex.enrichment.base.BaseEnricher.chunk_text",
                return_value=["chunk one"],
            ),
        ):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("Vietnamese" in p for p in prompts)
        assert not any("Respond in English" in p for p in prompts)


class TestTreeSummarizationLanguage:
    @pytest.mark.asyncio
    async def test_vietnamese_in_synthesis_prompt(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        tree = {
            "title": "Root",
            "content": "root content",
            "children": [{"title": "Child", "content": "child content", "children": []}],
        }
        fake_tree_index = MagicMock()
        fake_tree_index.tree_data = tree
        fake_tree_index.id = "TI_001"

        with patch("services.summarization_service.get_db_manager") as mock_dbm:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
                fake_tree_index
            )
            mock_dbm.return_value.session.return_value = mock_session

            await svc._hierarchical_tree_summarize("DOC_001", llm, task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("Vietnamese" in p for p in prompts)
