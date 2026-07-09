"""Regression: every pipeline prompt that embeds a large, potentially
non-Vietnamese source-text block must repeat the Vietnamese-output
instruction immediately before its generation cue, not just once near the
top of the prompt.

Root cause confirmed LIVE against the actual pipeline LLM (qwen3.5-9b on
:5011): given a single "respond in Vietnamese" instruction followed by
~4000 chars of Russian source text, the model ignored the instruction and
answered in Russian -- a recency-bias failure mode common in smaller/local
models. Repeating the instruction right before the generation point (e.g.
"Summary:", "JSON:") fixed this in a live test on real DOC_059 chapters.

This file locks in that fix across every prompt-construction site that was
found to have the same single-far-back-instruction shape, so a future edit
can't silently regress it back to a single occurrence.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _assert_lang_marker_repeated_near_end(prompt: str, tail_window: int = 350):
    """`Vietnamese` (or the language name generally) must appear at least
    twice, and the LAST occurrence must be within `tail_window` chars of
    the prompt's end (close to the generation cue, not just "somewhere
    after the first mention")."""
    count = prompt.count("Vietnamese")
    assert count >= 2, f"expected the language instruction at least twice, found {count} in:\n{prompt}"
    last_idx = prompt.rfind("Vietnamese")
    assert len(prompt) - last_idx < tail_window, (
        f"trailing language reminder is {len(prompt) - last_idx} chars from the end "
        f"(expected < {tail_window}) -- not close enough to the generation cue"
    )


def _make_llm(response="Summary text"):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(return_value=10)
    return llm


class TestMainContentServicePromptRecency:
    @pytest.mark.asyncio
    async def test_chapter_summary_prompt(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = _make_llm()
        node = {"title": "Chapter One", "content": "Русский текст. " * 50}

        await svc._summarize_chapter(llm, node, 1)

        prompt = llm.chat_completion.await_args[0][0]
        _assert_lang_marker_repeated_near_end(prompt)


class TestSummarizationServicePromptRecency:
    @pytest.mark.asyncio
    async def test_chunk_map_prompt(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        with patch(
            "core.pageindex.enrichment.base.BaseEnricher.chunk_text",
            return_value=["Текст на русском языке. " * 50],
        ):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        # First call is the per-chunk map prompt.
        _assert_lang_marker_repeated_near_end(prompts[0])

    @pytest.mark.asyncio
    async def test_final_synthesis_prompt(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        with patch(
            "core.pageindex.enrichment.base.BaseEnricher.chunk_text",
            return_value=["chunk one"],
        ):
            await svc._chunk_summarize(llm, "full text", task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        # Last call is the comprehensive-summary synthesis prompt.
        _assert_lang_marker_repeated_near_end(prompts[-1])

    @pytest.mark.asyncio
    async def test_tree_synthesis_prompt_with_children(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        tree = {
            "title": "Root",
            "content": "Русский текст корневого узла. " * 30,
            "children": [
                {"title": "Child", "content": "Дочерний текст. " * 30, "children": []}
            ],
        }
        fake_tree_index = MagicMock()
        fake_tree_index.tree_data = tree
        fake_tree_index.id = "TI_001"

        with patch("services.summarization_service.get_db_manager") as mock_dbm:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = fake_tree_index
            mock_dbm.return_value.session.return_value = mock_session

            await svc._hierarchical_tree_summarize("DOC_001", llm, task_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        for p in prompts:
            _assert_lang_marker_repeated_near_end(p)

    @pytest.mark.asyncio
    async def test_collapse_batch_prompt(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        svc = SummarizationService()
        llm = _make_llm(response="merged summary")
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        llm.encoding = None
        enricher = BaseEnricher(llm)

        entries = [f"### Section {i}\nТекст раздела " * 10 for i in range(6)]
        with patch("services.summarization_service.settings") as mock_settings:
            mock_settings.ai_input_budget_tokens = 5
            mock_settings.ai_max_concurrent_requests = 4
            await svc._collapse_to_budget(
                enricher, llm, entries, "OUTPUT LANGUAGE: You MUST respond entirely in Vietnamese. Do not use any other language for generated prose.\n\n", max_rounds=2
            )

        assert llm.chat_completion.await_count > 0
        for call in llm.chat_completion.call_args_list:
            _assert_lang_marker_repeated_near_end(call.args[0])


class TestResearchDirectionServicePromptRecency:
    @pytest.mark.asyncio
    async def test_directions_prompt(self):
        from services.research_direction_service import ResearchDirectionService

        svc = ResearchDirectionService()
        llm = _make_llm(response='[{"direction_name": "x", "is_predefined": false, "confidence": 0.9, "reasoning": "y"}]')

        with patch("services.research_direction_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=llm), \
             patch.object(svc, "_read_text", return_value="Русский текст документа. " * 100), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_extract_json", return_value=[]):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            await svc._do_extract("DOC_001", extraction_id=None)

        prompt = llm.chat_completion.await_args[0][0]
        _assert_lang_marker_repeated_near_end(prompt)


class TestUsageScopeServicePromptRecency:
    @pytest.mark.asyncio
    async def test_usage_scope_prompt(self):
        from services.usage_scope_service import UsageScopeService

        svc = UsageScopeService()
        llm = _make_llm(response='{"undergraduate": [], "master": [], "phd": [], "strong_research_groups": []}')
        llm.extract_json = MagicMock(return_value={})

        with patch("services.usage_scope_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=llm), \
             patch.object(svc, "_read_text", return_value="Русский текст документа. " * 100), \
             patch.object(svc, "_progress"), \
             patch("services.usage_scope_service._load_catalog", return_value={}):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            await svc._extract("DOC_001", None)

        prompt = llm.chat_completion.await_args[0][0]
        _assert_lang_marker_repeated_near_end(prompt)
