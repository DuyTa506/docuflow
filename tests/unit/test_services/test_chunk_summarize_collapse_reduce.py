"""Regression tests for the chunk-summarize fallback's collapse-reduce step.

Previously the fallback concatenated every chunk-summary with no budget
check before the final synthesis call — with enough chunks (exactly the
large-document, no-tree-index scenario this fallback exists for), that
concatenation could itself overflow the model's input budget.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_llm(chat_response="merged"):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=chat_response)
    llm.encoding = None  # forces truncate_to_tokens' char-heuristic fallback path
    return llm


class TestCollapseToBudget:
    @pytest.mark.asyncio
    async def test_no_collapse_needed_when_combined_fits_budget(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        svc = SummarizationService()
        llm = _make_llm()
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        enricher = BaseEnricher(llm)

        entries = ["### Section 1\nshort", "### Section 2\nalso short"]
        with patch("services.summarization_service.settings") as mock_settings:
            mock_settings.ai_input_budget_tokens = 1000
            mock_settings.ai_max_concurrent_requests = 4
            result = await svc._collapse_to_budget(enricher, llm, entries, "lang clause")

        assert result == "\n\n".join(entries)
        llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_collapses_many_entries_until_within_budget(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        svc = SummarizationService()
        # Each entry is ~10 tokens; a tiny budget forces multiple collapse rounds.
        llm = _make_llm(chat_response="merged summary")
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        enricher = BaseEnricher(llm)

        entries = [f"### Section {i}\nword " * 10 for i in range(20)]
        with patch("services.summarization_service.settings") as mock_settings:
            mock_settings.ai_input_budget_tokens = 15
            mock_settings.ai_max_concurrent_requests = 4
            result = await svc._collapse_to_budget(
                enricher, llm, entries, "lang clause", max_rounds=5
            )

        assert llm.chat_completion.await_count > 0
        # Final result must respect the budget (either converged or hard-truncated).
        assert enricher.count_tokens(result) <= 15

    @pytest.mark.asyncio
    async def test_gives_up_after_max_rounds_and_hard_truncates(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        svc = SummarizationService()
        # LLM never actually shortens anything — forces max_rounds exhaustion.
        llm = _make_llm(chat_response="word " * 10)
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        llm.encoding = None
        enricher = BaseEnricher(llm)

        entries = [f"### Section {i}\nword " * 10 for i in range(6)]
        with patch("services.summarization_service.settings") as mock_settings:
            mock_settings.ai_input_budget_tokens = 5
            mock_settings.ai_max_concurrent_requests = 4
            result = await svc._collapse_to_budget(
                enricher, llm, entries, "lang clause", max_rounds=2
            )

        # Hard-truncation safety net always applies regardless of convergence.
        assert enricher.count_tokens(result) <= max(5, len(result[:5 * 4].split()) + 1)


class TestPackIntoBatches:
    def test_groups_entries_within_budget(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        llm = MagicMock()
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        enricher = BaseEnricher(llm)

        entries = ["one two", "three four", "five six", "seven eight"]
        batches = SummarizationService._pack_into_batches(enricher, entries, budget=4)

        assert all(
            enricher.count_tokens("\n\n".join(b)) <= 4 or len(b) == 1
            for b in batches
        )
        assert sum(len(b) for b in batches) == len(entries)

    def test_truncates_single_oversized_entry(self):
        from services.summarization_service import SummarizationService
        from core.pageindex.enrichment.base import BaseEnricher

        llm = MagicMock()
        llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        llm.encoding = None
        enricher = BaseEnricher(llm)

        huge_entry = "word " * 50
        batches = SummarizationService._pack_into_batches(enricher, [huge_entry], budget=5)

        assert len(batches) == 1
        assert enricher.count_tokens(batches[0][0]) <= 5
