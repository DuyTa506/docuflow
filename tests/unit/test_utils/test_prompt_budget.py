"""Whole-request prompt budget tests."""

from unittest.mock import MagicMock

import pytest

from utils.prompt_budget import PromptBudget, PromptBudgetError, allocate_document_sample


def test_usage_scope_style_prompt_stays_within_cap():
    enricher = MagicMock()
    enricher.count_tokens = lambda text: len(text) // 4

    catalog = "0700001 Example discipline\n" * 400
    fixed_prefix = "RULES:\n" + catalog
    fixed_suffix = "JSON:"
    budget = PromptBudget(context_tokens=16384, output_reserve=3000, safety_margin=256)

    sample, meta = allocate_document_sample(
        document_id="DOC_X",
        text="document " * 50000,
        enricher=enricher,
        budget=budget,
        fixed_parts=[fixed_prefix, "DOCUMENT EXCERPT:\n", fixed_suffix],
        sample_builder=lambda token_budget: "x" * (token_budget * 4),
    )
    assert meta["total_tokens"] <= budget.input_cap
    assert sample


def test_fixed_content_overflow_raises():
    enricher = MagicMock()
    enricher.count_tokens = lambda text: 20000

    budget = PromptBudget(context_tokens=16384, output_reserve=3000)
    with pytest.raises(PromptBudgetError):
        allocate_document_sample(
            document_id="DOC_X",
            text="body",
            enricher=enricher,
            budget=budget,
            fixed_parts=["huge fixed prompt"],
            sample_builder=lambda _: "sample",
        )
