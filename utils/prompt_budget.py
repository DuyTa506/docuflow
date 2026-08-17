"""Whole-request token budgeting for pipeline LLM prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBudget:
    context_tokens: int
    output_reserve: int
    safety_margin: int = 256

    @property
    def input_cap(self) -> int:
        return max(1, self.context_tokens - self.output_reserve - self.safety_margin)


class PromptBudgetError(ValueError):
    """Raised when fixed prompt content alone exceeds the model context."""


def allocate_document_sample(
    *,
    document_id: str,
    text: str,
    enricher,
    budget: PromptBudget,
    fixed_parts: list[str],
    sample_builder: Callable[[int], str],
) -> tuple[str, dict]:
    """Reserve tokens for fixed instructions, then build the document sample."""
    count_tokens = enricher.count_tokens
    fixed_tokens = sum(count_tokens(part) for part in fixed_parts if part)
    sample_budget = budget.input_cap - fixed_tokens
    if sample_budget < 256:
        raise PromptBudgetError(
            f"fixed prompt uses {fixed_tokens} tokens; only {budget.input_cap} allowed "
            f"(context={budget.context_tokens}, reserve={budget.output_reserve})"
        )

    sample = sample_builder(sample_budget)
    total = fixed_tokens + count_tokens(sample)
    meta = {
        "fixed_tokens": fixed_tokens,
        "sample_tokens": count_tokens(sample),
        "total_tokens": total,
        "input_cap": budget.input_cap,
        "context_tokens": budget.context_tokens,
        "output_reserve": budget.output_reserve,
        "document_id": document_id,
    }
    if total > budget.input_cap:
        raise PromptBudgetError(
            f"assembled prompt {total} tokens exceeds cap {budget.input_cap} "
            f"(context={budget.context_tokens})"
        )
    logger.info(
        "prompt_budget document_id=%s fixed=%s sample=%s total=%s cap=%s",
        document_id,
        meta["fixed_tokens"],
        meta["sample_tokens"],
        meta["total_tokens"],
        meta["input_cap"],
    )
    return sample, meta


def build_pipeline_sample(document_id: str, text: str, enricher, token_budget: int) -> str:
    from utils.doc_sampling import build_pipeline_doc_sample

    return build_pipeline_doc_sample(document_id, text, enricher, token_budget)
