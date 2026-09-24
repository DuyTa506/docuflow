"""Terms drifted between units: "non-maskable interrupt" became two different
Vietnamese phrases a few paragraphs apart (E2E Q5). The document's bilingual
keywords give a small glossary that every translation unit now sees."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator
from utils.glossary import glossary_clause, parse_glossary

DISPLAYS = [
    "Hiệu ứng hạt đơn (单粒子效应)",
    "Ngắt không che được (Non-maskable interrupt)",
    "no parentheses here",
    None,
]


def test_parse_bilingual_display():
    assert parse_glossary(DISPLAYS) == {
        "单粒子效应": "Hiệu ứng hạt đơn",
        "Non-maskable interrupt": "Ngắt không che được",
    }


def test_clause_lists_only_terms_present_in_the_text():
    glossary = parse_glossary(DISPLAYS)
    clause = glossary_clause(glossary, "The NON-MASKABLE INTERRUPT line is sampled.")
    assert "Non-maskable interrupt → Ngắt không che được" in clause
    assert "单粒子效应" not in clause
    assert glossary_clause(glossary, "nothing relevant") == ""


@pytest.mark.asyncio
async def test_translation_prompt_carries_the_glossary():
    llm = MagicMock()
    llm.chat_completion_with_finish_reason = AsyncMock(return_value=("Đường ngắt", "stop"))
    t = StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi")
    t.count_tokens = MagicMock(return_value=10)
    t.glossary = parse_glossary(DISPLAYS)

    await t._translate_text_impl("The non-maskable interrupt line.")

    prompt = llm.chat_completion_with_finish_reason.await_args[0][0]
    assert "Ngắt không che được" in prompt
