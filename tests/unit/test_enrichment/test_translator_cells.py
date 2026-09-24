"""Table cells are translated in marked batches, not one LLM call per cell."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator


def _translator(answer):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=answer)
    t = StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi")
    t.count_tokens = MagicMock(return_value=10)
    return t, llm


@pytest.mark.asyncio
async def test_batch_answer_is_mapped_by_marker_not_by_line_order():
    t, llm = _translator("⟦2⟧ Điện áp\n⟦1⟧ Tên")

    out = await t.translate_cells(["Name", "Voltage", "Name"])

    assert out == ["Tên", "Điện áp", "Tên"]
    assert llm.chat_completion.await_count == 1
    assert "⟦3⟧" not in llm.chat_completion.await_args[0][0]  # duplicates sent once


@pytest.mark.asyncio
async def test_missing_marker_falls_back_to_single_title_translation():
    t, _ = _translator("⟦1⟧ Tên")
    t.translate_title = AsyncMock(return_value="Điện áp")

    out = await t.translate_cells(["Name", "Voltage"])

    assert out == ["Tên", "Điện áp"]
    t.translate_title.assert_awaited_once_with("Voltage")


@pytest.mark.asyncio
async def test_same_language_is_a_no_op():
    llm = MagicMock()
    llm.chat_completion = AsyncMock()
    t = StructuredTranslator(llm_client=llm, source_lang="vi", target_lang="vi")

    assert await t.translate_cells(["Tên"]) == ["Tên"]
    llm.chat_completion.assert_not_awaited()
