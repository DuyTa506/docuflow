"""§2.1 "Tóm tắt" exported the literal placeholder `[Chưa có — chạy summarize trước]`.

The digest reads its abstract from the *root* node's summary, but the level walk
summarises the nodes inside the tree — the synthetic root wrapper is not one of
them, so `tree_data["summary"]` stayed empty. Meanwhile every one of the
thousands of node summaries the stage had just paid for was checkpointed into
the tree and never read again.

Observed on N4.11.160: the digest listed 265 chapter summaries under §2.2 while
§2.1 showed the placeholder.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.summarization_service import compose_document_summary


def _llm(response="Tổng quan tài liệu."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None
    return llm


@pytest.mark.asyncio
async def test_existing_root_summary_is_used_as_is():
    llm = _llm()
    tree = {"title": "Doc", "summary": "Bản tóm tắt gốc.", "children": []}

    assert await compose_document_summary(llm, tree) == "Bản tóm tắt gốc."
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_root_summary_is_composed_from_top_level_node_summaries():
    llm = _llm()
    tree = {
        "title": "Архитектура компьютера",
        "children": [
            {"title": f"Глава {i}", "summary": f"Tóm tắt chương {i}.", "children": []}
            for i in range(1, 10)
        ],
    }

    result = await compose_document_summary(llm, tree)

    assert result == "Tổng quan tài liệu."
    prompt = llm.chat_completion.await_args.args[0]
    assert "Tóm tắt chương 1." in prompt
    assert "Tóm tắt chương 9." in prompt, "later chapters were dropped from the composition"


@pytest.mark.asyncio
async def test_returns_empty_when_nothing_to_compose_from():
    llm = _llm()
    tree = {"title": "Doc", "children": [{"title": "Section", "children": []}]}

    assert await compose_document_summary(llm, tree) == ""
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_failure_degrades_to_concatenated_node_summaries():
    """A failed composition must not resurrect the placeholder."""
    llm = _llm()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
    tree = {
        "title": "Doc",
        "children": [
            {"title": "Глава 1", "summary": "Nội dung chương một.", "children": []},
            {"title": "Глава 2", "summary": "Nội dung chương hai.", "children": []},
        ],
    }

    result = await compose_document_summary(llm, tree)

    assert "Nội dung chương một." in result
    assert "Nội dung chương hai." in result
