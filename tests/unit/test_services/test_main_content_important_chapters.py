"""§2.2 gave every unit the same 5-8 sentences, whatever its weight.

The official form (template/(File 3) Mau Tong thuat HLS 2025.docx) asks for the
core content of each chapter, "nhấn mạnh, chi tiết hơn về nội dung với các
chương quan trọng". On the Nội dung 4 set (52 articles of 3-6 units) the judge
found §2.2 conveyed 61% of the key points against 80-89% for full-context
references: a section holding half the article got the same 5-8 sentences as a
one-page conclusion.

A unit's share of the document's text now sets its budget: a dominant unit
(≥ IMPORTANT_UNIT_SHARE) is asked for more detail and given the output room.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import settings
from services.main_content_service import (
    IMPORTANT_UNIT_SHARE,
    MainContentService,
    _collect_chapter_nodes,
)
from utils.doc_kind import PROCEEDINGS

_TEXT = "Nội dung chuyên môn về hệ thống radar và xử lý tín hiệu. " * 30


def _llm(response="Tóm tắt chương."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None
    return llm


def _unit(share=None):
    node = {"title": "Chương 2", "content": "", "children": [{"content": _TEXT, "children": []}]}
    if share is not None:
        node["share"] = share
    return node


def _prompt_and_cap(llm):
    call = llm.chat_completion.await_args
    return call.args[0], call.kwargs.get("max_tokens")


class TestUnitShare:
    def test_units_carry_their_share_of_the_document(self):
        tree = {
            "title": "Document",
            "children": [
                {"title": "Глава 1. Введение", "content": "x" * 2000, "children": []},
                {"title": "Глава 2. Методы", "content": "x" * 6000, "children": []},
                {"title": "Глава 3. Выводы", "content": "x" * 2000, "children": []},
            ],
        }

        nodes, _ = _collect_chapter_nodes(tree)
        shares = [item["node"]["share"] for item in nodes]

        assert len(shares) == 3
        assert sum(shares) == pytest.approx(1.0, abs=0.01)
        assert shares[1] == pytest.approx(0.6, abs=0.01)


class TestChapterBudget:
    @pytest.mark.asyncio
    async def test_dominant_chapter_is_asked_for_more_detail(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _unit(share=0.5), 2)

        prompt, cap = _prompt_and_cap(llm)
        assert "8-12 sentences" in prompt
        assert cap > settings.main_content_chapter_max_tokens

    @pytest.mark.asyncio
    async def test_ordinary_chapter_keeps_the_standard_length(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _unit(share=0.1), 2)

        prompt, cap = _prompt_and_cap(llm)
        assert "5-8 sentences" in prompt
        assert "8-12 sentences" not in prompt
        assert cap == settings.main_content_chapter_max_tokens

    @pytest.mark.asyncio
    async def test_unit_without_a_share_keeps_the_standard_length(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _unit(), 2)

        prompt, _ = _prompt_and_cap(llm)
        assert "5-8 sentences" in prompt

    @pytest.mark.asyncio
    async def test_threshold_is_inclusive(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _unit(share=IMPORTANT_UNIT_SHARE), 2)

        assert "8-12 sentences" in _prompt_and_cap(llm)[0]

    @pytest.mark.asyncio
    async def test_dominant_proceedings_section_is_asked_for_more_detail(self):
        llm = _llm('{"summary": "Tóm tắt các bài báo.", "paper_count": null}')

        await MainContentService()._summarize_chapter(
            llm, _unit(share=0.5), 2, doc_kind=PROCEEDINGS
        )

        prompt, cap = _prompt_and_cap(llm)
        assert "8-12 sentences" in prompt
        assert cap > settings.main_content_chapter_max_tokens
