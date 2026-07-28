"""§2.2 summaries described each chapter's *opening*, not the chapter.

`_gather_node_text` is a head-biased pre-order walk capped at 4000 chars. For a
90-page chapter of an 816-page book that is ~1% of the material, all of it from
the front — so "tổng quát" (a synthesis of the whole chapter) was unreachable no
matter how good the model was.

Also covers the unit-selection seam: §2.2 units now come from
`utils.chapter_units.select_chapter_units` instead of "every child of the tree
root", and the selection metadata must reach `MainContent.details` so the
quality report can flag a fragmented digest.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import settings
from services.main_content_service import MainContentService, _collect_chapter_nodes


def _llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Tóm tắt chương.")
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    # MagicMock would otherwise fabricate a `.encoding`, sending
    # BaseEnricher.truncate_to_tokens down the tiktoken branch and returning
    # mock objects instead of text. None selects the char-heuristic branch.
    llm.encoding = None
    return llm


def _chapter_node(marker_head="ĐẦU_CHƯƠNG", marker_tail="CUỐI_CHƯƠNG"):
    """A chapter far larger than the old 4000-char head window."""
    filler = "Nội dung kỹ thuật về kiến trúc máy tính và bộ nhớ đệm. " * 60  # ~3.3k chars
    children = [{"title": f"Mục {i}", "content": filler, "children": []} for i in range(1, 13)]
    children[0]["content"] = marker_head + " " + filler
    children[-1]["content"] = marker_tail + " " + filler
    return {"title": "Глава 4. Уровень микроархитектуры", "content": "", "children": children}


class TestChapterSampling:
    @pytest.mark.asyncio
    async def test_prompt_covers_the_whole_chapter_not_just_its_head(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _chapter_node(), 4)

        prompt = llm.chat_completion.await_args.args[0]
        assert "ĐẦU_CHƯƠNG" in prompt
        assert "CUỐI_CHƯƠNG" in prompt, "late-chapter material never reached the model"
        # Every section is represented, not just the first few that fit a head window.
        assert all(f"Mục {i}" in prompt for i in range(1, 13))

    @pytest.mark.asyncio
    async def test_chapter_call_passes_max_tokens(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _chapter_node(), 4)

        assert (
            llm.chat_completion.await_args.kwargs.get("max_tokens")
            == settings.main_content_chapter_max_tokens
        )

    @pytest.mark.asyncio
    async def test_prompt_demands_synthesis_and_forbids_heading_enumeration(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _chapter_node(), 4)

        prompt = llm.chat_completion.await_args.args[0]
        lowered = prompt.lower()
        assert "entire chapter" in lowered or "whole chapter" in lowered
        assert "do not list" in lowered or "do not enumerate" in lowered
        assert "bullet" in lowered

    @pytest.mark.asyncio
    async def test_node_summaries_are_reused_when_present(self, monkeypatch):
        monkeypatch.setattr(settings, "main_content_prefer_node_summaries", True)
        node = _chapter_node()
        for child in node["children"]:
            child["summary"] = "TÓM_TẮT_NODE"
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, node, 4)

        prompt = llm.chat_completion.await_args.args[0]
        assert "TÓM_TẮT_NODE" in prompt

    @pytest.mark.asyncio
    async def test_node_summaries_ignored_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "main_content_prefer_node_summaries", False)
        node = _chapter_node()
        for child in node["children"]:
            child["summary"] = "TÓM_TẮT_NODE"
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, node, 4)

        assert "TÓM_TẮT_NODE" not in llm.chat_completion.await_args.args[0]


class TestUnitSelectionSeam:
    def test_collect_chapter_nodes_bounds_a_fragmented_tree(self):
        """The old code returned all 265 root children verbatim."""
        body = "Текст раздела о конвейере и кэш-памяти. " * 60
        children = []
        for chapter in range(1, 10):
            children.append({"title": f"Глава {chapter}. Тема", "content": "", "children": []})
            children += [
                {"title": f"{chapter}.{s} Подраздел", "content": body, "children": []}
                for s in range(1, 21)
            ]
        tree = {"title": "Document", "children": children}

        units, meta = _collect_chapter_nodes(tree)

        assert len(units) == 9, [u["node"]["title"] for u in units]
        assert [u["number"] for u in units] == list(range(1, 10))
        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert meta["median_unit_chars"] > settings.main_content_min_unit_chars

    def test_returns_empty_for_empty_tree(self):
        units, meta = _collect_chapter_nodes({})
        assert units == []
        assert meta["unit_selection_tier"] is None
