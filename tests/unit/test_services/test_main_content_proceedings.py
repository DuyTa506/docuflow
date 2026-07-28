"""§2.2 của kỷ yếu không phải §2.2 của sách.

Một mục trong kỷ yếu là một (hoặc vài) BBKH độc lập của các tác giả khác nhau.
Dùng prompt "tóm tắt chương sách" cho nó thì model sẽ dựng ra một mạch nội dung
xuyên suốt vốn không tồn tại.

Số BBKH là thứ duy nhất trong mục này không suy ra được bằng luật: một tiêu đề
"Kết luận" có thể là mục con của một bài, cũng có thể là tên một bài. Nên nó
được hỏi model — nhưng hỏi trên **toàn bộ** danh sách tiêu đề, không phải trên
đoạn trích đã lấy mẫu, vì mẫu thì không đếm được. Không đếm được thì bỏ trống,
không đoán.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService

_LONG = "Nội dung bài báo khoa học. " * 40


def _unit(member_titles=None):
    return {
        "title": "Khoa học máy tính",
        "content": "",
        "member_titles": member_titles or [],
        "children": [{"title": "x", "content": _LONG, "children": []}],
    }


def _llm(payload):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value=payload if isinstance(payload, str) else json.dumps(payload)
    )
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    # None keeps BaseEnricher on the char-heuristic branch; a MagicMock
    # `.encoding` sends it into tiktoken and returns mock objects for text.
    llm.encoding = None
    return llm


class TestProceedingsPrompt:
    @pytest.mark.asyncio
    async def test_prompt_says_independent_papers_not_chapter(self):
        llm = _llm({"summary": "Tóm tắt cụm.", "paper_count": 5})

        await MainContentService()._summarize_chapter(llm, _unit(), 1, doc_kind="proceedings")

        prompt = llm.chat_completion.call_args[0][0]
        assert "independent scientific papers" in prompt
        assert "NOT a chapter of a single continuous work" in prompt

    @pytest.mark.asyncio
    async def test_the_complete_heading_list_is_handed_over(self):
        titles = ["Bài A", "Giới thiệu", "Bài B", "Kết luận"]
        llm = _llm({"summary": "s", "paper_count": 2})

        await MainContentService()._summarize_chapter(llm, _unit(titles), 1, doc_kind="proceedings")

        prompt = llm.chat_completion.call_args[0][0]
        for title in titles:
            assert title in prompt

    @pytest.mark.asyncio
    async def test_book_mode_still_uses_the_chapter_prompt(self):
        llm = _llm("Tóm tắt chương.")

        await MainContentService()._summarize_chapter(llm, _unit(), 1)

        prompt = llm.chat_completion.call_args[0][0]
        assert "book chapter" in prompt
        assert "BBKH" not in prompt


class TestPaperCount:
    @pytest.mark.asyncio
    async def test_a_counted_cluster_carries_its_count(self):
        llm = _llm({"summary": "Các nghiên cứu về botnet.", "paper_count": 5})

        chapter, degraded, _ = await MainContentService()._summarize_chapter(
            llm, _unit(), 1, doc_kind="proceedings"
        )

        assert chapter["paper_count"] == 5
        assert chapter["content"] == "Các nghiên cứu về botnet."
        assert degraded is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("count", [None, 0, 1, "nhiều"])
    async def test_an_uncountable_section_carries_no_count(self, count):
        """Không có `paper_count` ⇒ render thành BBKH đơn lẻ. Đó là câu trả lời
        trung thực; "gồm 1 BBKH" hoặc một con số bịa thì không."""
        llm = _llm({"summary": "Một bài báo.", "paper_count": count})

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _unit(), 1, doc_kind="proceedings"
        )

        assert "paper_count" not in chapter

    @pytest.mark.asyncio
    async def test_prose_instead_of_json_still_yields_a_summary(self):
        llm = _llm("Cụm này gồm các nghiên cứu về mạng cảm biến.")

        chapter, degraded, _ = await MainContentService()._summarize_chapter(
            llm, _unit(), 2, doc_kind="proceedings"
        )

        assert chapter["content"] == "Cụm này gồm các nghiên cứu về mạng cảm biến."
        assert "paper_count" not in chapter
        assert degraded is False, "mất con số thì không phải là lỗi runtime"

    @pytest.mark.asyncio
    async def test_llm_failure_is_reported_as_degraded(self):
        llm = _llm("")
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("down"))

        chapter, degraded, _ = await MainContentService()._summarize_chapter(
            llm, _unit(), 1, doc_kind="proceedings"
        )

        assert degraded is True
        assert chapter["content"]


class TestDefaultTitle:
    @pytest.mark.asyncio
    async def test_untitled_unit_is_labelled_bbkh_not_chuong(self):
        llm = _llm({"summary": "s", "paper_count": None})
        node = {"title": "", "content": "", "children": [{"content": _LONG, "children": []}]}

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, node, 7, doc_kind="proceedings"
        )

        assert chapter["title_vi"] == "BBKH 7"
