"""Từ khoá read the outline, never the document.

With a TreeIndex present the stage built candidates out of node *titles*,
whole *summaries* and whole *body strings* — each one entire field as a single
candidate — then kept the 50 heaviest. On a book with hundreds of headings
that list is 100% titles, and the "context" handed to the model was the same
titles again. Nothing in the pipeline ever looked at the document's own words,
which is exactly the reviewer's complaint: keywords not drawn from the core
content.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

TREE = {
    "title": "Kiến trúc máy tính",
    "summary": "Cuốn sách trình bày kiến trúc máy tính từ mức cổng logic lên tới hệ điều hành.",
    "content": "Đoạn thân bài dài nói về pipeline, bộ nhớ đệm và tập lệnh vi xử lý.",
    "children": [
        {
            "title": "Chương 1: Tổ chức hệ thống máy tính",
            "summary": "Tóm tắt về tổ chức hệ thống.",
            "content": "Nội dung chi tiết về bus, thanh ghi và chu kỳ lệnh.",
            "children": [],
        }
    ],
}


def _make_llm():
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value="[]")
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _session(tree_index):
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    s.query.return_value.filter.return_value.order_by.return_value.first.return_value = tree_index
    s.query.return_value.filter.return_value.first.return_value = None
    return s


class TestTreeCandidates:
    def test_whole_body_text_is_not_a_candidate(self):
        from services.keyword_service import KeywordService

        candidates = [c["keyword"] for c in KeywordService()._tree_candidates(TREE)]

        assert TREE["content"] not in candidates
        assert TREE["children"][0]["content"] not in candidates

    def test_whole_summary_is_not_a_candidate(self):
        from services.keyword_service import KeywordService

        candidates = [c["keyword"] for c in KeywordService()._tree_candidates(TREE)]

        assert TREE["summary"] not in candidates

    def test_titles_are_still_candidates(self):
        from services.keyword_service import KeywordService

        candidates = [c["keyword"] for c in KeywordService()._tree_candidates(TREE)]

        assert any("Tổ chức hệ thống máy tính" in c for c in candidates)


class TestContentIsRead:
    @pytest.mark.asyncio
    async def test_tfidf_runs_even_when_a_tree_exists(self):
        """Outline titles alone cannot surface a term the headings never name."""
        from services.keyword_service import KeywordService

        svc = KeywordService()
        llm = _make_llm()
        tree_index = MagicMock()
        tree_index.tree_data = TREE

        with (
            patch("services.keyword_service.get_db_manager") as dbm,
            patch("api.dependencies.get_llm_client", return_value=llm),
            patch("utils.tree_payload.get_tree_payload", return_value=TREE),
            patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
            patch.object(svc, "_read_text", return_value="bộ nhớ đệm nhiều tầng " * 200),
            patch.object(svc, "_progress"),
            patch.object(svc, "_extract_json", return_value=[]),
            patch.object(
                svc,
                "_content_candidates",
                return_value=[{"keyword": "bộ nhớ đệm", "score": 0.9}],
            ) as content,
        ):
            dbm.return_value.session.return_value = _session(tree_index)
            await svc._do_extract("DOC_001", 20)

        content.assert_called_once()
        prompt = llm.chat_completion.await_args.args[0]
        assert "bộ nhớ đệm" in prompt

    @pytest.mark.asyncio
    async def test_prompt_carries_a_document_excerpt_not_only_titles(self):
        from services.keyword_service import KeywordService

        svc = KeywordService()
        llm = _make_llm()
        tree_index = MagicMock()
        tree_index.tree_data = TREE
        marker = "DAU_HIEU_TRONG_THAN_BAI"

        with (
            patch("services.keyword_service.get_db_manager") as dbm,
            patch("api.dependencies.get_llm_client", return_value=llm),
            patch("utils.tree_payload.get_tree_payload", return_value=TREE),
            patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
            patch.object(svc, "_read_text", return_value=f"{marker} " + "nội dung " * 500),
            patch.object(svc, "_progress"),
            patch.object(svc, "_extract_json", return_value=[]),
            patch.object(svc, "_content_candidates", return_value=[]),
        ):
            dbm.return_value.session.return_value = _session(tree_index)
            await svc._do_extract("DOC_001", 20)

        prompt = llm.chat_completion.await_args.args[0]
        assert marker in prompt, "the model must see the document, not just its outline"
