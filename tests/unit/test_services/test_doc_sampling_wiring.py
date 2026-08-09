"""The head-only digest stages (research, usage_scope, keywords-fallback)
must see content from the END of a large document, not just its first
~budget tokens — regression for the head-only truncation that blinded these
stages to >95% of a 700-page book.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

TAIL_MARKER = "CUOI_TAI_LIEU_MARKER"

# ~600k chars ≈ 150k heuristic tokens — far beyond any input budget, so a
# head-only truncation provably drops the tail marker.
LONG_TEXT = "Đoạn mở đầu tài liệu. " + ("từ đệm nội dung " * 40_000) + f" {TAIL_MARKER} hết."

CATALOG = {
    "undergraduate": [
        {
            "code": "74801",
            "name": "Máy tính",
            "children": [{"code": "7480101", "name": "Khoa học máy tính"}],
        }
    ]
}


def _make_llm(response=""):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _mock_session():
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    s.query.return_value.filter.return_value.all.return_value = []
    s.query.return_value.filter.return_value.first.return_value = None
    s.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    return s


@pytest.mark.asyncio
async def test_research_directions_prompt_covers_document_tail():
    from services.research_direction_service import ResearchDirectionService

    svc = ResearchDirectionService()
    llm = _make_llm("[]")

    with (
        patch("services.research_direction_service.get_db_manager") as mock_dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
        patch.object(svc, "_read_text", return_value=LONG_TEXT),
        patch.object(svc, "_progress"),
        patch.object(svc, "_parse_json_list", return_value=([], False)),
    ):
        mock_dbm.return_value.session.return_value = _mock_session()
        await svc._do_extract("DOC_001", extraction_id=None)

    prompt = llm.chat_completion.await_args[0][0]
    assert TAIL_MARKER in prompt


@pytest.mark.asyncio
async def test_usage_scope_prompt_covers_document_tail():
    from services.usage_scope_service import UsageScopeService

    svc = UsageScopeService()
    llm = _make_llm('{"undergraduate": []}')
    llm.extract_json = MagicMock(return_value={})

    with (
        patch("services.usage_scope_service.get_db_manager") as mock_dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
        patch.object(svc, "_read_text", return_value=LONG_TEXT),
        patch.object(svc, "_progress"),
        patch("services.usage_scope_service.load_catalog", return_value=CATALOG),
    ):
        mock_dbm.return_value.session.return_value = _mock_session()
        await svc._extract("DOC_001", None)

    prompt = llm.chat_completion.await_args[0][0]
    assert TAIL_MARKER in prompt


@pytest.mark.asyncio
async def test_keywords_fallback_context_covers_document_tail():
    from services.keyword_service import KeywordService

    svc = KeywordService()
    llm = _make_llm("[]")

    with (
        patch("services.keyword_service.get_db_manager") as mock_dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
        patch.object(svc, "_read_text", return_value=LONG_TEXT),
        patch.object(svc, "_progress"),
        patch.object(svc, "_extract_json", return_value=[]),
        patch.object(
            svc,
            "_content_candidates",
            return_value=[{"keyword": "từ đệm", "score": 0.5}],
        ),
    ):
        mock_dbm.return_value.session.return_value = _mock_session()
        await svc._extract("DOC_001", 10)

    prompt = llm.chat_completion.await_args[0][0]
    assert TAIL_MARKER in prompt
