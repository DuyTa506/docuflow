"""Tests for keyword validation and atomic persistence behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.keyword_validation import validate_keyword_batch


SOURCE = (
    "We study large language models and text-to-SQL on the Spider benchmark. "
    "1. Introduction explains the motivation."
)


class TestKeywordValidation:
    def test_rejects_paragraph_like_keyword(self):
        items = [
            {"keyword": "large language models", "weight": 0.9},
            {
                "keyword": SOURCE,
                "weight": 0.8,
            },
        ]
        validated, diag = validate_keyword_batch(items, source_text=SOURCE, pool_size=10)
        assert len(validated) == 1
        assert diag["rejected"].get("paragraph")

    def test_requires_grounding(self):
        items = [{"keyword": "quantum gravity", "weight": 0.9}]
        validated, _ = validate_keyword_batch(items, source_text=SOURCE, pool_size=10)
        assert validated == []


@pytest.mark.asyncio
async def test_keyword_failure_preserves_existing_rows():
    from services.keyword_service import KeywordService

    svc = KeywordService()
    llm = MagicMock()
    llm.count_tokens = MagicMock(return_value=100)
    llm.chat_completion = AsyncMock(return_value="not json")

    with patch("api.dependencies.get_llm_client", return_value=llm), patch.object(
        svc, "_read_text", return_value=SOURCE
    ), patch.object(svc, "_content_candidates", return_value=[{"keyword": "large language models", "score": 0.9}]), patch.object(
        svc, "_llm_candidates", return_value=[]
    ), patch(
        "services.keyword_service.get_db_manager"
    ) as mock_db_mgr:
        session = MagicMock()
        mock_db_mgr.return_value.session.return_value.__enter__.return_value = session
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        with pytest.raises(ValueError):
            await svc._do_extract("DOC_TEST", 20, extraction_id="EXT_1", task_id=None)

        session.query.return_value.filter.return_value.delete.assert_not_called()
