"""Silent-failure visibility: JSON-parse failures should log, not vanish."""

import logging
from unittest.mock import MagicMock

import pytest


class TestBaseServiceExtractJsonLogging:
    def test_logs_warning_and_returns_empty_list_on_failure(self, caplog):
        from services.base_service import BaseTaskService

        svc = BaseTaskService()
        llm = MagicMock()
        llm.extract_json = MagicMock(side_effect=ValueError("boom"))

        with caplog.at_level(logging.WARNING, logger="services.base_service"):
            result = svc._extract_json(llm, "not json at all")

        assert result == []
        assert any("boom" in r.message for r in caplog.records)

    def test_returns_list_unchanged_when_extraction_succeeds(self):
        from services.base_service import BaseTaskService

        svc = BaseTaskService()
        llm = MagicMock()
        llm.extract_json = MagicMock(return_value=[1, 2, 3])

        assert svc._extract_json(llm, "irrelevant") == [1, 2, 3]


class TestLlmClientExtractJsonLogging:
    def test_logs_warning_before_raising_on_no_json_found(self, caplog):
        import json as json_module

        from core.pageindex.llm.openai_client import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)

        with caplog.at_level(logging.WARNING, logger="core.pageindex.llm.llm_client_base"):
            with pytest.raises(json_module.JSONDecodeError):
                client.extract_json("no json here whatsoever")

        assert any("no valid JSON" in r.message for r in caplog.records)


class TestUsageScopeExtractJsonLogging:
    @pytest.mark.asyncio
    async def test_logs_and_falls_back_to_empty_scope_on_failure(self, caplog):
        from unittest.mock import AsyncMock, patch

        from services.usage_scope_service import UsageScopeService

        svc = UsageScopeService()
        llm = AsyncMock()
        llm.chat_completion = AsyncMock(return_value="not valid json")
        llm.extract_json = MagicMock(side_effect=ValueError("bad json"))
        llm.count_tokens = MagicMock(return_value=10)

        with (
            patch("services.usage_scope_service.get_db_manager") as mock_dbm,
            patch.object(svc, "_read_text", return_value="doc text"),
            patch.object(svc, "_progress"),
            patch("api.dependencies.get_llm_client", return_value=llm),
            patch(
                "services.usage_scope_service.load_catalog",
                # Non-empty: an empty catalog now short-circuits before the LLM
                # call, which is the branch this test is NOT about.
                return_value={
                    "undergraduate": ["Ngành Khoa học máy tính"],
                    "master": [],
                    "phd": [],
                    "strong_research_groups": [],
                },
            ),
        ):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_dbm.return_value.session.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = MagicMock()

            with caplog.at_level(logging.WARNING, logger="services.usage_scope_service"):
                result = await svc._extract("DOC_001", task_id=None)

        assert result == {
            "undergraduate": [],
            "master": [],
            "phd": [],
            "strong_research_groups": [],
        }
        assert any("bad json" in r.message for r in caplog.records)
