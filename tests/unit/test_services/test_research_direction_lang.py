"""Tests that research direction prompts use the configured output language."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_llm():
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(
        return_value='[{"direction_name": "ML", "is_predefined": false, "confidence": 0.9, "reasoning": "text"}]'
    )
    llm.count_tokens = MagicMock(return_value=10)
    return llm


class TestResearchDirectionLanguage:
    @pytest.mark.asyncio
    async def test_vi_language_clause_present(self):
        from services.research_direction_service import ResearchDirectionService
        svc = ResearchDirectionService()
        llm = _make_llm()

        with patch("config.settings.settings.research_output_lang", "vi"), \
             patch("services.research_direction_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=llm), \
             patch.object(svc, "_read_text", return_value="document text"), \
             patch.object(svc, "_find_task_id", return_value=None), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_extract_json", return_value=[
                 {"direction_name": "ML", "is_predefined": False,
                  "confidence": 0.9, "reasoning": "text"}
             ]):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            await svc._do_extract("DOC_001", extraction_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("Vietnamese" in p for p in prompts), "Prompt must mention Vietnamese"

    @pytest.mark.asyncio
    async def test_en_language_clause_present(self):
        from services.research_direction_service import ResearchDirectionService
        svc = ResearchDirectionService()
        llm = _make_llm()

        with patch("config.settings.settings.research_output_lang", "en"), \
             patch("services.research_direction_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=llm), \
             patch.object(svc, "_read_text", return_value="document text"), \
             patch.object(svc, "_find_task_id", return_value=None), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_extract_json", return_value=[
                 {"direction_name": "ML", "is_predefined": False,
                  "confidence": 0.9, "reasoning": "text"}
             ]):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            await svc._do_extract("DOC_001", extraction_id=None)

        prompts = [call.args[0] for call in llm.chat_completion.call_args_list]
        assert any("English" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_char_cap_replaced_by_token_truncation(self):
        """The prompt must NOT use the old text[:15000] cap (whole text must pass in)."""
        from services.research_direction_service import ResearchDirectionService
        svc = ResearchDirectionService()
        long_text = "word " * 4000  # > 15000 chars

        llm = _make_llm()
        mock_settings = MagicMock()
        mock_settings.research_output_lang = "vi"
        mock_settings.ai_chunk_tokens = 100000

        with patch("services.research_direction_service.settings", mock_settings), \
             patch("services.research_direction_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=llm), \
             patch.object(svc, "_read_text", return_value=long_text), \
             patch.object(svc, "_find_task_id", return_value=None), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_extract_json", return_value=[]):
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            await svc._do_extract("DOC_001", extraction_id=None)

        prompt = llm.chat_completion.call_args.args[0]
        # When budget is large the full text should be present (not hard-truncated at 15k)
        assert len(prompt) > 15000, "With large context window, full text should be in prompt"
