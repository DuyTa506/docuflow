"""Tests that TranslationService passes ai_chunk_tokens to StructuredTranslator."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTranslationChunkSize:
    @pytest.mark.asyncio
    async def test_structured_translator_receives_settings_chunk_size(self):
        from services.translation_service import TranslationService
        svc = TranslationService()

        mock_llm = AsyncMock()
        mock_llm.count_tokens = MagicMock(return_value=10)

        mock_dt = MagicMock()
        mock_dt.normalized_content = "text to translate"
        mock_dt.ocr_content = None

        mock_doc = MagicMock()
        mock_doc.source_language = "en"

        mock_settings = MagicMock()
        mock_settings.ai_chunk_tokens = 50000

        with patch("services.translation_service.settings", mock_settings), \
             patch("services.translation_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=mock_llm), \
             patch.object(svc, "_find_task_id", return_value=None), \
             patch.object(svc, "_progress"), \
             patch("core.pageindex.enrichment.translator.StructuredTranslator") as MockTranslator:

            mock_translator_instance = MagicMock()
            mock_translator_instance.chunk_size = 50000
            mock_translator_instance.chunk_text = MagicMock(return_value=["chunk"])
            mock_translator_instance.translate_text = AsyncMock(return_value="translated")
            MockTranslator.return_value = mock_translator_instance

            mock_repo = MagicMock()
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.get.return_value = mock_doc

            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_dbm.return_value.session.return_value = mock_session

            with patch("services.translation_service.get_db_manager") as mock_dbm2, \
                 patch("data.repositories.DocumentRepository", return_value=mock_repo):
                mock_session2 = MagicMock()
                mock_session2.__enter__ = MagicMock(return_value=mock_session2)
                mock_session2.__exit__ = MagicMock(return_value=False)
                mock_dbm2.return_value.session.return_value = mock_session2

                await svc._translate("DOC_001", "vi", "general", translation_id=None)

            _, kwargs = MockTranslator.call_args
            assert kwargs.get("chunk_size") == 50000, \
                f"Expected chunk_size=50000, got {kwargs.get('chunk_size')}"
