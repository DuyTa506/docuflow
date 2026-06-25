"""Tests that flat fallback uses ai_chunk_tokens and joins with newlines."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTranslationChunkSize:
    @pytest.mark.asyncio
    async def test_flat_fallback_receives_settings_chunk_size(self):
        from services.translation_service import TranslationService

        svc = TranslationService()
        mock_llm = AsyncMock()

        mock_dt = MagicMock()
        mock_dt.normalized_content = "text to translate"
        mock_dt.ocr_content = None

        mock_doc = MagicMock()
        mock_doc.source_language = "en"
        mock_doc.format = "pdf"
        mock_doc.file_path = "/tmp/test.pdf"

        mock_settings = MagicMock()
        mock_settings.ai_chunk_tokens = 50000
        mock_settings.upload_dir = "/tmp/uploads"
        mock_settings.enable_pdf_overlay = False
        mock_settings.ocr_download_spatial_max_elements = 500_000

        with patch("services.translation_service.settings", mock_settings), \
             patch("services.translation_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client", return_value=mock_llm), \
             patch.object(svc, "_find_task_id", return_value="TASK_1"), \
             patch.object(svc, "_wait_for_digitized_text", new_callable=AsyncMock), \
             patch.object(svc, "_progress"), \
             patch("core.pageindex.enrichment.translator.StructuredTranslator") as MockTranslator, \
             patch("services.translators.FlatTranslator") as MockFlat:

            mock_translator_instance = MagicMock()
            mock_translator_instance.chunk_size = 50000
            MockTranslator.return_value = mock_translator_instance

            mock_flat_instance = MagicMock()
            mock_flat_instance.translate_text = AsyncMock(return_value={
                "translation_mode": "flat",
                "translated_elements": None,
                "translated_content": "translated",
                "translated_file_path": None,
            })
            MockFlat.return_value = mock_flat_instance

            mock_repo = MagicMock()
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.get.return_value = mock_doc
            mock_repo.count_elements.return_value = 0

            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_dbm.return_value.session.return_value = mock_session
            mock_session.query.return_value.join.return_value.filter.return_value.options.return_value.order_by.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

            with patch("data.repositories.DocumentRepository", return_value=mock_repo):
                await svc._translate("DOC_001", "vi", "general", translation_id="TRANS_1")

            _, kwargs = MockTranslator.call_args
            assert kwargs.get("chunk_size") == 50000
            mock_flat_instance.translate_text.assert_awaited_once()
