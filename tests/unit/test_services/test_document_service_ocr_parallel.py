"""Regression tests for parallel scanned-page OCR extraction.

Guards against reintroducing the OcrExtractor.page_result race: a single
shared OcrExtractor instance stashes its result on `self.page_result`, so
reusing one instance across concurrent extract_page() calls would let a
faster page's result clobber a slower page's before it's read back.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _session_cm(mock_session):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_session)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestOcrPageParallelExtraction:
    @pytest.mark.asyncio
    async def test_fresh_extractor_per_page_and_order_preserved(self):
        from services.document_service import DocumentService

        # Page 1 is the slowest — if the old shared-instance bug were still
        # present, its result could get clobbered by page 2/3 finishing first.
        delays = {1: 0.03, 2: 0.01, 3: 0.02}
        created_instances = []

        class FakeOcrExtractor:
            def __init__(self, client, file_path):
                self.client = client
                self.file_path = file_path
                self.page_result = None
                created_instances.append(self)

            async def extract_page(self, page_number):
                await asyncio.sleep(delays[page_number])
                result = MagicMock()
                result.markdown = f"page{page_number}"
                result.layout_elements = []
                self.page_result = result
                return []

        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_doc.source_language = "en"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc
        mock_db_manager = MagicMock()
        mock_db_manager.session.side_effect = lambda: _session_cm(mock_session)

        captured = {}

        class FakeDigitizedText:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch(
            "services.extractors.docling_pdf_extractor.DoclingPdfExtractor"
        ) as MockPdfExtractor, patch(
            "services.extractors.docling_pdf_extractor.classify_pages",
            return_value={1: "scanned", 2: "scanned", 3: "scanned"},
        ), patch(
            "services.extractors.ocr_extractor.OcrExtractor", FakeOcrExtractor
        ), patch(
            "services.storage_service.DocumentStorageService"
        ) as MockStorage, patch(
            "openai.AsyncOpenAI"
        ), patch(
            "services.document_service.DigitizedText", FakeDigitizedText
        ), patch(
            "services.task_manager.task_manager.submit", return_value="TASK_X"
        ), patch(
            "services.export_service.export_service.cache_ocr_exports_after_extract",
            new_callable=AsyncMock,
        ):
            MockPdfExtractor.return_value._doc = MagicMock()
            MockStorage.return_value.save_page_result = MagicMock()

            svc = DocumentService()
            await svc._run_extraction_body(
                "DOC_TEST",
                file_path="/tmp/fake.pdf",
                fmt="pdf",
                total_pages=3,
                task_id=None,
                db_manager=mock_db_manager,
            )

        assert len(created_instances) == 3, "expected one OcrExtractor per scanned page"

        ocr_content = captured.get("ocr_content") or ""
        assert ocr_content.index("page1") < ocr_content.index("page2") < ocr_content.index("page3")
