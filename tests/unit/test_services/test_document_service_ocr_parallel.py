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

        page_store = []  # (page_number, markdown) persisted per page

        class FakeOcrExtractor:
            def __init__(self, client, file_path):
                self.client = client
                self.file_path = file_path
                self.page_result = None
                created_instances.append(self)

            async def extract_page(self, page_number):
                await asyncio.sleep(delays[page_number])
                result = MagicMock()
                result.page_number = page_number
                result.markdown = f"page{page_number}"
                result.layout_elements = []
                self.page_result = result
                return []

        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_doc.source_language = "en"
        mock_doc.quality_report = None

        def query_side_effect(*args):
            q = MagicMock()
            if len(args) == 2:
                # (Page.page_number, Page.markdown_content) — final assembly
                # reads persisted pages back from the DB in page order.
                q.filter.return_value.order_by.return_value.all.return_value = sorted(page_store)
            else:
                model = args[0] if args else None
                if model is FakeDigitizedText:
                    # Upsert path: no existing DigitizedText → insert FakeDigitizedText
                    q.filter.return_value.order_by.return_value.first.return_value = None
                    q.filter.return_value.first.return_value = None
                else:
                    q.filter.return_value.first.return_value = mock_doc
                    q.filter.return_value.all.return_value = []
                    q.filter.return_value.order_by.return_value.all.return_value = []
            return q

        mock_session.query.side_effect = query_side_effect
        mock_db_manager = MagicMock()
        mock_db_manager.session.side_effect = lambda: _session_cm(mock_session)

        captured = {}

        class FakeDigitizedText:
            document_id = MagicMock()
            created_at = MagicMock()

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def __setattr__(self, key, value):
                captured[key] = value
                object.__setattr__(self, key, value)

        with (
            patch(
                "services.extractors.docling_pdf_extractor.DoclingPdfExtractor"
            ) as MockPdfExtractor,
            patch(
                "services.extractors.docling_pdf_extractor.classify_pages",
                return_value={1: "scanned", 2: "scanned", 3: "scanned"},
            ),
            patch("services.extractors.ocr_extractor.OcrExtractor", FakeOcrExtractor),
            patch("services.storage_service.DocumentStorageService") as MockStorage,
            patch("openai.AsyncOpenAI"),
            patch("services.document_service.DigitizedText", FakeDigitizedText),
            patch("services.task_manager.task_manager.submit", return_value="TASK_X"),
            # Extraction auto-submits a tree build; it is durable (Temporal)
            # now and irrelevant to page-parallelism assertions.
            patch("services.stage_dispatch.submit_stage", new_callable=AsyncMock),
            patch(
                "services.export_service.export_service.cache_ocr_exports_after_extract",
                new_callable=AsyncMock,
            ),
            patch("data.repositories.DocumentRepository") as MockRepo,
        ):
            MockPdfExtractor.return_value._doc = MagicMock()
            MockRepo.return_value.count_elements.return_value = 0
            MockStorage.return_value.save_page_result = MagicMock(
                side_effect=lambda document_id, page_result, page_type: page_store.append(
                    (page_result.page_number, page_result.markdown)
                )
            )

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


class TestDegenerateOcrPageIsSkipped:
    @pytest.mark.asyncio
    async def test_one_degenerate_page_does_not_fail_the_job(self):
        from services.document_service import DocumentService
        from services.extractors.ocr_extractor import (
            DegenerateOcrError,
            degenerate_ocr_placeholder,
        )

        created_instances = []
        page_store = []
        unified_saves = []

        class FakeOcrExtractor:
            def __init__(self, client, file_path):
                self.client = client
                self.file_path = file_path
                self.page_result = None
                created_instances.append(self)

            async def extract_page(self, page_number, **_kwargs):
                if page_number == 2:
                    raise DegenerateOcrError("Degenerate OCR output detected (repetition loop)")
                result = MagicMock()
                result.page_number = page_number
                result.markdown = f"page{page_number}"
                result.layout_elements = []
                self.page_result = result
                return []

        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_doc.source_language = "en"
        mock_doc.quality_report = None

        def query_side_effect(*args):
            q = MagicMock()
            if len(args) == 2:
                q.filter.return_value.order_by.return_value.all.return_value = sorted(page_store)
            else:
                model = args[0] if args else None
                if model is FakeDigitizedText:
                    q.filter.return_value.order_by.return_value.first.return_value = None
                    q.filter.return_value.first.return_value = None
                else:
                    q.filter.return_value.first.return_value = mock_doc
                    q.filter.return_value.all.return_value = []
                    q.filter.return_value.order_by.return_value.all.return_value = []
            return q

        mock_session.query.side_effect = query_side_effect
        mock_db_manager = MagicMock()
        mock_db_manager.session.side_effect = lambda: _session_cm(mock_session)

        captured = {}

        class FakeDigitizedText:
            document_id = MagicMock()
            created_at = MagicMock()

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def __setattr__(self, key, value):
                captured[key] = value
                object.__setattr__(self, key, value)

        with (
            patch(
                "services.extractors.docling_pdf_extractor.DoclingPdfExtractor"
            ) as MockPdfExtractor,
            patch(
                "services.extractors.docling_pdf_extractor.classify_pages",
                return_value={1: "scanned", 2: "scanned", 3: "scanned"},
            ),
            patch("services.extractors.ocr_extractor.OcrExtractor", FakeOcrExtractor),
            patch("services.storage_service.DocumentStorageService") as MockStorage,
            patch("openai.AsyncOpenAI"),
            patch("services.document_service.DigitizedText", FakeDigitizedText),
            patch("services.task_manager.task_manager.submit", return_value="TASK_X"),
            patch("services.stage_dispatch.submit_stage", new_callable=AsyncMock),
            patch(
                "services.export_service.export_service.cache_ocr_exports_after_extract",
                new_callable=AsyncMock,
            ),
            patch("data.repositories.DocumentRepository") as MockRepo,
        ):
            MockPdfExtractor.return_value._doc = MagicMock()
            MockRepo.return_value.count_elements.return_value = 0
            MockStorage.return_value.save_page_result = MagicMock(
                side_effect=lambda document_id, page_result, page_type: page_store.append(
                    (page_result.page_number, page_result.markdown)
                )
            )

            def _save_unified(**kwargs):
                unified_saves.append(kwargs)
                page_store.append((kwargs["page_number"], kwargs["markdown_content"]))

            MockStorage.return_value.save_unified_elements = MagicMock(
                side_effect=lambda **kwargs: _save_unified(**kwargs)
            )

            svc = DocumentService()
            result = await svc._run_extraction_body(
                "DOC_TEST",
                file_path="/tmp/fake.pdf",
                fmt="pdf",
                total_pages=3,
                task_id=None,
                db_manager=mock_db_manager,
            )

        assert result["has_ocr_failures"] is True
        assert result["failed_ocr_pages"] == [2]
        assert result["pages_processed"] == 3
        assert len(created_instances) == 3
        assert unified_saves[0]["page_number"] == 2
        assert unified_saves[0]["page_type"] == "ocr_failed"
        assert unified_saves[0]["markdown_content"] == degenerate_ocr_placeholder(2)

        ocr_content = captured.get("ocr_content") or ""
        assert "page1" in ocr_content
        assert "page3" in ocr_content
        assert degenerate_ocr_placeholder(2) in ocr_content
