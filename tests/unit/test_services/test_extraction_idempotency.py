"""Extraction retries must not duplicate pages / DigitizedText or poison EXTRACTED.

Live regression (DOC_013): a Temporal retry after EXTRACTED + export crash
inserted a second DigitizedText row and could re-insert pages. Export failure
must fail the task only when pages + text already exist.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.db_models import DigitizedText, Document, Page
from services.document_service import DocumentService
from workflows.activities.extraction_activities import (
    ExtractionRunInput,
    fail_extraction_activity,
)


def _session_cm(session):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=session)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestFailExtractionPreservesExtracted:
    @pytest.mark.asyncio
    async def test_does_not_mark_failed_when_pages_and_text_exist(self):
        doc = MagicMock()
        doc.processing_status = "EXTRACTED"
        session = MagicMock()

        def _query(*entities):
            q = MagicMock()
            target = entities[0] if entities else None
            if target is Document:
                q.filter.return_value.first.return_value = doc
            else:
                # Page.id / DigitizedText.id existence probes
                q.filter.return_value.first.return_value = ("row",)
            return q

        session.query.side_effect = _query
        dbm = MagicMock()
        dbm.session.return_value = _session_cm(session)

        with (
            patch(
                "workflows.activities.extraction_activities.get_db_manager",
                return_value=dbm,
            ),
            patch("services.task_manager.TaskManager.mark_terminal"),
            patch("services.pipeline.job_queue.kick_queue"),
        ):
            await fail_extraction_activity(
                ExtractionRunInput("DOC_013", "TSK_1", resume=True),
                error="export boom",
            )

        assert doc.processing_status == "EXTRACTED"

    @pytest.mark.asyncio
    async def test_marks_failed_when_artifacts_missing(self):
        doc = MagicMock()
        doc.processing_status = "EXTRACT_IN_PROGRESS"
        session = MagicMock()

        def _query(model):
            q = MagicMock()
            if model is Document:
                q.filter.return_value.first.return_value = doc
            else:
                q.filter.return_value.first.return_value = None
            return q

        session.query.side_effect = _query
        dbm = MagicMock()
        dbm.session.return_value = _session_cm(session)

        with (
            patch(
                "workflows.activities.extraction_activities.get_db_manager",
                return_value=dbm,
            ),
            patch("services.task_manager.TaskManager.mark_terminal"),
            patch("services.pipeline.job_queue.kick_queue"),
        ):
            await fail_extraction_activity(
                ExtractionRunInput("DOC_X", "TSK_1"),
                error="ocr boom",
            )

        assert doc.processing_status == "FAILED"


class TestExtractedResumeExportOnly:
    @pytest.mark.asyncio
    async def test_resume_after_extracted_skips_body_and_rebuilds_exports(self):
        doc = MagicMock()
        doc.processing_status = "EXTRACTED"
        doc.file_path = "documents/DOC_013/file.pdf"
        doc.format = "pdf"
        doc.total_pages = 3

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = doc
        dbm = MagicMock()
        dbm.session.return_value = _session_cm(session)

        repo = MagicMock()
        repo.get_digitized_text.return_value = MagicMock()
        repo.count_pages.return_value = 3
        repo.count_elements.return_value = 10

        cache = AsyncMock()
        with (
            patch("services.document_service.get_db_manager", return_value=dbm),
            patch(
                "data.repositories.DocumentRepository",
                return_value=repo,
            ),
            patch(
                "services.export_service.export_service.invalidate_ocr_exports"
            ) as inv,
            patch(
                "services.export_service.export_service.cache_ocr_exports_after_extract",
                cache,
            ),
            patch("services.document_service.TaskManager.update_progress"),
            patch.object(
                DocumentService,
                "_run_extraction_body",
                new_callable=AsyncMock,
            ) as body,
        ):
            result = await DocumentService()._run_extraction(
                "DOC_013",
                task_id="TSK_1",
                resume=True,
                mark_failed_on_error=False,
                attempt=2,
            )

        body.assert_not_awaited()
        inv.assert_called_once_with("DOC_013")
        cache.assert_awaited()
        assert doc.processing_status == "EXTRACTED"
        assert result["pages_processed"] == 3


class TestPageUpsert:
    def test_save_unified_elements_updates_existing_page(self):
        from services.storage_service import DocumentStorageService

        existing = Page(
            id="PG_OLD",
            document_id="DOC_1",
            page_number=2,
            markdown_content="old",
            page_type="text",
        )
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = existing

        storage = DocumentStorageService(session)
        with (
            patch.object(storage, "_upload_page_image", return_value=(None, None)),
            patch.object(storage, "_replace_layout_elements") as replace,
        ):
            page = storage.save_unified_elements(
                document_id="DOC_1",
                page_number=2,
                markdown_content="new md",
                layout_dicts=[{"label": "text", "text_content": "hi", "bbox_x1": 0}],
                page_type="text",
            )

        assert page is existing
        assert existing.markdown_content == "new md"
        session.add.assert_not_called()
        replace.assert_called_once()
        session.commit.assert_called()
