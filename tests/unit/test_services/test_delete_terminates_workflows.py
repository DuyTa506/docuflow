"""Deleting a document must kill its running Temporal workflows.

Before this, DELETE /documents/{id} only removed DB rows + storage: a
running OCR kept burning GPU on a nonexistent document, and orphaned
digest/translation workflows hit "Document not found" (a retryable
ValueError) — which the extraction wait-gates would retry forever.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError


class TestEnsureExtractedMissingDocIsTerminal:
    @pytest.mark.asyncio
    async def test_missing_document_raises_non_retryable(self):
        from services.pipeline.stage_runners import ensure_extracted

        repo = MagicMock()
        repo.get.return_value = None
        with patch("data.repositories.DocumentRepository", return_value=repo):
            with pytest.raises(ApplicationError) as exc:
                await ensure_extracted("DOC_GONE")
        assert exc.value.non_retryable is True


class TestTerminateDocumentWorkflows:
    @pytest.mark.asyncio
    async def test_terminates_extraction_digest_and_all_translations(self):
        from services.pipeline import temporal_client as tc

        langs = [("vi",), ("en",)]
        session = MagicMock()
        session.query.return_value.filter.return_value.distinct.return_value.all.return_value = (
            langs
        )
        manager = MagicMock()
        manager.session.return_value.__enter__ = MagicMock(return_value=session)
        manager.session.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(tc, "get_db_manager", return_value=manager),
            patch.object(tc, "terminate_running_extraction", new=AsyncMock()) as t_ex,
            patch.object(tc, "terminate_running_digest", new=AsyncMock()) as t_dig,
            patch.object(tc, "terminate_running_translation", new=AsyncMock()) as t_tr,
        ):
            await tc.terminate_document_workflows("DOC_001")

        t_ex.assert_awaited_once_with("DOC_001")
        t_dig.assert_awaited_once_with("DOC_001")
        assert t_tr.await_count == 2

    @pytest.mark.asyncio
    async def test_one_failing_terminate_does_not_block_the_rest(self):
        """Best-effort: Temporal being briefly unreachable for one workflow
        must not abort the delete or skip the remaining terminations."""
        from services.pipeline import temporal_client as tc

        session = MagicMock()
        session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []
        manager = MagicMock()
        manager.session.return_value.__enter__ = MagicMock(return_value=session)
        manager.session.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(tc, "get_db_manager", return_value=manager),
            patch.object(
                tc, "terminate_running_extraction", new=AsyncMock(side_effect=RuntimeError("rpc"))
            ),
            patch.object(tc, "terminate_running_digest", new=AsyncMock()) as t_dig,
            patch.object(tc, "terminate_running_translation", new=AsyncMock()),
        ):
            await tc.terminate_document_workflows("DOC_001")

        t_dig.assert_awaited_once_with("DOC_001")
