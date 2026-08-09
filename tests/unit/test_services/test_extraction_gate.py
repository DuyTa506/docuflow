"""Extraction gate (`ensure_extracted`) semantics for digest & translation.

A fresh upload fires digest/translation from the FE immediately, while OCR
of a book-length PDF runs for hours. The gate activity must:
- keep raising a retryable error while extraction is pending (Temporal's
  default unlimited retry turns that into a wait-loop), and
- fail FAST (non-retryable) when extraction itself FAILED — otherwise the
  workflow "waits" forever on a document that will never be extracted.
"""

from unittest.mock import MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError

from services.pipeline.stage_runners import ensure_extracted


def _patch_doc(status):
    doc = MagicMock()
    doc.processing_status = status
    repo = MagicMock()
    repo.get.return_value = doc
    return patch("data.repositories.DocumentRepository", return_value=repo)


class TestEnsureExtracted:
    @pytest.mark.asyncio
    async def test_pending_extraction_raises_retryable(self):
        with _patch_doc("INIT"):
            with pytest.raises(ValueError) as exc:
                await ensure_extracted("DOC_X")
        assert not isinstance(exc.value, ApplicationError)

    @pytest.mark.asyncio
    async def test_extracted_passes(self):
        with _patch_doc("EXTRACTED"):
            await ensure_extracted("DOC_X")

    @pytest.mark.asyncio
    async def test_failed_extraction_raises_non_retryable(self):
        """FAILED OCR must not park the workflow in an infinite wait."""
        with _patch_doc("FAILED"):
            with pytest.raises(ApplicationError) as exc:
                await ensure_extracted("DOC_X")
        assert exc.value.non_retryable is True


class TestDigestGateShowsWaitingState:
    @pytest.mark.asyncio
    async def test_mirror_reports_waiting_while_unextracted(self):
        """UI showed digest RUNNING/'Pipeline started' while OCR was still
        PENDING — the gate must surface a 'waiting for extraction' message
        on each retry so pipeline-status tells the truth."""
        from workflows.activities.digest_activities import (
            PipelineStageInput,
            ensure_extracted_activity,
        )

        inp = PipelineStageInput(document_id="DOC_X", parent_task_id="TASK_1")
        with (
            _patch_doc("INIT"),
            patch("workflows.activities.digest_activities.update_pipeline_mirror") as mirror,
        ):
            with pytest.raises(ValueError):
                await ensure_extracted_activity(inp)
        assert mirror.called
        msg = mirror.call_args.kwargs.get("message", "")
        assert "trích xuất" in msg.lower() or "ocr" in msg.lower()
