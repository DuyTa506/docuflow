"""processing_status=FAILED must be terminal, not per-attempt.

Regression (DOC_068): `_run_extraction` marked the document FAILED on every
failed attempt while the Temporal ExtractionWorkflow was still retrying
(3 attempts, 1-2 min apart). The digest/translation extraction gates treat
FAILED as non-retryable, so a transient attempt-1 blip permanently killed
the digest of a document whose extraction later succeeded. In Temporal mode
only fail_extraction_activity (after retries are exhausted) may mark FAILED.
"""

import inspect
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from services.document_service import DocumentService


def _fake_db(doc):
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = doc

    manager = MagicMock()

    @contextmanager
    def cm():
        yield session

    manager.session = cm
    return manager


@pytest.mark.asyncio
async def test_default_marks_failed_for_legacy_path():
    doc = MagicMock()
    doc.processing_status = "INIT"
    svc = DocumentService()
    with (
        patch("services.document_service.get_db_manager", return_value=_fake_db(doc)),
        patch.object(svc, "_run_extraction_body", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            await svc._run_extraction("DOC_X")
    assert doc.processing_status == "FAILED"


@pytest.mark.asyncio
async def test_temporal_mode_leaves_status_untouched_on_attempt_failure():
    doc = MagicMock()
    doc.processing_status = "INIT"
    svc = DocumentService()
    with (
        patch("services.document_service.get_db_manager", return_value=_fake_db(doc)),
        patch.object(svc, "_run_extraction_body", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            await svc._run_extraction("DOC_X", mark_failed_on_error=False)
    # stays in-progress (EXTRACT_IN_PROGRESS) so the digest/translation
    # gates keep waiting through Temporal's remaining retries
    assert doc.processing_status != "FAILED"


def test_temporal_activity_disables_per_attempt_marking():
    from workflows.activities import extraction_activities

    src = inspect.getsource(extraction_activities)
    assert "mark_failed_on_error=False" in src
