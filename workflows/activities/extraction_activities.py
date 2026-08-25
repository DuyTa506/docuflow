"""Temporal activities for the OCR/extraction workflow.

Resume model: extraction persists each page as it completes; on a retry
(activity attempt > 1) `_run_extraction(resume=True)` keeps existing
artifacts and skips already-stored pages, so a crash at page 650 of a
700-page book re-OCRs only the missing tail.
"""

from dataclasses import dataclass
from typing import Any

from temporalio import activity

from data.database import get_db_manager
from workflows.activities._common import _with_heartbeat


@dataclass
class ExtractionRunInput:
    document_id: str
    parent_task_id: str
    # True for a new workflow that retries a previously FAILED document.
    # Activity attempts > 1 resume independently below.
    resume: bool = False


@activity.defn(name="run_extraction")
async def run_extraction_activity(inp: ExtractionRunInput) -> dict[str, Any]:
    from services.document_service import DocumentService

    resume = inp.resume or activity.info().attempt > 1
    attempt = activity.info().attempt
    if resume:
        activity.logger.info(
            "Extraction retry for %s — resuming from stored pages", inp.document_id
        )

    try:
        # DocumentService owns the Docling CPU phase lease and the process-wide
        # vLLM request limiter.
        return await _with_heartbeat(
            DocumentService()._run_extraction(
                inp.document_id,
                task_id=inp.parent_task_id,
                resume=resume,
                attempt=attempt,
                # Temporal owns retries: a failed attempt must not mark the doc
                # FAILED (the digest/translation gates treat that as terminal);
                # fail_extraction_activity marks it after retries are exhausted.
                mark_failed_on_error=False,
            ),
        )
    except Exception as exc:
        from temporalio.exceptions import ApplicationError

        from services.extractors.ocr_extractor import DegenerateOcrError

        # Belt-and-suspenders: the page loop skips DegenerateOcrError, but if
        # one leaks, Temporal must not resume the whole book on the same page.
        if isinstance(exc, DegenerateOcrError) or type(exc).__name__ == "DegenerateOcrError":
            raise ApplicationError(
                str(exc),
                type="DegenerateOcrError",
                non_retryable=True,
            ) from exc
        raise
    finally:
        # Keep the cleanup for rollback configurations that select CUDA. In a
        # `finally` because a failed run may already have loaded the models.
        from utils.gpu_memory import release_cached_gpu_memory

        release_cached_gpu_memory()


@activity.defn(name="finalize_extraction")
async def finalize_extraction_activity(inp: ExtractionRunInput, meta: dict = None) -> dict:
    from services.task_manager import TaskManager

    meta = meta or {}
    n_failed = len(meta.get("failed_ocr_pages") or [])
    message = (
        f"Trích xuất hoàn tất — {n_failed} trang OCR lỗi" if n_failed else "Trích xuất hoàn tất"
    )
    with get_db_manager().session() as db:
        TaskManager.mark_terminal(
            db,
            inp.parent_task_id,
            status="COMPLETED",
            result=meta,
            message=message,
            commit=False,
        )
    from config.capacity import SLOT_EXTRACT
    from services.pipeline.job_queue import kick_queue

    kick_queue(SLOT_EXTRACT)
    return meta or {}


@activity.defn(name="fail_extraction")
async def fail_extraction_activity(inp: ExtractionRunInput, error: str) -> None:
    from data.db_models import DigitizedText, Document, Page
    from services.task_manager import TaskManager

    with get_db_manager().session() as db:
        doc = db.query(Document).filter(Document.id == inp.document_id).first()
        if doc:
            # Export is a side effect: if pages + DigitizedText already exist the
            # document is usable — fail the task only, do not poison EXTRACTED.
            has_pages = (
                db.query(Page.id).filter(Page.document_id == inp.document_id).first() is not None
            )
            has_text = (
                db.query(DigitizedText.id)
                .filter(DigitizedText.document_id == inp.document_id)
                .first()
                is not None
            )
            if not (has_pages and has_text):
                doc.processing_status = "FAILED"
        TaskManager.mark_terminal(
            db,
            inp.parent_task_id,
            status="FAILED",
            error=error[:2000],
            commit=False,
        )
    from config.capacity import SLOT_EXTRACT
    from services.pipeline.job_queue import kick_queue

    kick_queue(SLOT_EXTRACT)
