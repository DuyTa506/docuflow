"""Temporal activities for the OCR/extraction workflow.

Resume model: extraction persists each page as it completes; on a retry
(activity attempt > 1) `_run_extraction(resume=True)` keeps existing
artifacts and skips already-stored pages, so a crash at page 650 of a
700-page book re-OCRs only the missing tail.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from temporalio import activity

from data.database import get_db_manager
from workflows.activities._common import _with_heartbeat


@dataclass
class ExtractionRunInput:
    document_id: str
    parent_task_id: str


@activity.defn(name="run_extraction")
async def run_extraction_activity(inp: ExtractionRunInput) -> dict[str, Any]:
    from services.document_service import DocumentService

    resume = activity.info().attempt > 1
    if resume:
        activity.logger.info(
            "Extraction retry for %s — resuming from stored pages", inp.document_id
        )

    return await _with_heartbeat(
        DocumentService()._run_extraction(
            inp.document_id,
            task_id=inp.parent_task_id,
            resume=resume,
            # Temporal owns retries: a failed attempt must not mark the doc
            # FAILED (the digest/translation gates treat that as terminal);
            # fail_extraction_activity marks it after retries are exhausted.
            mark_failed_on_error=False,
        )
    )


@activity.defn(name="finalize_extraction")
async def finalize_extraction_activity(inp: ExtractionRunInput, meta: dict = None) -> dict:
    from data.db_models import Task

    with get_db_manager().session() as db:
        task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
        if task:
            task.status = "COMPLETED"
            task.progress = 100
            task.result = meta or {}
            task.message = "Extraction completed"
            task.updated_at = datetime.utcnow()
    return meta or {}


@activity.defn(name="fail_extraction")
async def fail_extraction_activity(inp: ExtractionRunInput, error: str) -> None:
    from data.db_models import Document, Task

    with get_db_manager().session() as db:
        doc = db.query(Document).filter(Document.id == inp.document_id).first()
        if doc:
            doc.processing_status = "FAILED"
        task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
        if task:
            task.status = "FAILED"
            task.error = error[:2000]
            task.updated_at = datetime.utcnow()
