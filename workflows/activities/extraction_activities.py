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
from workflows.activities.stage_rerun_activities import _progress_probe


@dataclass
class ExtractionRunInput:
    document_id: str
    parent_task_id: str


@activity.defn(name="run_extraction")
async def run_extraction_activity(inp: ExtractionRunInput) -> dict[str, Any]:
    from services.document_service import DocumentService

    resume = activity.info().attempt > 1
    attempt = activity.info().attempt
    if resume:
        activity.logger.info(
            "Extraction retry for %s — resuming from stored pages", inp.document_id
        )

    try:
        from services.gpu_lease import RESOURCE_DOCLING, gpu_lease

        async with gpu_lease(RESOURCE_DOCLING, f"extract:{inp.document_id}"):
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
                stall_probe=_progress_probe(inp.parent_task_id),
                stall_timeout=45 * 60,
            )
    finally:
        # Docling leaves layout/TableFormer/CodeFormula in PyTorch's pool once
        # it finishes. Not releasing them starves vLLM OCR of the memory it needs
        # to start — that crash-looped the backend 447 times on 2026-08-06. In a
        # `finally` because a failed run has already loaded the models too.
        from utils.gpu_memory import release_cached_gpu_memory

        release_cached_gpu_memory()


@activity.defn(name="finalize_extraction")
async def finalize_extraction_activity(inp: ExtractionRunInput, meta: dict = None) -> dict:
    from services.task_manager import TaskManager

    with get_db_manager().session() as db:
        TaskManager.mark_terminal(
            db,
            inp.parent_task_id,
            status="COMPLETED",
            result=meta or {},
            message="Extraction completed",
            commit=False,
        )
    return meta or {}


@activity.defn(name="fail_extraction")
async def fail_extraction_activity(inp: ExtractionRunInput, error: str) -> None:
    from data.db_models import Document
    from services.task_manager import TaskManager

    with get_db_manager().session() as db:
        doc = db.query(Document).filter(Document.id == inp.document_id).first()
        if doc:
            doc.processing_status = "FAILED"
        TaskManager.mark_terminal(
            db,
            inp.parent_task_id,
            status="FAILED",
            error=error[:2000],
            commit=False,
        )
