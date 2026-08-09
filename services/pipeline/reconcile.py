"""Reconcile Temporal-owned Task rows against Temporal's own state.

Liveness for this work lives in Temporal, not in the API process. The startup
orphan sweep therefore skips these rows (killing them on an API restart would
abort healthy worker-side runs) — which left nothing at all to close a row
whose workflow died without reaching its finalize step: worker OOM, terminate,
retries exhausted, or history aged past the retention TTL.

Symptom: a document stuck "RUNNING 75%" in the UI for eleven days.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence

from config.settings import settings

logger = logging.getLogger(__name__)

_LIVE = ("RUNNING", "CONTINUED_AS_NEW")
_DEAD = ("FAILED", "TERMINATED", "TIMED_OUT", "CANCELED")


class ReconcileAction(Enum):
    LEAVE = "leave"
    COMPLETE = "complete"
    FAIL = "fail"


def reconcile_decision(
    workflow_status: Optional[str],
    *,
    row_age_hours: float,
    retention_hours: Optional[float] = None,
) -> ReconcileAction:
    """What to do with an open Task row given its workflow's status.

    ``workflow_status`` is None when Temporal has no such workflow. That is
    ambiguous — it means either "never started" (a race with the row being
    created moments ago) or "closed and aged out of history". Only past the
    retention TTL is it safe to read as dead.
    """
    if workflow_status in _LIVE:
        return ReconcileAction.LEAVE
    if workflow_status == "COMPLETED":
        return ReconcileAction.COMPLETE
    if workflow_status in _DEAD:
        return ReconcileAction.FAIL

    ttl = settings.temporal_retention_hours if retention_hours is None else retention_hours
    return ReconcileAction.FAIL if row_age_hours > ttl else ReconcileAction.LEAVE


def workflow_ids_for_task(
    task_type: str,
    document_id: str,
    translation_languages: Sequence[str],
) -> list[str]:
    """Candidate workflow ids for an open row of *task_type*.

    A TRANSLATE row carries no target language, so every language the document
    has is a candidate; the rest map one-to-one.
    """
    from services.pipeline.temporal_client import (
        extraction_workflow_id,
        translation_workflow_id,
        workflow_id_for_document,
    )
    from services.stage_dispatch import STAGE_RUNNERS, stage_workflow_id

    if task_type == "DIGEST_PIPELINE":
        return [workflow_id_for_document(document_id)]
    if task_type == "EXTRACT":
        return [extraction_workflow_id(document_id)]
    if task_type == "TRANSLATE":
        return [translation_workflow_id(document_id, lang) for lang in translation_languages]
    if task_type in STAGE_RUNNERS:
        return [stage_workflow_id(document_id, task_type)]
    return []


def aggregate_status(statuses: Sequence[Optional[str]]) -> Optional[str]:
    """Collapse several candidate workflows into one verdict.

    Any live workflow means the row is alive; otherwise a completed one beats a
    missing one, since "missing" is the weakest evidence we have.
    """
    for status in statuses:
        if status in _LIVE:
            return status
    for status in statuses:
        if status == "COMPLETED":
            return status
    for status in statuses:
        if status in _DEAD:
            return status
    return None


async def _workflow_status(workflow_id: str) -> Optional[str]:
    from temporalio.service import RPCError, RPCStatusCode

    from services.pipeline.temporal_client import get_temporal_client

    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    try:
        desc = await handle.describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return None
        raise
    return desc.status.name if desc.status is not None else None


async def reconcile_document_tasks(document_id: str) -> int:
    """Reconcile this document's open Temporal-owned rows. Returns rows changed.

    Called lazily from the status endpoints, so a stuck row is corrected
    exactly when somebody looks at it, and on worker startup for the rest.
    """
    from data.database import get_db_manager
    from data.db_models import Document, Task, Translation
    from services.task_manager import temporal_owned_task_types

    owned = temporal_owned_task_types()
    changed = 0
    now = datetime.utcnow()

    with get_db_manager().session() as db:
        open_rows = (
            db.query(Task)
            .filter(
                Task.document_id == document_id,
                Task.status.in_(("PENDING", "RUNNING")),
                Task.task_type.in_(tuple(owned)),
            )
            .all()
        )
        if not open_rows:
            return 0
        languages = [
            row[0]
            for row in db.query(Translation.target_language)
            .filter(Translation.document_id == document_id)
            .distinct()
            .all()
        ]
        pending = [(t.id, t.task_type, t.updated_at or t.created_at) for t in open_rows]

    verdicts: dict[str, ReconcileAction] = {}
    for task_id, task_type, stamp in pending:
        wf_ids = workflow_ids_for_task(task_type, document_id, languages)
        try:
            statuses = [await _workflow_status(wf_id) for wf_id in wf_ids]
        except Exception as exc:
            # Temporal unreachable: leave every row alone rather than mass-fail
            # healthy work on a transient outage.
            logger.warning("Reconcile skipped for %s: %s", document_id, exc)
            return 0
        age_hours = ((now - stamp).total_seconds() / 3600.0) if stamp else 0.0
        verdicts[task_id] = reconcile_decision(aggregate_status(statuses), row_age_hours=age_hours)

    with get_db_manager().session() as db:
        for task_id, action in verdicts.items():
            if action is ReconcileAction.LEAVE:
                continue
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task or task.status not in ("PENDING", "RUNNING"):
                continue
            if action is ReconcileAction.COMPLETE:
                task.status = "COMPLETED"
                task.progress = 100
            else:
                task.status = "FAILED"
                task.error = (
                    (task.error or "") + "\nWorkflow no longer running (reconciled)."
                ).strip()
            task.updated_at = now
            changed += 1

            # The digest mirror is what the UI actually polls — a closed parent
            # task with pipeline_state still RUNNING keeps the spinner going.
            if task.task_type == "DIGEST_PIPELINE":
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc and doc.pipeline_state in ("PENDING", "RUNNING"):
                    doc.pipeline_state = "DONE" if action is ReconcileAction.COMPLETE else "FAILED"
                    doc.pipeline_message = (
                        "Pipeline completed"
                        if action is ReconcileAction.COMPLETE
                        else "Pipeline stopped without finishing (reconciled)"
                    )
                    doc.updated_at = now
        db.commit()

    if changed:
        logger.info("Reconciled %d stale task row(s) for %s", changed, document_id)
    return changed


async def reconcile_all_open_tasks() -> int:
    """Sweep every document with open Temporal-owned rows (worker startup)."""
    from data.database import get_db_manager
    from data.db_models import Task
    from services.task_manager import temporal_owned_task_types

    owned = temporal_owned_task_types()
    with get_db_manager().session() as db:
        document_ids = [
            row[0]
            for row in db.query(Task.document_id)
            .filter(
                Task.status.in_(("PENDING", "RUNNING")),
                Task.task_type.in_(tuple(owned)),
                Task.document_id.isnot(None),
            )
            .distinct()
            .all()
        ]

    total = 0
    for document_id in document_ids:
        try:
            total += await reconcile_document_tasks(document_id)
        except Exception as exc:
            logger.warning("Reconcile failed for %s: %s", document_id, exc)
    return total
