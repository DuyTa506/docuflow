"""Soft overflow wait queue: start Temporal immediately under the safety ceiling.

Under host/user soft caps, HTTP submit starts Temporal directly. Only overflow
rows carry ``progress_meta.queued``; ``dispatch_waiting`` unqueues the oldest
eligible waiter when a slot frees. Real GPU/LLM scheduling is Docling
``gpu_lease``, ``AI_MAX_CONCURRENT_REQUESTS``, and vLLM ``max-num-seqs``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from config.capacity import (
    HEAVY_STAGE_TYPES,
    SLOT_DIGEST,
    SLOT_EXTRACT,
    SLOT_TRANSLATE,
    capacity_profile,
)
from services.pipeline.admission import (
    AdmissionRejected,
    assert_can_admit,
    count_user_open,
    is_queued,
    mark_dispatched,
    mark_queued,
    queue_wait_message,
)

logger = logging.getLogger(__name__)

_DISPATCH_TYPES = {
    SLOT_TRANSLATE: ("TRANSLATE",),
    SLOT_EXTRACT: ("EXTRACT",),
    SLOT_DIGEST: ("DIGEST_PIPELINE", *sorted(HEAVY_STAGE_TYPES)),
}

_locks: dict[str, asyncio.Lock] = {}

# Stable advisory-lock keys per slot (Postgres pg_advisory_xact_lock).
_ADVISORY_KEYS = {
    SLOT_TRANSLATE: 0xD0CF_7101,
    SLOT_EXTRACT: 0xD0CF_7102,
    SLOT_DIGEST: 0xD0CF_7103,
}


def _lock_for(slot: str) -> asyncio.Lock:
    lock = _locks.get(slot)
    if lock is None:
        lock = asyncio.Lock()
        _locks[slot] = lock
    return lock


def _advisory_lock(db, slot: str) -> None:
    """Cross-process mutex for drain (API + Temporal workers share Postgres).

    SQLite / non-Postgres engines used in unit tests are a no-op.
    """
    key = _ADVISORY_KEYS.get(slot)
    if key is None:
        return
    bind = db.get_bind() if hasattr(db, "get_bind") else getattr(db, "bind", None)
    dialect = getattr(getattr(bind, "dialect", None), "name", "") if bind else ""
    if dialect != "postgresql":
        return
    from sqlalchemy import text

    db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})


def kick_queue(slot: str) -> None:
    """Schedule a drain on the running loop (API request or Temporal activity)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No event loop to dispatch %s queue", slot)
        return
    loop.create_task(_dispatch_safe(slot))


async def _dispatch_safe(slot: str) -> None:
    try:
        await dispatch_waiting(slot)
    except Exception:
        logger.exception("Failed to dispatch waiting %s jobs", slot)


async def drain_waiting_queues() -> None:
    for slot in (SLOT_TRANSLATE, SLOT_EXTRACT, SLOT_DIGEST):
        await dispatch_waiting(slot)


async def start_or_enqueue(
    db,
    *,
    slot: str,
    task,
    fairness_key: Optional[str],
    start: Callable[[], Awaitable[Any]],
    extra_meta: Optional[dict] = None,
) -> bool:
    """Start Temporal now if under soft ceiling; else queue and kick.

    Returns True if ``start`` was awaited, False if the task was enqueued.
    Caller must have already inserted ``task`` and committed (or flushed) it.
    """
    try:
        assert_can_admit(db, slot, user_id=fairness_key, excluding_task_id=task.id)
    except AdmissionRejected as exc:
        meta = dict(extra_meta or {})
        if fairness_key is not None:
            meta.setdefault("fairness_key", fairness_key)
        mark_queued(task, extra=meta, message=queue_wait_message(exc))
        db.commit()
        kick_queue(slot)
        return False
    await start()
    return True


async def dispatch_waiting(slot: str) -> None:
    """Start as many queued overflow jobs of this slot as soft capacity allows.

    A waiter blocked only by the per-user cap is skipped (not a hard stop) so
    other users' jobs keep draining — head-of-line blocking used to freeze the
    whole queue behind one saturated teacher.
    """
    async with _lock_for(slot):
        skipped: set[str] = set()
        while True:
            payload = _claim_next(slot, skip_task_ids=skipped)
            if payload is None:
                return
            if payload.get("skip"):
                continue
            task_id = payload.get("parent_task_id")
            try:
                await _start_claimed(payload)
            except AdmissionRejected:
                _requeue(payload)
                if task_id:
                    skipped.add(task_id)
                continue
            except Exception:
                logger.exception(
                    "Starting queued %s job %s failed; leaving it waiting",
                    slot,
                    task_id,
                )
                _requeue(payload)
                if task_id:
                    skipped.add(task_id)
                continue


def _user_at_cap(db, user_id: Optional[str], *, excluding_task_id: Optional[str]) -> bool:
    if not user_id:
        return False
    cap = capacity_profile()
    return count_user_open(db, user_id, excluding_task_id=excluding_task_id) >= cap.max_jobs_per_user


def _claim_next(slot: str, *, skip_task_ids: Optional[set[str]] = None) -> Optional[dict[str, Any]]:
    from data.database import get_db_manager
    from data.db_models import Document, Task, Translation

    types = _DISPATCH_TYPES.get(slot)
    if not types:
        return None
    skip_task_ids = skip_task_ids or set()
    db_manager = get_db_manager()
    with db_manager.session() as db:
        _advisory_lock(db, slot)
        try:
            assert_can_admit(db, slot)
        except AdmissionRejected:
            return None
        waiters = (
            db.query(Task)
            .filter(Task.task_type.in_(types), Task.status == "PENDING")
            .order_by(Task.created_at.asc())
            .all()
        )
        task = None
        for candidate in waiters:
            if not is_queued(candidate):
                continue
            if candidate.id in skip_task_ids:
                continue
            meta = (
                dict(candidate.progress_meta)
                if isinstance(candidate.progress_meta, dict)
                else {}
            )
            fairness_key = meta.get("fairness_key")
            if not fairness_key and candidate.document_id:
                doc = db.query(Document).filter(Document.id == candidate.document_id).first()
                if doc is not None:
                    fairness_key = doc.user_id
            if _user_at_cap(db, fairness_key, excluding_task_id=candidate.id):
                continue
            task = candidate
            break
        if task is None:
            return None
        meta = dict(task.progress_meta) if isinstance(task.progress_meta, dict) else {}
        mark_dispatched(task)
        payload: dict[str, Any] = {
            "slot": slot,
            "task_type": task.task_type,
            "document_id": task.document_id,
            "parent_task_id": task.id,
            "fairness_key": meta.get("fairness_key"),
            "target_language": meta.get("target_language"),
            "domain": meta.get("domain") or "general",
            "translation_id": meta.get("translation_id"),
            "stage_options": meta.get("stage_options"),
        }
        if not payload["fairness_key"] and task.document_id:
            doc = db.query(Document).filter(Document.id == task.document_id).first()
            if doc is not None:
                payload["fairness_key"] = doc.user_id
        if slot == SLOT_TRANSLATE and not payload["translation_id"]:
            trans = (
                db.query(Translation)
                .filter(
                    Translation.document_id == task.document_id,
                    Translation.status.in_(("PENDING", "IN_PROGRESS")),
                )
                .order_by(Translation.created_at.desc())
                .first()
            )
            if trans is not None:
                payload["translation_id"] = trans.id
                payload["target_language"] = payload["target_language"] or trans.target_language
        if slot == SLOT_TRANSLATE and not payload["translation_id"]:
            logger.error(
                "Queued TRANSLATE %s has no translation row; failing it",
                task.id,
            )
            from services.task_manager import TaskManager

            TaskManager.mark_terminal(
                db,
                task.id,
                status="FAILED",
                error="Queued translation missing translation_id",
                commit=False,
            )
            db.commit()
            return {"slot": slot, "skip": True}
        db.commit()
        return payload


def _requeue(payload: dict[str, Any]) -> None:
    from data.database import get_db_manager
    from data.db_models import Task

    task_id = payload.get("parent_task_id")
    if not task_id or payload.get("skip"):
        return
    extra = {
        "fairness_key": payload.get("fairness_key"),
        "target_language": payload.get("target_language"),
        "domain": payload.get("domain"),
        "translation_id": payload.get("translation_id"),
        "stage_options": payload.get("stage_options"),
    }
    with get_db_manager().session() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None or task.status not in ("PENDING", "RUNNING"):
            return
        mark_queued(task, extra=extra)
        db.commit()


async def _start_claimed(payload: dict[str, Any]) -> None:
    if payload.get("skip"):
        return
    from services.pipeline.temporal_client import (
        start_digest_workflow,
        start_extraction_workflow,
        start_stage_workflow,
        start_translation_workflow,
    )

    slot = payload["slot"]
    if slot == SLOT_TRANSLATE:
        await start_translation_workflow(
            document_id=payload["document_id"],
            translation_id=payload["translation_id"],
            parent_task_id=payload["parent_task_id"],
            target_language=payload["target_language"] or "vi",
            domain=payload.get("domain") or "general",
            fairness_key=payload.get("fairness_key"),
        )
        return
    if slot == SLOT_EXTRACT:
        await start_extraction_workflow(
            document_id=payload["document_id"],
            parent_task_id=payload["parent_task_id"],
            fairness_key=payload.get("fairness_key"),
        )
        return
    if slot == SLOT_DIGEST:
        task_type = payload.get("task_type") or "DIGEST_PIPELINE"
        if task_type in HEAVY_STAGE_TYPES:
            await start_stage_workflow(
                document_id=payload["document_id"],
                stage=task_type,
                task_id=payload["parent_task_id"],
                options=payload.get("stage_options"),
                fairness_key=payload.get("fairness_key"),
            )
            return
        await start_digest_workflow(
            payload["document_id"],
            fairness_key=payload.get("fairness_key"),
            parent_task_id=payload["parent_task_id"],
        )


async def submit_digest(
    db, document_id: str, fairness_key: str | None = None
) -> tuple[str, str, bool]:
    """Create (or reuse) a DIGEST_PIPELINE row; start Temporal under soft ceiling.

    Returns (workflow_id, parent_task_id, reused). Overflow only is queued.
    """
    from data.db_models import Task
    from services.pipeline.stage_runners import create_parent_task
    from services.pipeline.temporal_client import start_digest_workflow, workflow_id_for_document

    existing = (
        db.query(Task)
        .filter(
            Task.document_id == document_id,
            Task.task_type == "DIGEST_PIPELINE",
            Task.status.in_(("PENDING", "RUNNING")),
        )
        .order_by(Task.created_at.desc())
        .first()
    )
    wf_id = workflow_id_for_document(document_id)
    if existing:
        if is_queued(existing):
            kick_queue(SLOT_DIGEST)
        return wf_id, existing.id, True

    parent_task_id = create_parent_task(db, document_id)
    task = db.query(Task).filter(Task.id == parent_task_id).first()
    if task is not None:
        task.message = "Đang khởi chạy…"
        if fairness_key:
            meta = dict(task.progress_meta) if isinstance(task.progress_meta, dict) else {}
            meta["fairness_key"] = fairness_key
            task.progress_meta = meta
        db.commit()

        async def _start():
            await start_digest_workflow(
                document_id,
                fairness_key=fairness_key,
                parent_task_id=parent_task_id,
            )

        await start_or_enqueue(
            db,
            slot=SLOT_DIGEST,
            task=task,
            fairness_key=fairness_key,
            start=_start,
            extra_meta={"fairness_key": fairness_key},
        )
    return wf_id, parent_task_id, False
