"""
In-process async background task manager.

Uses asyncio.create_task() with DB-backed status tracking.
No external task queue required.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, Optional, Union

from sqlalchemy.orm import Session

from config.settings import settings
from data.database import get_db_manager
from data.db_models import Task, Translation
from data.id_generator import IdGenerator
from data.repositories import TaskRepository
from services.eta import finish_eta, sanitize_eta, sanitize_progress_meta, update_eta

logger = logging.getLogger(__name__)

# Task types always owned by Temporal workers — they survive an API restart,
# so the startup orphan sweep must leave them alone.
TEMPORAL_TASK_TYPES = {"DIGEST_PIPELINE"}


def temporal_owned_task_types() -> set[str]:
    """Task types whose work runs in the Temporal worker, not this process.

    Derived rather than hardcoded: each type is owned by Temporal exactly when
    its routing flag is on, so flipping a flag can't leave the startup sweep
    failing rows that are alive in the worker.
    """
    owned = set(TEMPORAL_TASK_TYPES)
    if settings.translation_use_temporal:
        owned.add("TRANSLATE")
    if settings.ocr_use_temporal:
        owned.add("EXTRACT")
    if settings.stage_rerun_use_temporal:
        from services.stage_dispatch import STAGE_RUNNERS

        owned.update(STAGE_RUNNERS)
    return owned


def fail_orphaned_tasks(db: Session) -> int:
    """Fail Task/Translation rows orphaned by a process crash or restart.

    In-process asyncio tasks die with the process; any PENDING/RUNNING task
    (except Temporal-owned types) found at startup can never complete.

    TRANSLATE/EXTRACT rows are only "owned" by the API process — and thus
    genuinely orphaned by its restart — when their respective
    translation_use_temporal/ocr_use_temporal settings are off. When on
    (the default), that work runs in the separate Temporal worker process,
    which an API restart doesn't touch, so those rows must be left alone.

    Returns the number of rows failed.
    """
    now = datetime.utcnow()
    count = 0

    temporal_task_types = temporal_owned_task_types()

    orphaned_tasks = (
        db.query(Task)
        .filter(
            Task.status.in_(["PENDING", "RUNNING"]),
            Task.task_type.notin_(temporal_task_types),
        )
        .all()
    )
    for task in orphaned_tasks:
        TaskManager.mark_terminal(
            db,
            task.id,
            status="FAILED",
            error=((task.error or "") + "\nOrphaned by server restart.").strip(),
            now=now,
            commit=False,
        )
        count += 1

    if not settings.translation_use_temporal:
        orphaned_translations = (
            db.query(Translation).filter(Translation.status.in_(["PENDING", "IN_PROGRESS"])).all()
        )
        for trn in orphaned_translations:
            trn.status = "FAILED"
            count += 1

    db.commit()
    return count


class TaskManager:
    """
    Singleton task manager.

    * ``submit()``          — create DB record + launch async task
    * ``get_status()``      — read task status from DB
    * ``update_progress()`` — called by service coroutines to report %
    """

    _running_tasks: Dict[str, asyncio.Task] = {}
    _concurrency_sem: Optional[asyncio.Semaphore] = None

    @classmethod
    def _concurrency_semaphore(cls) -> asyncio.Semaphore:
        if cls._concurrency_sem is None:
            from config.settings import settings

            cls._concurrency_sem = asyncio.Semaphore(settings.max_concurrent_tasks)
        return cls._concurrency_sem

    def get_active_task_id(
        self,
        db: Session,
        document_id: Optional[str],
        task_type: str,
    ) -> Optional[str]:
        """Return an in-flight task id, or None. Marks orphaned rows as FAILED."""
        if not document_id:
            return None

        active = TaskRepository(db).find_active(document_id, task_type)
        if not active:
            return None

        if active.id in self._running_tasks:
            return active.id

        # PENDING briefly before asyncio picks it up
        if active.status == "PENDING" and active.created_at:
            age = (datetime.utcnow() - active.created_at).total_seconds()
            if age < 60:
                return active.id

        # RUNNING/PENDING in DB but no live coroutine → stale after crash/restart
        task = db.query(Task).filter(Task.id == active.id).first()
        if task:
            self.mark_terminal(
                db,
                task.id,
                status="FAILED",
                error=(task.error or "") + "\nStale task reset (no active worker).",
            )
        return None

    # ── Submit ──────────────────────────────────────────────────────

    def submit(
        self,
        db: Session,
        document_id: Optional[str],
        task_type: str,
        coro: Union[Coroutine, None] = None,
        *,
        coro_factory: Optional[Callable[[str], Coroutine]] = None,
        dedupe: bool = True,
    ) -> str:
        """
        Persist a PENDING task row, wrap *coro* with status bookkeeping,
        and launch it via ``asyncio.create_task``.

        Pass either *coro* or *coro_factory(task_id)* so the coroutine receives
        the persisted task id without a latest-task DB lookup.

        When *dedupe* is True, returns the existing task id if the same
        document already has a PENDING/RUNNING task of this type.
        """
        if coro is None and coro_factory is None:
            raise ValueError("submit() requires coro or coro_factory")
        if coro is not None and coro_factory is not None:
            raise ValueError("submit() accepts only one of coro or coro_factory")

        if dedupe:
            existing = self.get_active_task_id(db, document_id, task_type)
            if existing:
                return existing
        # Use task_type as prefix (e.g. OCR_001, SUMMARIZE_003) for readability.
        # We still use the shared "tasks" counter so IDs remain globally unique.
        raw_id = IdGenerator.next_id(db, "tasks")  # e.g. TASK_042
        seq_num = raw_id.split("_")[-1]  # e.g. 042
        task_id = f"{task_type}_{seq_num}"  # e.g. SUMMARIZE_042

        task_row = Task(
            id=task_id,
            document_id=document_id,
            task_type=task_type,
            status="PENDING",
            progress=0,
        )
        db.add(task_row)
        db.commit()

        actual_coro = coro_factory(task_id) if coro_factory else coro

        # Wrap and launch (global semaphore limits concurrent pipeline workers)
        sem = self._concurrency_semaphore()
        asyncio_task = asyncio.create_task(self._run_wrapper(task_id, actual_coro, sem))
        self._running_tasks[task_id] = asyncio_task
        return task_id

    # ── Wrapper ─────────────────────────────────────────────────────

    async def _run_wrapper(
        self,
        task_id: str,
        coro: Coroutine,
        sem: asyncio.Semaphore,
    ):
        """Execute *coro*, updating DB status on start / success / failure."""
        async with sem:
            db_manager = get_db_manager()

            # Mark RUNNING
            with db_manager.session() as db:
                self.update_progress(db, task_id, 0)

            try:
                result = await coro
                # Mark COMPLETED
                with db_manager.session() as db:
                    self.mark_terminal(
                        db,
                        task_id,
                        status="COMPLETED",
                        result=(
                            result if isinstance(result, (dict, list)) else {"detail": str(result)}
                        ),
                    )
            except Exception as exc:
                tb = traceback.format_exc()
                with db_manager.session() as db:
                    self.mark_terminal(db, task_id, status="FAILED", error=f"{exc}\n{tb}")
            finally:
                self._running_tasks.pop(task_id, None)

    # ── Query ───────────────────────────────────────────────────────

    def get_status(self, db: Session, task_id: str) -> Optional[dict]:
        """Return serialisable dict with task status, or ``None``."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return None
        self.refresh_eta_state(db, task)
        return self.serialize_task(task)

    @staticmethod
    def refresh_eta_state(db: Session, task: Task, *, now: Optional[datetime] = None) -> bool:
        """Lazily publish a stall transition when callbacks have gone silent."""

        if (
            task.status != "RUNNING"
            or not isinstance(task.progress_meta, dict)
            or (task.eta or {}).get("state") in {"stalled", "terminal", "waiting_upstream"}
        ):
            return False
        now = now or datetime.utcnow()
        public_eta, private_state = update_eta(db, task, task.progress_meta, now=now)
        old_semantic = {
            key: (task.eta or {}).get(key)
            for key in ("state", "low_seconds", "high_seconds", "confidence")
        }
        new_semantic = {
            key: public_eta.get(key)
            for key in ("state", "low_seconds", "high_seconds", "confidence")
        }
        if old_semantic == new_semantic:
            return False
        task.eta = public_eta
        task.eta_estimator_state = private_state
        task.updated_at = now
        db.commit()
        return True

    @staticmethod
    def serialize_task(task: Task, *, include_result: bool = True) -> dict:
        """Return the allow-listed REST/SSE task representation."""

        payload = {
            "task_id": task.id,
            "document_id": task.document_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "progress_meta": sanitize_progress_meta(task.progress_meta),
            "eta": sanitize_eta(task.eta),
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }
        if include_result:
            payload["result"] = task.result
            payload["error"] = task.error
        return payload

    # ── Progress ────────────────────────────────────────────────────

    @staticmethod
    def update_progress(
        db: Session,
        task_id: str,
        progress: int,
        message: str = "",
        progress_meta: Optional[dict] = None,
        *,
        now: Optional[datetime] = None,
        commit: bool = True,
    ) -> bool:
        """Atomically advance status, monotonic progress/work units, and ETA.

        Temporal-routed activities (translation/extraction) have no
        in-process wrapper to flip PENDING -> RUNNING before work starts, so
        a progress report is itself proof the task is running.
        """
        now = now or datetime.utcnow()
        task = db.query(Task).filter(Task.id == task_id).with_for_update().first()
        if not task or task.status in {"COMPLETED", "FAILED"}:
            return False

        clean_meta = sanitize_progress_meta(progress_meta) if progress_meta is not None else None
        if clean_meta is not None and task.progress_meta:
            old_meta = sanitize_progress_meta(task.progress_meta) or {}
            same_segment = all(
                old_meta.get(key) == clean_meta.get(key)
                for key in ("pipeline", "mode", "stage", "attempt")
            )
            old_done = old_meta.get("units_done")
            new_done = clean_meta.get("units_done")
            if (
                same_segment
                and old_done is not None
                and new_done is not None
                and int(new_done) < int(old_done)
            ):
                logger.warning(
                    "Ignoring regressive task units task_id=%s old=%s new=%s",
                    task_id,
                    old_done,
                    new_done,
                )
                return False
            if same_segment:
                for key in ("unit_kind", "units_done", "units_total", "checkpoint_units", "stages"):
                    if clean_meta.get(key) is None and old_meta.get(key) is not None:
                        clean_meta[key] = old_meta[key]

        task.status = "RUNNING"
        task.progress = max(int(task.progress or 0), min(max(int(progress), 0), 100))
        if message:
            task.message = message[:1000]
        if clean_meta is not None:
            task.progress_meta = clean_meta
            phase = clean_meta.get("phase")
            if phase in {"active", "exporting", "finalizing"} and task.started_at is None:
                task.started_at = now
            task.eta, task.eta_estimator_state = update_eta(db, task, clean_meta, now=now)
        elif task.started_at is None:
            # Legacy/non-ETA tasks are active as soon as their coroutine reports.
            task.started_at = now
        task.updated_at = now
        if commit:
            db.commit()
        else:
            db.flush()
        return True

    @staticmethod
    def mark_terminal(
        db: Session,
        task_id: str,
        *,
        status: str,
        result: Any = None,
        error: Optional[str] = None,
        message: Optional[str] = None,
        now: Optional[datetime] = None,
        commit: bool = True,
    ) -> bool:
        """Apply a terminal transition once; stale callbacks cannot resurrect it."""

        if status not in {"COMPLETED", "FAILED"}:
            raise ValueError("Terminal task status must be COMPLETED or FAILED")
        now = now or datetime.utcnow()
        task = db.query(Task).filter(Task.id == task_id).with_for_update().first()
        if not task or task.status in {"COMPLETED", "FAILED"}:
            return False
        task.status = status
        task.progress = 100 if status == "COMPLETED" else int(task.progress or 0)
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if message:
            task.message = message[:1000]
        task.completed_at = now
        task.eta, task.eta_estimator_state = finish_eta(
            db,
            task,
            success=status == "COMPLETED",
            now=now,
        )
        task.updated_at = now
        if commit:
            db.commit()
        else:
            db.flush()
        return True


# Module-level singleton
task_manager = TaskManager()
