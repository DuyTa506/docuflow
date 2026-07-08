"""
In-process async background task manager.

Uses asyncio.create_task() with DB-backed status tracking.
No external task queue required.
"""
import asyncio
import traceback
from datetime import datetime
from typing import Callable, Coroutine, Dict, Optional, Union

from sqlalchemy.orm import Session

from data.db_models import Task
from data.database import get_db_manager
from data.id_generator import IdGenerator


from data.repositories import TaskRepository


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
            task.status = "FAILED"
            task.error = (task.error or "") + "\nStale task reset (no active worker)."
            task.updated_at = datetime.utcnow()
            db.commit()
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
        raw_id  = IdGenerator.next_id(db, "tasks")          # e.g. TASK_042
        seq_num = raw_id.split("_")[-1]                      # e.g. 042
        task_id = f"{task_type}_{seq_num}"                   # e.g. SUMMARIZE_042

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
                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    task.status = "RUNNING"
                    task.updated_at = datetime.utcnow()

            try:
                result = await coro
                # Mark COMPLETED
                with db_manager.session() as db:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    if task:
                        task.status = "COMPLETED"
                        task.progress = 100
                        task.result = result if isinstance(result, (dict, list)) else {"detail": str(result)}
                        task.updated_at = datetime.utcnow()
            except Exception as exc:
                tb = traceback.format_exc()
                with db_manager.session() as db:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    if task:
                        task.status = "FAILED"
                        task.error = f"{exc}\n{tb}"
                        task.updated_at = datetime.utcnow()
            finally:
                self._running_tasks.pop(task_id, None)

    # ── Query ───────────────────────────────────────────────────────

    def get_status(self, db: Session, task_id: str) -> Optional[dict]:
        """Return serialisable dict with task status, or ``None``."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return None
        return {
            "task_id": task.id,
            "document_id": task.document_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }

    # ── Progress ────────────────────────────────────────────────────

    @staticmethod
    def update_progress(
        db: Session,
        task_id: str,
        progress: int,
        message: str = "",
    ):
        """Called by service coroutines to report percentage progress."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.progress = min(progress, 100)
            if message:
                task.message = message
            task.updated_at = datetime.utcnow()
            db.commit()


# Module-level singleton
task_manager = TaskManager()
