"""
In-process async background task manager.

Uses asyncio.create_task() with DB-backed status tracking.
No external task queue required.
"""
import asyncio
import traceback
from datetime import datetime
from typing import Coroutine, Dict, Optional

from sqlalchemy.orm import Session

from data.db_models import Task
from data.database import get_db_manager
from data.id_generator import IdGenerator


class TaskManager:
    """
    Singleton task manager.

    * ``submit()``          — create DB record + launch async task
    * ``get_status()``      — read task status from DB
    * ``update_progress()`` — called by service coroutines to report %
    """

    _running_tasks: Dict[str, asyncio.Task] = {}

    # ── Submit ──────────────────────────────────────────────────────

    def submit(
        self,
        db: Session,
        document_id: Optional[str],
        task_type: str,
        coro: Coroutine,
    ) -> str:
        """
        Persist a PENDING task row, wrap *coro* with status bookkeeping,
        and launch it via ``asyncio.create_task``.

        Returns the task_id (e.g. ``TASK_042``).
        """
        task_id = IdGenerator.next_id(db, "tasks")

        task_row = Task(
            id=task_id,
            document_id=document_id,
            task_type=task_type,
            status="PENDING",
            progress=0,
        )
        db.add(task_row)
        db.commit()

        # Wrap and launch
        asyncio_task = asyncio.create_task(self._run_wrapper(task_id, coro))
        self._running_tasks[task_id] = asyncio_task
        return task_id

    # ── Wrapper ─────────────────────────────────────────────────────

    async def _run_wrapper(self, task_id: str, coro: Coroutine):
        """Execute *coro*, updating DB status on start / success / failure."""
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

    def list_tasks(self, db: Session, document_id: Optional[str] = None) -> list:
        """List tasks, optionally filtered by document_id."""
        query = db.query(Task)
        if document_id:
            query = query.filter(Task.document_id == document_id)
        tasks = query.order_by(Task.created_at.desc()).all()
        return [
            {
                "task_id": t.id,
                "document_id": t.document_id,
                "task_type": t.task_type,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }
            for t in tasks
        ]

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
