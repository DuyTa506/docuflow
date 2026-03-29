"""
Task repository — queries for Task model.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import Task


class TaskRepository:
    """CRUD + query methods for background tasks. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, task_id: str) -> Optional[Task]:
        """Return a task by ID or None."""
        return self.db.query(Task).filter(Task.id == task_id).first()

    def list(self, document_id: Optional[str] = None) -> List[Task]:
        """Return tasks, optionally filtered by document_id."""
        query = self.db.query(Task)
        if document_id:
            query = query.filter(Task.document_id == document_id)
        return query.order_by(Task.created_at.desc()).all()

    def find_latest(self, document_id: str, task_type: str) -> Optional[Task]:
        """Return the most recent task of a given type for a document."""
        return (
            self.db.query(Task)
            .filter(
                Task.document_id == document_id,
                Task.task_type == task_type,
            )
            .order_by(Task.created_at.desc())
            .first()
        )
