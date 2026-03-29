"""
Task status endpoints.

GET /api/v2/tasks/{task_id}
GET /api/v2/tasks?document_id=...
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from data.db_models import User
from services.task_manager import task_manager

router = APIRouter(prefix="/api/v2/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get the current status / progress / result of a background task."""
    status = task_manager.get_status(db, task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("")
async def list_tasks(
    document_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List tasks, optionally filtered by document_id."""
    return task_manager.list_tasks(db, document_id=document_id)
