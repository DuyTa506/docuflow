"""
Task status endpoints.

GET /api/v2/tasks/{task_id}
GET /api/v2/tasks?document_id=...
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import (
    get_authorized_document,
    get_current_user,
    get_db,
    list_authorized_tasks,
    sanitize_task_payload,
)
from data.db_models import User
from services.task_manager import task_manager

router = APIRouter(prefix="/api/v2/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the current status / progress / result of a background task."""
    status = task_manager.get_status(db, task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    doc_id = status.get("document_id")
    if doc_id:
        get_authorized_document(doc_id, user, db)
    return sanitize_task_payload(status, user)


@router.get("")
async def list_tasks(
    document_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List tasks visible to the caller, optionally filtered by document_id."""
    return list_authorized_tasks(db, user, document_id=document_id)
