"""
Main content extraction endpoints.

POST /api/v2/documents/{id}/main-content   — Extract (background)
GET  /api/v2/documents/{id}/main-content   — Get result
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from api.schemas import MainContentResponse, TaskSubmittedResponse
from data.db_models import User, MainContent
from data.repositories import DocumentRepository
from services.main_content_service import MainContentService

router = APIRouter(prefix="/api/v2/documents", tags=["main-content"])
_svc = MainContentService()


@router.post("/{document_id}/main-content", response_model=TaskSubmittedResponse)
async def start_main_content_extraction(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Extract structured main content as a background task."""
    try:
        task_id = _svc.submit(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, message="Main content extraction task submitted")


@router.get("/{document_id}/main-content")
async def get_main_content(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get extracted main content for a document."""
    doc_repo = DocumentRepository(db)
    if not doc_repo.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    mc = (
        db.query(MainContent)
        .filter(MainContent.document_id == document_id)
        .order_by(MainContent.created_at.desc())
        .first()
    )
    if not mc:
        return {"document_id": document_id, "details": None, "message": "No main content extracted yet"}
    return MainContentResponse(
        id=mc.id,
        document_id=mc.document_id,
        details=mc.details,
        created_at=mc.created_at.isoformat() if mc.created_at else None,
    )
