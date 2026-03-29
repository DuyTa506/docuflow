"""
Summarization endpoints.

POST /api/v2/documents/{id}/summaries     — Generate summary (background)
GET  /api/v2/documents/{id}/summaries     — List summaries
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from api.schemas import SummaryRequest, SummaryResponse, TaskSubmittedResponse, SummaryListItem
from data.db_models import User
from data.repositories import DocumentRepository, SummaryRepository
from services.summarization_service import SummarizationService

router = APIRouter(prefix="/api/v2/documents", tags=["summaries"])
_svc = SummarizationService()


@router.post("/{document_id}/summaries", response_model=TaskSubmittedResponse)
async def start_summarization(
    document_id: str,
    body: SummaryRequest = SummaryRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Generate a document summary as a background task."""
    try:
        task_id = _svc.submit(db, document_id, body.summary_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, message="Summarization task submitted")


@router.get("/{document_id}/summaries", response_model=list[SummaryListItem])
async def list_summaries(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all summaries for a document."""
    doc_repo = DocumentRepository(db)
    if not doc_repo.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    summary_repo = SummaryRepository(db)
    summaries = summary_repo.list(document_id)
    return [
        SummaryListItem(
            id=s.id,
            document_id=s.document_id,
            summary_type=s.summary_type,
            content=s.content,
            created_at=s.created_at.isoformat() if s.created_at else None,
        )
        for s in summaries
    ]
