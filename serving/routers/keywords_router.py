"""
Keyword extraction endpoints.

POST /api/v2/documents/{id}/keywords   — Extract (background)
GET  /api/v2/documents/{id}/keywords   — Get keywords
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from api.schemas import KeywordsRequest, KeywordsResponse, KeywordWithWeight, TaskSubmittedResponse
from data.db_models import User
from data.repositories import DocumentRepository, KeywordRepository
from services.keyword_service import KeywordService

router = APIRouter(prefix="/api/v2/documents", tags=["keywords"])
_svc = KeywordService()


@router.post("/{document_id}/keywords", response_model=TaskSubmittedResponse)
async def start_keyword_extraction(
    document_id: str,
    body: KeywordsRequest = KeywordsRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Extract keywords as a background task."""
    try:
        task_id = _svc.submit(db, document_id, body.max_keywords)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, message="Keyword extraction task submitted")


@router.get("/{document_id}/keywords", response_model=KeywordsResponse)
async def get_keywords(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get extracted keywords for a document."""
    doc_repo = DocumentRepository(db)
    if not doc_repo.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    kw_repo = KeywordRepository(db)
    assocs = kw_repo.get_for_document(document_id)

    return KeywordsResponse(
        document_id=document_id,
        keywords=[
            KeywordWithWeight(keyword=kw.keyword_name, weight=assoc.weight)
            for assoc, kw in assocs
        ],
    )
