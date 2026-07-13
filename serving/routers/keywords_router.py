"""
Keyword extraction endpoints.

POST /api/v2/documents/{id}/keywords                     — Extract (background)
GET  /api/v2/documents/{id}/keywords                     — Get current keywords + latest extraction status
GET  /api/v2/documents/{id}/keywords/extractions         — List extraction job history
GET  /api/v2/documents/{id}/keywords/extractions/{eid}   — Get one extraction job
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from api.schemas import (
    KeywordExtractionListItem,
    KeywordsRequest,
    KeywordsResponse,
    KeywordWithWeight,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, KeywordRepository
from services.keyword_service import KeywordService

router = APIRouter(prefix="/api/v2/documents", tags=["keywords"])
_svc = KeywordService()


def _extraction_to_item(e) -> KeywordExtractionListItem:
    return KeywordExtractionListItem(
        id=e.id,
        document_id=e.document_id,
        status=e.status,
        max_keywords=e.max_keywords,
        total_keywords=e.total_keywords,
        error=e.error,
        created_at=e.created_at.isoformat() if e.created_at else None,
        updated_at=e.updated_at.isoformat() if e.updated_at else None,
    )


@router.post("/{document_id}/keywords", response_model=TaskSubmittedResponse)
async def start_keyword_extraction(
    document_id: str,
    body: KeywordsRequest = KeywordsRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Extract keywords as a background task."""
    get_authorized_document(document_id, _user, db)
    try:
        task_id, extraction_id, reused = _svc.submit(db, document_id, body.max_keywords)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(
        task_id=task_id,
        resource_id=extraction_id,
        message=(
            "Keyword extraction already in progress"
            if reused
            else "Keyword extraction task submitted"
        ),
    )


@router.get("/{document_id}/keywords", response_model=KeywordsResponse)
async def get_keywords(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get current keywords for a document plus the latest extraction job status."""
    get_authorized_document(document_id, _user, db)

    kw_repo = KeywordRepository(db)
    assocs = kw_repo.get_for_document(document_id)
    latest = kw_repo.get_latest_extraction(document_id)

    return KeywordsResponse(
        document_id=document_id,
        keywords=[
            KeywordWithWeight(keyword=kw.keyword_name, weight=assoc.weight) for assoc, kw in assocs
        ],
        latest_extraction=_extraction_to_item(latest) if latest else None,
    )


@router.get(
    "/{document_id}/keywords/extractions",
    response_model=List[KeywordExtractionListItem],
)
async def list_keyword_extractions(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List keyword-extraction jobs for a document, newest first."""
    get_authorized_document(document_id, _user, db)
    return [_extraction_to_item(e) for e in KeywordRepository(db).list_extractions(document_id)]


@router.get(
    "/{document_id}/keywords/extractions/{extraction_id}",
    response_model=KeywordExtractionListItem,
)
async def get_keyword_extraction(
    document_id: str,
    extraction_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get status / metadata of a specific keyword-extraction job."""
    get_authorized_document(document_id, _user, db)
    e = KeywordRepository(db).get_extraction(extraction_id, document_id)
    if not e:
        raise HTTPException(status_code=404, detail="Keyword extraction not found")
    return _extraction_to_item(e)
