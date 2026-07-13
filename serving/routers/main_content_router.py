"""
Main content extraction endpoints.

POST /api/v2/documents/{id}/main-content       — Extract (background)
GET  /api/v2/documents/{id}/main-content       — Get latest extraction (incl. status)
GET  /api/v2/documents/{id}/main-content/list  — List all extraction jobs
GET  /api/v2/documents/{id}/main-content/{id}  — Get specific extraction
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from api.schemas import (
    MainContentListItem,
    MainContentResponse,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, MainContentRepository
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
    get_authorized_document(document_id, _user, db)
    try:
        task_id, main_content_id, reused = _svc.submit(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(
        task_id=task_id,
        resource_id=main_content_id,
        message=(
            "Main content extraction already in progress"
            if reused
            else "Main content extraction task submitted"
        ),
    )


@router.get("/{document_id}/main-content")
async def get_main_content(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get the most recent main-content extraction for a document (incl. status)."""
    get_authorized_document(document_id, _user, db)
    repo = MainContentRepository(db)
    items = repo.list(document_id)
    if not items:
        return {
            "document_id": document_id,
            "details": None,
            "message": "No main content extracted yet",
        }
    mc = items[0]
    return MainContentResponse(
        id=mc.id,
        document_id=mc.document_id,
        details=mc.details,
        status=mc.status,
        created_at=mc.created_at.isoformat() if mc.created_at else None,
        updated_at=mc.updated_at.isoformat() if mc.updated_at else None,
    )


@router.get("/{document_id}/main-content/list", response_model=list[MainContentListItem])
async def list_main_content(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all main-content extraction jobs for a document, newest first."""
    get_authorized_document(document_id, _user, db)
    items = MainContentRepository(db).list(document_id)
    return [
        MainContentListItem(
            id=mc.id,
            document_id=mc.document_id,
            status=mc.status,
            has_details=mc.details is not None,
            created_at=mc.created_at.isoformat() if mc.created_at else None,
            updated_at=mc.updated_at.isoformat() if mc.updated_at else None,
        )
        for mc in items
    ]


@router.get("/{document_id}/main-content/{main_content_id}", response_model=MainContentResponse)
async def get_main_content_by_id(
    document_id: str,
    main_content_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get a specific main-content extraction by id (incl. status)."""
    get_authorized_document(document_id, _user, db)
    mc = MainContentRepository(db).get(main_content_id, document_id)
    if not mc:
        raise HTTPException(status_code=404, detail="Main content not found")
    return MainContentResponse(
        id=mc.id,
        document_id=mc.document_id,
        details=mc.details,
        status=mc.status,
        created_at=mc.created_at.isoformat() if mc.created_at else None,
        updated_at=mc.updated_at.isoformat() if mc.updated_at else None,
    )
