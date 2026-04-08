"""
Translation endpoints.

POST /api/v2/documents/{id}/translations          — Start translation
GET  /api/v2/documents/{id}/translations           — List translations
GET  /api/v2/documents/{id}/translations/{tid}     — Get specific
PUT  /api/v2/documents/{id}/translations/{tid}     — Edit
POST /api/v2/documents/{id}/translations/{tid}/approve — Approve
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, require_role
from api.schemas import (
    TranslationRequest,
    TranslationResponse,
    TranslationListItem,
    TranslationEditRequest,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, TranslationRepository
from services.translation_service import TranslationService

router = APIRouter(prefix="/api/v2/documents", tags=["translations"])
_svc = TranslationService()


@router.post("/{document_id}/translations", response_model=TaskSubmittedResponse)
async def start_translation(
    document_id: str,
    body: TranslationRequest = TranslationRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Start document translation as a background task."""
    try:
        task_id = _svc.submit(db, document_id, body.target_language, body.domain)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, message="Translation task submitted")


@router.get("/{document_id}/translations", response_model=List[TranslationListItem])
async def list_translations(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all translations for a document."""
    doc_repo = DocumentRepository(db)
    if not doc_repo.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    trans_repo = TranslationRepository(db)
    translations = trans_repo.list(document_id)
    return [
        TranslationListItem(
            id=t.id,
            document_id=t.document_id,
            target_language=t.target_language,
            status=t.status,
            created_at=t.created_at.isoformat() if t.created_at else None,
            updated_at=t.updated_at.isoformat() if t.updated_at else None,
        )
        for t in translations
    ]


@router.get("/{document_id}/translations/{translation_id}", response_model=TranslationResponse)
async def get_translation(
    document_id: str,
    translation_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get a specific translation."""
    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    return TranslationResponse(
        id=t.id,
        document_id=t.document_id,
        target_language=t.target_language,
        translated_content=t.translated_content,
        status=t.status,
        created_at=t.created_at.isoformat() if t.created_at else None,
        updated_at=t.updated_at.isoformat() if t.updated_at else None,
    )


@router.put("/{document_id}/translations/{translation_id}", response_model=TranslationResponse)
async def edit_translation(
    document_id: str,
    translation_id: str,
    body: TranslationEditRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Edit translation content (manual correction)."""
    trans_repo = TranslationRepository(db)
    t = trans_repo.update(translation_id, document_id, body.translated_content)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    return TranslationResponse(
        id=t.id,
        document_id=t.document_id,
        target_language=t.target_language,
        translated_content=t.translated_content,
        status=t.status,
        created_at=t.created_at.isoformat() if t.created_at else None,
        updated_at=t.updated_at.isoformat() if t.updated_at else None,
    )


@router.post("/{document_id}/translations/{translation_id}/approve")
async def approve_translation(
    document_id: str,
    translation_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(require_role("ADMIN", "LIBRARIAN")),
):
    """Approve a translation (LIBRARIAN+ only)."""
    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    t.status = "APPROVED"
    db.commit()
    return {"id": t.id, "status": t.status, "message": "Translation approved"}
