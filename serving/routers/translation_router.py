"""
Translation endpoints.

POST /api/v2/documents/{id}/translations                    — Start translation
GET  /api/v2/documents/{id}/translations                    — List translations
GET  /api/v2/documents/{id}/translations/{tid}              — Get specific
GET  /api/v2/documents/{id}/translations/{tid}/download     — Download as .docx file
POST /api/v2/documents/{id}/translations/{tid}/upload       — Override via .txt/.docx file
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from api.schemas import (
    TranslationRequest,
    TranslationResponse,
    TranslationListItem,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, TranslationRepository
from services.translation_service import TranslationService
from utils.file_download import build_docx_response, safe_filename
from utils.file_upload import extract_text_from_upload

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
        task_id, translation_id = _svc.submit(db, document_id, body.target_language, body.domain)
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    return TaskSubmittedResponse(
        task_id=task_id,
        resource_id=translation_id,
        message="Translation task submitted",
    )


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


@router.get("/{document_id}/translations/{translation_id}/download")
async def download_translation(
    document_id: str,
    translation_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Download a translation as a .docx file."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if user.role != "ADMIN" and doc.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    if not t.translated_content:
        raise HTTPException(status_code=404, detail="Translation has no content")
    if t.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Translation is not yet complete")

    lang = t.target_language.upper()
    filename = f"translation_{lang}_{safe_filename(doc.title)}.docx"
    return build_docx_response(
        filename,
        t.translated_content,
        title=doc.title,
        headings=[f"Translation ({lang})"],
    )


@router.post("/{document_id}/translations/{translation_id}/upload", response_model=TranslationResponse)
async def upload_translation(
    document_id: str,
    translation_id: str,
    file: UploadFile = File(..., description="Corrected translation as .txt or .docx"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Override translation content by uploading a corrected .txt or .docx file."""
    text = await extract_text_from_upload(file)
    trans_repo = TranslationRepository(db)
    t = trans_repo.update(translation_id, document_id, text)
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
