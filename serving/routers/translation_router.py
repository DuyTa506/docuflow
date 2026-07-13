"""
Translation endpoints.

POST /api/v2/documents/{id}/translations                    — Start translation
GET  /api/v2/documents/{id}/translations                    — List translations
GET  /api/v2/documents/{id}/translations/{tid}              — Get specific
GET  /api/v2/documents/{id}/translations/{tid}/download     — Download as .docx file
POST /api/v2/documents/{id}/translations/{tid}/upload       — Override via .txt/.docx file
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from api.schemas import (
    TaskSubmittedResponse,
    TranslationListItem,
    TranslationRequest,
    TranslationResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, TranslationRepository
from services.export_service import export_service
from services.translation_service import TranslationService
from utils.file_download import build_stored_file_response
from utils.file_upload import extract_text_from_upload
from utils.preview_text import preview_flat_text, preview_translated_elements
from utils.translation_elements import deserialize_translated_elements

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
    get_authorized_document(document_id, _user, db)
    try:
        task_id, translation_id, reused = await _svc.submit_async(
            db, document_id, body.target_language, body.domain, fairness_key=_user.id
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    return TaskSubmittedResponse(
        task_id=task_id,
        resource_id=translation_id,
        message="Translation already in progress" if reused else "Translation task submitted",
    )


@router.delete("/{document_id}/translations/{translation_id}")
async def cancel_translation(
    document_id: str,
    translation_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Cancel a running Temporal translation workflow (no-op on the legacy
    in-process path, which has no cancellation)."""
    get_authorized_document(document_id, _user, db)
    trans = TranslationRepository(db).get(translation_id, document_id)
    if not trans:
        raise HTTPException(status_code=404, detail="Translation not found")

    from services.pipeline.temporal_client import cancel_translation_workflow

    cancelled = await cancel_translation_workflow(document_id, trans.target_language)
    if not cancelled:
        raise HTTPException(status_code=409, detail="No running translation workflow to cancel")

    from data.db_models import Task, Translation

    trans_row = db.query(Translation).filter(Translation.id == translation_id).first()
    if trans_row:
        trans_row.status = "FAILED"
    task = (
        db.query(Task)
        .filter(
            Task.document_id == document_id,
            Task.task_type == "TRANSLATE",
            Task.status.in_(["PENDING", "RUNNING"]),
        )
        .order_by(Task.created_at.desc())
        .first()
    )
    if task:
        task.status = "FAILED"
        task.error = "Cancelled by user"
    db.commit()
    return {"cancelled": True, "translation_id": translation_id}


@router.get("/{document_id}/translations", response_model=List[TranslationListItem])
async def list_translations(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all translations for a document."""
    get_authorized_document(document_id, _user, db)
    trans_repo = TranslationRepository(db)
    translations = trans_repo.list(document_id)
    return [
        TranslationListItem(
            id=t.id,
            document_id=t.document_id,
            target_language=t.target_language,
            translation_mode=t.translation_mode,
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
    preview_pages: Optional[int] = Query(
        None,
        ge=1,
        description="If set, return only a preview of the first N pages.",
    ),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get a specific translation."""
    get_authorized_document(document_id, _user, db)
    trans_repo = TranslationRepository(db)
    doc_repo = DocumentRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")

    content = t.translated_content
    truncated = False
    total_pages = doc_repo.get_pages(document_id)
    total_page_count = len(total_pages) if total_pages else None

    if preview_pages and content:
        if t.translation_mode in ("element_based", "block_based") and t.translated_elements:
            elements = deserialize_translated_elements(t.translated_elements)
            if elements:
                max_page = max((e.get("page_number") or 1) for e in elements)
                total_page_count = max_page
                content, truncated = preview_translated_elements(elements, preview_pages)
        else:
            content, truncated = preview_flat_text(content, preview_pages)

    return TranslationResponse(
        id=t.id,
        document_id=t.document_id,
        target_language=t.target_language,
        translated_content=content,
        translation_mode=t.translation_mode,
        status=t.status,
        truncated=truncated,
        total_pages=total_page_count,
        created_at=t.created_at.isoformat() if t.created_at else None,
        updated_at=t.updated_at.isoformat() if t.updated_at else None,
    )


@router.get("/{document_id}/translations/{translation_id}/download")
async def download_translation(
    document_id: str,
    translation_id: str,
    source: str = Query(
        "auto",
        pattern="^(auto|structured|flat)$",
        description="auto=docx file or spatial elements when available; structured=spatial; flat=legacy markdown",
    ),
    format: str = Query(
        "docx",
        pattern="^(docx|pdf)$",
        description="docx=Word; pdf=overlay PDF or spatial docx→PDF",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Download a translation as .docx or .pdf (streamed from MinIO cache)."""
    doc = get_authorized_document(document_id, user, db)

    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    if t.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Translation is not yet complete")

    try:
        key, filename, media_type = await asyncio.to_thread(
            export_service.get_or_build_translation_export,
            db,
            doc,
            t,
            source=source,
            fmt=format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    return await asyncio.to_thread(
        build_stored_file_response, key, download_name=filename, content_type=media_type
    )


@router.post(
    "/{document_id}/translations/{translation_id}/upload", response_model=TranslationResponse
)
async def upload_translation(
    document_id: str,
    translation_id: str,
    file: UploadFile = File(..., description="Corrected translation as .txt or .docx"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Override translation content by uploading a corrected .txt or .docx file."""
    get_authorized_document(document_id, _user, db)
    text = await extract_text_from_upload(file)
    trans_repo = TranslationRepository(db)
    t = trans_repo.update(translation_id, document_id, text)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    from utils.storage_keys import translation_file_key

    for ext in ("docx", "pdf"):
        export_service.storage.delete(translation_file_key(document_id, translation_id, ext))
    return TranslationResponse(
        id=t.id,
        document_id=t.document_id,
        target_language=t.target_language,
        translated_content=t.translated_content,
        translation_mode=t.translation_mode,
        status=t.status,
        created_at=t.created_at.isoformat() if t.created_at else None,
        updated_at=t.updated_at.isoformat() if t.updated_at else None,
    )
