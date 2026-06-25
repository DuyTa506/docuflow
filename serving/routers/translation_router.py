"""
Translation endpoints.

POST /api/v2/documents/{id}/translations                    — Start translation
GET  /api/v2/documents/{id}/translations                    — List translations
GET  /api/v2/documents/{id}/translations/{tid}              — Get specific
GET  /api/v2/documents/{id}/translations/{tid}/download     — Download as .docx file
POST /api/v2/documents/{id}/translations/{tid}/upload       — Override via .txt/.docx file
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from api.schemas import (
    TranslationRequest,
    TranslationResponse,
    TranslationListItem,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, TranslationRepository
from services.translation_service import TranslationService
from utils.file_download import (
    build_docx_bytes_from_elements,
    build_docx_response,
    build_docx_response_from_elements,
    build_original_file_response,
    build_pdf_response,
    docx_bytes_to_pdf_bytes,
    safe_filename,
)
from utils.file_upload import extract_text_from_upload
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
        task_id, translation_id, reused = _svc.submit(
            db, document_id, body.target_language, body.domain
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
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get a specific translation."""
    get_authorized_document(document_id, _user, db)
    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
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
    """Download a translation as .docx or .pdf."""
    doc = get_authorized_document(document_id, user, db)

    trans_repo = TranslationRepository(db)
    t = trans_repo.get(translation_id, document_id)
    if not t:
        raise HTTPException(status_code=404, detail="Translation not found")
    if t.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Translation is not yet complete")

    lang = t.target_language.upper()
    base = f"translation_{lang}_{safe_filename(doc.title)}"
    filename = f"{base}.docx"

    use_structured = source in ("auto", "structured")

    if use_structured and t.translation_mode == "pdf_overlay" and t.translated_file_path:
        if format == "pdf":
            return build_original_file_response(
                t.translated_file_path,
                download_name=f"{base}.pdf",
            )
        raise HTTPException(
            status_code=400,
            detail="PDF overlay translation has no DOCX export — use format=pdf",
        )

    if use_structured and t.translation_mode == "docx_inplace" and t.translated_file_path:
        download_name = doc.original_filename or filename
        if not download_name.lower().endswith(".docx"):
            download_name = filename
        if format == "pdf":
            try:
                with open(t.translated_file_path, "rb") as f:
                    pdf_bytes = docx_bytes_to_pdf_bytes(f.read())
                return build_pdf_response(f"{base}.pdf", pdf_bytes)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc
        return build_original_file_response(t.translated_file_path, download_name=download_name)

    if use_structured and t.translation_mode == "element_based" and t.translated_elements:
        elements = deserialize_translated_elements(t.translated_elements)
        if elements:
            if format == "pdf":
                try:
                    docx_bytes = build_docx_bytes_from_elements(elements, title=doc.title)
                    pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
                    return build_pdf_response(f"{base}.pdf", pdf_bytes)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc
            return build_docx_response_from_elements(
                filename,
                elements,
                title=doc.title,
            )

    if not t.translated_content:
        raise HTTPException(status_code=404, detail="Translation has no content")

    if format == "pdf":
        try:
            from utils.file_download import build_docx_bytes_from_content

            docx_bytes = build_docx_bytes_from_content(
                t.translated_content,
                title=doc.title,
                headings=[f"Translation ({lang})"],
                structured=source != "flat",
            )
            pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
            return build_pdf_response(f"{base}.pdf", pdf_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

    return build_docx_response(
        filename,
        t.translated_content,
        title=doc.title,
        headings=[f"Translation ({lang})"],
        structured=source != "flat",
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
    get_authorized_document(document_id, _user, db)
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
        translation_mode=t.translation_mode,
        status=t.status,
        created_at=t.created_at.isoformat() if t.created_at else None,
        updated_at=t.updated_at.isoformat() if t.updated_at else None,
    )
