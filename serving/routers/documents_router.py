"""
Document management endpoints (v2).

POST   /api/v2/documents/upload
POST   /api/v2/documents/{id}/extract
POST   /api/v2/documents/{id}/text/upload   — Override OCR/extracted text via file
GET    /api/v2/documents
GET    /api/v2/documents/{id}
GET    /api/v2/documents/{id}/text
GET    /api/v2/documents/{id}/text/download — Download OCR/normalized text as .txt file
GET    /api/v2/documents/{id}/pages
GET    /api/v2/documents/{id}/elements
DELETE /api/v2/documents/{id}
"""

import asyncio
import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from api.schemas import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentTextResponse,
    DocumentUploadResponse,
    ElementListItem,
    PageListItem,
    TaskSubmittedResponse,
)
from config.settings import settings
from data.db_models import Task, User
from data.repositories import DocumentRepository
from data.repositories.document_repo import delete_document_cascade
from services.document_service import DocumentService
from services.export_service import export_service
from utils.file_download import build_stored_file_response
from utils.file_upload import extract_text_from_upload

router = APIRouter(prefix="/api/v2/documents", tags=["documents"])
_doc_svc = DocumentService()


# ── Upload ──────────────────────────────────────────────────────────


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source_language: str = Form("en"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a PDF / image / DOCX / DOC file and register it as a document."""
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save file to a temp path; service moves it under uploads/<doc_id>/
    import uuid

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".doc"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    tmp_name = f"_upload_{uuid.uuid4().hex}{ext}"
    dest = os.path.join(settings.upload_dir, tmp_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = _doc_svc.upload_document(
        db,
        file_path_on_disk=dest,
        original_filename=file.filename,
        user_id=user.id,
        title=title,
        source_language=source_language,
    )

    # Auto-trigger OCR/extraction — the user shouldn't need a second click.
    # A submit failure (e.g. Temporal briefly down) must not fail the upload:
    # the document is saved and extraction can be started manually.
    extract_task_id = None
    if settings.auto_extract_on_upload:
        try:
            extract_task_id, _ = await _doc_svc.submit_extraction_async(
                db, doc.id, fairness_key=user.id
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Auto-extraction submit failed for %s (upload still OK): %s", doc.id, exc
            )

    return DocumentUploadResponse(
        document_id=doc.id,
        title=doc.title,
        format=doc.format,
        total_pages=doc.total_pages,
        processing_status=doc.processing_status,
        extract_task_id=extract_task_id,
    )


# ── Trigger unified extraction ──────────────────────────────────────


@router.post("/{document_id}/extract", response_model=TaskSubmittedResponse)
async def start_extraction(
    document_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Start unified extraction as a background task.

    Routes automatically based on document format:
    - DOCX / DOC  → python-docx (DOC first converted via LibreOffice)
    - PDF text    → PyMuPDF direct text extraction (per text page)
    - PDF scanned → DeepSeek vLLM OCR (per scanned page)
    - Image       → DeepSeek vLLM OCR
    """
    get_authorized_document(document_id, user, db)
    try:
        task_id, reused = await _doc_svc.submit_extraction_async(
            db, document_id, fairness_key=user.id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(
        task_id=task_id,
        message="Extraction already in progress" if reused else "Extraction task submitted",
    )


# ── List documents ──────────────────────────────────────────────────


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    List documents (paginated).

    Visibility rules:
    - ADMIN  → sees all documents from all users
    - MEMBER → sees only their own documents

    Each item includes a `task_summary` dict with the latest status of each
    pipeline task (EXTRACT, TRANSLATE, SUMMARIZE, KEYWORDS, etc.).
    """
    import math

    repo = DocumentRepository(db)
    offset = (page - 1) * limit

    if user.role == "ADMIN":
        docs = repo.list(limit=limit, offset=offset)
        total = repo.count()
    else:
        docs = repo.list_for_user(user.id, limit=limit, offset=offset)
        total = repo.count_for_user(user.id)

    total_pages = math.ceil(total / limit) if total > 0 else 1

    # Build task summary for each document in one batch query
    doc_ids = [d.id for d in docs]
    task_summary_map: dict[str, dict[str, str]] = {doc_id: {} for doc_id in doc_ids}
    if doc_ids:
        tasks = (
            db.query(Task)
            .filter(Task.document_id.in_(doc_ids))
            .order_by(Task.created_at.asc())  # asc so latest overwrites earlier
            .all()
        )
        for t in tasks:
            task_summary_map[t.document_id][t.task_type] = t.status

    items = [
        DocumentListItem(
            id=d.id,
            title=d.title,
            original_filename=d.original_filename,
            format=d.format,
            total_pages=d.total_pages,
            processing_status=d.processing_status,
            source_language=d.source_language,
            created_at=d.created_at.isoformat() if d.created_at else None,
            task_summary=task_summary_map.get(d.id) or None,
        )
        for d in docs
    ]
    return DocumentListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )


# ── Get document detail ─────────────────────────────────────────────


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get document metadata.

    Access rules: users can only access their own documents.
    ADMIN can access any document.
    """
    doc = get_authorized_document(document_id, user, db)
    return DocumentDetailResponse(
        id=doc.id,
        title=doc.title,
        original_filename=doc.original_filename,
        source_language=doc.source_language,
        format=doc.format,
        file_type=doc.file_type,
        total_pages=doc.total_pages,
        processing_status=doc.processing_status,
        user_id=doc.user_id,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


# ── Get document text (OCR / normalized) ────────────────────────────


@router.get("/{document_id}/text", response_model=DocumentTextResponse)
async def get_document_text(
    document_id: str,
    preview_pages: Optional[int] = Query(
        None,
        ge=1,
        description="If set, return only the first N pages of text (preview mode).",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get OCR and/or normalized text for a document (own docs only; admin can access all)."""
    from utils.content_storage import read_text_field
    from utils.preview_text import preview_flat_text, preview_from_page_markdown
    from utils.text_assembly import assemble_ocr_from_pages

    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    dt = repo.get_digitized_text(document_id)
    if not dt:
        return DocumentTextResponse(document_id=document_id)

    truncated = False

    if preview_pages:
        # Preview mode: fetch only the pages actually needed instead of every
        # page in the document, so response time doesn't scale with the
        # document's total page count.
        total_pages = repo.count_pages(document_id)
        pages = repo.get_pages(document_id, limit=preview_pages) if total_pages else []
    else:
        pages = repo.get_pages(document_id)
        total_pages = len(pages) if pages else None

    if pages:
        if preview_pages:
            # `pages` is already limited to `preview_pages` rows via SQL —
            # join them as-is rather than re-slicing.
            ocr_content, _, _ = preview_from_page_markdown(pages, None, field="markdown_content")
            truncated = total_pages is not None and len(pages) < total_pages
        else:
            ocr_content = assemble_ocr_from_pages(pages)
    else:
        ocr_content = read_text_field(inline=dt.ocr_content, key=dt.ocr_content_key)
        if preview_pages:
            ocr_content, ocr_trunc = preview_flat_text(ocr_content, preview_pages)
            truncated = truncated or ocr_trunc

    full_normalized = (
        read_text_field(
            inline=dt.normalized_content,
            key=dt.normalized_content_key,
        )
        or ocr_content
    )

    if preview_pages:
        normalized_content, norm_trunc = preview_flat_text(full_normalized, preview_pages)
        truncated = truncated or norm_trunc
    else:
        normalized_content = full_normalized

    return DocumentTextResponse(
        document_id=document_id,
        ocr_content=ocr_content,
        normalized_content=normalized_content,
        truncated=truncated,
        total_pages=total_pages,
    )


# ── Export cache status ─────────────────────────────────────────────


@router.get("/{document_id}/exports/status")
async def export_cache_status(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Report which download exports are already cached in MinIO (fast download)."""
    get_authorized_document(document_id, _user, db)
    return await asyncio.to_thread(export_service.export_cache_status, db, document_id)


# ── Download OCR / normalized text as file ──────────────────────────


@router.get("/{document_id}/text/download")
async def download_document_text(
    document_id: str,
    type: str = Query("ocr", pattern="^(ocr|normalized)$"),
    mode: str = Query(
        "auto",
        pattern="^(auto|markdown|spatial|plain)$",
        description="auto=layout PDF or spatial DOCX when elements exist; spatial=force spatial DOCX; markdown/plain=flat text",
    ),
    format: str = Query(
        "docx",
        pattern="^(docx|pdf)$",
        description="docx=Word (spatial reflow); pdf=layout-faithful PDF when elements exist",
    ),
    source: str = Query(
        "auto",
        pattern="^(auto|original|extracted)$",
        description="auto=original file for docx/doc, extracted export for pdf/image; "
        "original=uploaded file; extracted=build docx from digitized text",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
      Download document text or the original uploaded file from MinIO.

    - DOCX / DOC: default (`source=auto`) returns the **original uploaded file**.
    - PDF / image: default builds a structured .docx from extracted/OCR text (cached in MinIO).
    """
    from utils.file_download import is_native_word_document

    doc = get_authorized_document(document_id, user, db)

    try:
        key, filename, media_type = await asyncio.to_thread(
            export_service.get_or_build_ocr_export,
            db,
            doc,
            content_type=type,
            mode=mode,
            fmt=format,
            source=source,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("No ") and ("found" in msg or "available" in msg):
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=500, detail=f"Export failed: {msg}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    if source in ("auto", "original") and is_native_word_document(doc.format) and format == "docx":
        download_name = doc.original_filename or filename
    else:
        download_name = filename

    return await asyncio.to_thread(
        build_stored_file_response, key, download_name=download_name, content_type=media_type
    )


# ── Upload corrected OCR / text ──────────────────────────────────────


@router.post("/{document_id}/text/upload", response_model=DocumentTextResponse)
async def upload_document_text(
    document_id: str,
    file: UploadFile = File(..., description="Corrected document text as .txt or .docx"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Override the extracted/OCR text by uploading a corrected .txt or .docx file.
    Writes to normalized_content (the cleaned text used by all downstream pipeline steps).
    """
    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    if not repo.get_digitized_text(document_id):
        raise HTTPException(
            status_code=409,
            detail="No extracted text exists yet. Run /extract first, then upload your correction.",
        )

    text = await extract_text_from_upload(file)
    dt = repo.update_digitized_text(document_id, text)
    export_service.invalidate_ocr_exports(document_id)
    return DocumentTextResponse(
        document_id=document_id,
        ocr_content=dt.ocr_content,
        normalized_content=dt.normalized_content,
    )


# ── Get pages with markdown ─────────────────────────────────────────


@router.get("/{document_id}/pages", response_model=list[PageListItem])
async def get_document_pages(
    document_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all pages for a document (own docs only; admin can access all)."""
    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    pages = repo.get_pages(document_id)
    return [
        PageListItem(
            id=p.id,
            page_number=p.page_number,
            markdown_content=p.markdown_content,
            image_width=p.image_width,
            image_height=p.image_height,
        )
        for p in pages
    ]


@router.get("/{document_id}/pages/{page_number}/image")
async def get_page_image(
    document_id: str,
    page_number: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Stream a page scan image from MinIO (fallback: legacy base64 in DB)."""
    import base64

    from services.object_storage import get_object_storage

    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    pages = repo.get_pages(document_id)
    page = next((p for p in pages if p.page_number == page_number), None)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    if page.image_key:
        storage = get_object_storage()
        if await asyncio.to_thread(storage.exists, page.image_key):
            return await asyncio.to_thread(
                build_stored_file_response,
                page.image_key,
                download_name=f"page_{page_number:04d}.jpg",
                content_type="image/jpeg",
            )

    if page.image_base64:
        data = base64.b64decode(page.image_base64)
        from fastapi.responses import Response

        return Response(content=data, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="No image available for this page")


# ── Get layout elements ─────────────────────────────────────────────


@router.get("/{document_id}/elements", response_model=list[ElementListItem])
async def get_document_elements(
    document_id: str,
    label: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get layout elements (bounding boxes) for a document (own docs only; admin can access all)."""
    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    elements = repo.get_elements(document_id, label=label)
    result = []
    for elem in elements:
        result.append(
            ElementListItem(
                id=elem.id,
                label=elem.label,
                text_content=elem.text_content,
                bbox={
                    "x1": elem.bbox_x1,
                    "y1": elem.bbox_y1,
                    "x2": elem.bbox_x2,
                    "y2": elem.bbox_y2,
                },
                bbox_normalized=(
                    {
                        "x1": elem.bbox_norm_x1,
                        "y1": elem.bbox_norm_y1,
                        "x2": elem.bbox_norm_x2,
                        "y2": elem.bbox_norm_y2,
                    }
                    if elem.bbox_norm_x1 is not None
                    else None
                ),
                page_number=elem.page.page_number if elem.page else None,
                page_id=elem.page_id,
                sequence_order=elem.sequence_order,
                has_crop_image=bool(elem.crop_image_key or elem.crop_image_base64),
            )
        )
    return result


# ── Delete document ─────────────────────────────────────────────────


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Permanently delete a document and all its related data (pages, OCR text,
    translations, summaries, keywords, research directions, tasks).

    Access: ADMIN can delete any document; regular users can only delete their own.
    The uploaded file on disk is also removed.
    """
    repo = DocumentRepository(db)
    doc = get_authorized_document(document_id, user, db)

    export_service.invalidate_document(document_id)
    await asyncio.to_thread(delete_document_cascade, document_id)
