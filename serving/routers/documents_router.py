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
import os
import shutil
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Form
from fastapi import status as http_status

from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from api.schemas import (
    TaskSubmittedResponse,
    DocumentUploadResponse,
    DocumentDetailResponse,
    DocumentTextResponse,
    DocumentListItem,
    DocumentListResponse,
    PageListItem,
    ElementListItem,
)
from config.settings import settings
from data.db_models import User, Task
from data.repositories import DocumentRepository
from data.repositories.document_repo import delete_document_cascade
from services.document_service import DocumentService
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

    return DocumentUploadResponse(
        document_id=doc.id,
        title=doc.title,
        format=doc.format,
        total_pages=doc.total_pages,
        processing_status=doc.processing_status,
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
        task_id, reused = _doc_svc.submit_extraction(db, document_id)
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
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get OCR and/or normalized text for a document (own docs only; admin can access all)."""
    get_authorized_document(document_id, user, db)
    repo = DocumentRepository(db)
    dt = repo.get_digitized_text(document_id)
    return DocumentTextResponse(
        document_id=document_id,
        ocr_content=dt.ocr_content if dt else None,
        normalized_content=dt.normalized_content if dt else None,
    )


# ── Download OCR / normalized text as file ──────────────────────────

@router.get("/{document_id}/text/download")
async def download_document_text(
    document_id: str,
    type: str = Query("ocr", pattern="^(ocr|normalized)$"),
    mode: str = Query(
        "auto",
        pattern="^(auto|markdown|spatial|plain)$",
        description="auto=spatial when layout elements exist, else markdown; plain=legacy line dump",
    ),
    format: str = Query(
        "docx",
        pattern="^(docx|pdf)$",
        description="docx=Word export; pdf=spatial docx converted via LibreOffice (scanned) or original PDF",
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
    Download document text or the original uploaded file.

    - DOCX / DOC: default (`source=auto`) returns the **original uploaded file**
      (no markdown round-trip — preserves layout, tables, styles).
    - PDF / image: default builds a structured .docx from extracted/OCR text.
    - `source=extracted` forces export from digitized text (e.g. after manual correction).
    """
    from utils.file_download import (
        build_docx_bytes_from_content,
        build_docx_bytes_from_elements,
        build_docx_response,
        build_docx_response_from_elements,
        build_original_file_response,
        build_pdf_response,
        docx_bytes_to_pdf_bytes,
        is_native_word_document,
        safe_filename,
    )

    repo = DocumentRepository(db)
    doc = get_authorized_document(document_id, user, db)

    use_original = source == "original" or (
        source == "auto" and is_native_word_document(doc.format)
    )
    if use_original:
        download_name = doc.original_filename or os.path.basename(doc.file_path or "")
        fmt = (doc.format or "").lower()
        if format == "pdf":
            if fmt == "pdf":
                return build_original_file_response(doc.file_path, download_name=download_name)
            if is_native_word_document(fmt):
                if not doc.file_path or not os.path.isfile(doc.file_path):
                    raise HTTPException(status_code=404, detail="Original file not found on disk.")
                try:
                    with open(doc.file_path, "rb") as f:
                        docx_bytes = f.read()
                    if fmt == "doc":
                        from services.extractors.doc_converter import convert_doc_to_docx

                        docx_path = convert_doc_to_docx(doc.file_path)
                        with open(docx_path, "rb") as f:
                            docx_bytes = f.read()
                    pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
                    pdf_name = f"{os.path.splitext(download_name)[0]}.pdf"
                    return build_pdf_response(pdf_name, pdf_bytes)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="PDF export of original document requires LibreOffice",
                    ) from exc
            raise HTTPException(
                status_code=400,
                detail="PDF export not supported for this document format",
            )
        return build_original_file_response(doc.file_path, download_name=download_name)

    dt = repo.get_digitized_text(document_id)
    if not dt:
        raise HTTPException(status_code=404, detail="No extracted text found. Run /extract first.")

    content = dt.ocr_content if type == "ocr" else dt.normalized_content
    if not content:
        raise HTTPException(status_code=404, detail=f"No {type} content available.")

    base = f"{type}_{safe_filename(doc.title)}"
    filename = f"{base}.docx"

    use_spatial = mode in ("auto", "spatial")
    text_overridden = bool(getattr(dt, "text_overridden", False))
    spatial_cap = settings.ocr_download_spatial_max_elements
    elements = []
    element_count = 0
    if use_spatial and not text_overridden:
        element_count = repo.count_elements(document_id)
        if element_count > 0 and element_count <= spatial_cap:
            elements = repo.get_elements(document_id)
    if (
        use_spatial
        and elements
        and mode != "markdown"
    ):
        if format == "pdf":
            try:
                docx_bytes = build_docx_bytes_from_elements(elements, title=doc.title)
                pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
                return build_pdf_response(f"{base}.pdf", pdf_bytes)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc
        return build_docx_response_from_elements(filename, elements, title=doc.title)

    structured = mode != "plain"
    if format == "pdf":
        try:
            docx_bytes = build_docx_bytes_from_content(
                content,
                title=doc.title,
                structured=structured,
            )
            pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
            return build_pdf_response(f"{base}.pdf", pdf_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

    return build_docx_response(
        filename,
        content,
        title=doc.title,
        structured=structured,
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
        result.append(ElementListItem(
            id=elem.id,
            label=elem.label,
            text_content=elem.text_content,
            bbox={
                "x1": elem.bbox_x1,
                "y1": elem.bbox_y1,
                "x2": elem.bbox_x2,
                "y2": elem.bbox_y2,
            },
            bbox_normalized={
                "x1": elem.bbox_norm_x1,
                "y1": elem.bbox_norm_y1,
                "x2": elem.bbox_norm_x2,
                "y2": elem.bbox_norm_y2,
            } if elem.bbox_norm_x1 is not None else None,
            page_number=elem.page.page_number if elem.page else None,
            page_id=elem.page_id,
            sequence_order=elem.sequence_order,
            has_crop_image=bool(elem.crop_image_base64),
        ))
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
    paths_to_unlink = repo.collect_file_paths(document_id)

    await asyncio.to_thread(delete_document_cascade, document_id)

    for path in paths_to_unlink:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    doc_dir = os.path.join(settings.upload_dir, document_id)
    if os.path.isdir(doc_dir):
        try:
            shutil.rmtree(doc_dir, ignore_errors=True)
        except OSError:
            pass
