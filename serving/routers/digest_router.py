"""
Digest endpoints.

POST /api/v2/documents/{id}/digest          → assemble + return JSON
GET  /api/v2/documents/{id}/digest/download → assemble + return .docx file
GET  /api/v2/documents/{id}/digest/admin    → read "Thông tin quản trị CSDL"
PUT  /api/v2/documents/{id}/digest/admin    → set it
"""

import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from data.db_models import Document, User
from services.digest_renderer import DigestRenderer
from services.digest_service import DigestService
from services.export_service import export_service
from utils.digest_admin import normalize_digest_admin
from utils.doc_kind import DOC_KINDS, normalize_doc_kind, resolve_doc_kind
from utils.file_download import build_bytes_file_response, build_stored_file_response

router = APIRouter(prefix="/api/v2/documents", tags=["digest"])

_digest_svc = DigestService()
_renderer = DigestRenderer()


# ── JSON preview ─────────────────────────────────────────────────────


@router.post("/{document_id}/digest")
async def get_digest(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Assemble a digest from all existing DB results for this document.

    Returns a JSON representation of the digest.
    Missing sections are listed under the 'missing' key — run the
    relevant services first (summarize, main_content, keywords,
    research_directions) and call this endpoint again.
    """
    get_authorized_document(document_id, _user, db)
    try:
        digest = _digest_svc.assemble(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "document_id": digest.document_id,
        "title": digest.title,
        "source_language": digest.source_language,
        "original_filename": digest.original_filename,
        "bibliographic": digest.bibliographic,
        "abstract": digest.abstract,
        "main_content": digest.main_content,
        "chapters": [
            {
                "number": c.number,
                "title_vi": c.title_vi,
                "title_original": c.title_original,
                "content": c.content,
            }
            for c in digest.chapters
        ],
        "keywords": [
            {"keyword": k.keyword, "display": k.display, "weight": k.weight}
            for k in digest.keywords
        ],
        "usage_scope": digest.usage_scope,
        "research_directions": [
            {"direction_name": d.direction_name, "confidence": d.confidence}
            for d in digest.research_directions
        ],
        "digest_admin": {
            "reviewer": digest.reviewer,
            "reviewer_approved": digest.reviewer_approved,
            "entry_date": digest.entry_date,
        },
        "missing": digest.missing,
    }


# ── Admin block ──────────────────────────────────────────────────────


def _document_or_404(document_id: str, db: Session) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/digest/admin")
async def get_digest_admin(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Read the librarian-entered "Thông tin quản trị CSDL" block."""
    get_authorized_document(document_id, _user, db)
    return normalize_digest_admin(_document_or_404(document_id, db).digest_admin)


@router.put("/{document_id}/digest/admin")
async def set_digest_admin(
    document_id: str,
    body: dict = Body(..., description="reviewer / reviewer_approved / entry_date"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Set the admin block. Nothing derives these — they are entered by hand."""
    get_authorized_document(document_id, _user, db)
    doc = _document_or_404(document_id, db)

    try:
        cleaned = normalize_digest_admin(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    doc.digest_admin = cleaned
    db.commit()
    # The rendered digest is cached in MinIO by document id; leaving it in
    # place would keep serving the version with the block still empty.
    export_service.invalidate_digest_export(document_id)
    return cleaned


# ── Document kind (book / kỷ yếu) ─────────────────────────────────────


@router.get("/{document_id}/digest/kind")
async def get_digest_kind(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Thể loại đang áp dụng cho §2.1/§2.2, và nó đến từ đâu.

    `doc_kind_source`: `explicit` người dùng đặt · `detected` nhận diện được ·
    `default` không có tín hiệu nào, coi là sách.
    """
    get_authorized_document(document_id, _user, db)
    doc = _document_or_404(document_id, db)
    return {
        **resolve_doc_kind(doc.digest_doc_kind, title=doc.title or ""),
        "override": doc.digest_doc_kind,
        "allowed": list(DOC_KINDS),
    }


@router.put("/{document_id}/digest/kind")
async def set_digest_kind(
    document_id: str,
    body: dict = Body(..., description='{"doc_kind": "book" | "proceedings" | null}'),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Ghi đè thể loại. `null` trả về chế độ tự nhận diện.

    Đổi thể loại không tự viết lại §2.2 — các mục đã được tóm tắt bằng prompt
    của thể loại cũ. Chạy lại `POST .../main-content` để sinh lại nội dung.
    """
    get_authorized_document(document_id, _user, db)
    doc = _document_or_404(document_id, db)

    try:
        kind = normalize_doc_kind(body.get("doc_kind"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    doc.digest_doc_kind = kind
    db.commit()
    export_service.invalidate_digest_export(document_id)
    return {
        **resolve_doc_kind(kind, title=doc.title or ""),
        "override": kind,
        "rerun_required": "POST /api/v2/documents/{id}/main-content",
    }


# ── DOCX download ─────────────────────────────────────────────────────


@router.get("/{document_id}/digest/download")
async def download_digest(
    document_id: str,
    format: str = Query("docx", pattern="^(docx|pdf)$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Assemble digest and return a formatted .docx or .pdf file for download
    (cached in MinIO).

    The file follows the official 'Mau Tong thuat Book' template.
    Sections that haven't been processed yet are left blank with a note.
    """
    get_authorized_document(document_id, _user, db)

    try:
        key, filename, media_type, data = await asyncio.to_thread(
            export_service.get_or_build_digest_export,
            db,
            document_id,
            format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Digest export failed: {exc}") from exc

    if data is not None:
        export_service.schedule_export_put(key, data, content_type=media_type)
        return build_bytes_file_response(data, filename, media_type)

    return await asyncio.to_thread(
        build_stored_file_response, key, download_name=filename, content_type=media_type
    )
