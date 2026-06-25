"""
Digest endpoints.

POST /api/v2/documents/{id}/digest          → assemble + return JSON
GET  /api/v2/documents/{id}/digest/download → assemble + return .docx file
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from data.db_models import User
from services.digest_service import DigestService
from services.digest_renderer import DigestRenderer

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
        "abstract": digest.abstract,
        "main_content": digest.main_content,
        "keywords": [{"keyword": k.keyword, "weight": k.weight} for k in digest.keywords],
        "research_directions": [
            {"direction_name": d.direction_name, "confidence": d.confidence}
            for d in digest.research_directions
        ],
        "missing": digest.missing,
    }


# ── DOCX download ─────────────────────────────────────────────────────

@router.get("/{document_id}/digest/download")
async def download_digest(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Assemble digest and return a formatted .docx file for download.

    The file follows the official 'Mau Tong thuat Book' template.
    Sections that haven't been processed yet are left blank with a note.
    """
    get_authorized_document(document_id, _user, db)
    try:
        digest = _digest_svc.assemble(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    docx_bytes = _renderer.render(digest)

    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in digest.title)[:60]
    filename = f"digest_{safe_title}.docx"

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
