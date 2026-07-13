"""
Digest endpoints.

POST /api/v2/documents/{id}/digest          → assemble + return JSON
GET  /api/v2/documents/{id}/digest/download → assemble + return .docx file
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from data.db_models import User
from services.digest_renderer import DigestRenderer
from services.digest_service import DigestService
from services.export_service import export_service
from utils.file_download import build_stored_file_response

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
    Assemble digest and return a formatted .docx file for download (cached in MinIO).

    The file follows the official 'Mau Tong thuat Book' template.
    Sections that haven't been processed yet are left blank with a note.
    """
    get_authorized_document(document_id, _user, db)

    try:
        key, filename, media_type = await asyncio.to_thread(
            export_service.get_or_build_digest_export,
            db,
            document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Digest export failed: {exc}") from exc

    return await asyncio.to_thread(
        build_stored_file_response, key, download_name=filename, content_type=media_type
    )
