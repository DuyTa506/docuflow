"""
Summarization endpoints.

POST /api/v2/documents/{id}/summaries              — Generate summary (background)
GET  /api/v2/documents/{id}/summaries              — List summaries (with status)
GET  /api/v2/documents/{id}/summaries/{sid}        — Get specific summary
GET  /api/v2/documents/{id}/summaries/{sid}/download — Download as .docx file
POST /api/v2/documents/{id}/summaries/{sid}/upload — Override via .txt/.docx file
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from api.schemas import (
    SummaryRequest,
    SummaryResponse,
    SummaryListItem,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, SummaryRepository
from services.summarization_service import SummarizationService
from utils.file_download import build_docx_response, safe_filename
from utils.file_upload import extract_text_from_upload

router = APIRouter(prefix="/api/v2/documents", tags=["summaries"])
_svc = SummarizationService()


@router.post("/{document_id}/summaries", response_model=TaskSubmittedResponse)
async def start_summarization(
    document_id: str,
    body: SummaryRequest = SummaryRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Generate a document summary as a background task."""
    get_authorized_document(document_id, _user, db)
    try:
        task_id, summary_id, reused = _svc.submit(db, document_id, body.summary_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(
        task_id=task_id,
        resource_id=summary_id,
        message="Summarization already in progress" if reused else "Summarization task submitted",
    )


@router.get("/{document_id}/summaries", response_model=list[SummaryListItem])
async def list_summaries(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all summaries for a document."""
    get_authorized_document(document_id, _user, db)
    summary_repo = SummaryRepository(db)
    summaries = summary_repo.list(document_id)
    return [
        SummaryListItem(
            id=s.id,
            document_id=s.document_id,
            summary_type=s.summary_type,
            content=s.content,
            status=s.status,
            created_at=s.created_at.isoformat() if s.created_at else None,
            updated_at=s.updated_at.isoformat() if s.updated_at else None,
        )
        for s in summaries
    ]


@router.get("/{document_id}/summaries/{summary_id}", response_model=SummaryResponse)
async def get_summary(
    document_id: str,
    summary_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get a specific summary (incl. status)."""
    get_authorized_document(document_id, _user, db)
    s = SummaryRepository(db).get(summary_id, document_id)
    if not s:
        raise HTTPException(status_code=404, detail="Summary not found")
    return SummaryResponse(
        id=s.id,
        document_id=s.document_id,
        summary_type=s.summary_type,
        content=s.content,
        status=s.status,
        created_at=s.created_at.isoformat() if s.created_at else None,
        updated_at=s.updated_at.isoformat() if s.updated_at else None,
    )


@router.get("/{document_id}/summaries/{summary_id}/download")
async def download_summary(
    document_id: str,
    summary_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Download a summary as a .docx file."""
    doc = get_authorized_document(document_id, user, db)

    s = SummaryRepository(db).get(summary_id, document_id)
    if not s:
        raise HTTPException(status_code=404, detail="Summary not found")
    if not s.content:
        raise HTTPException(status_code=404, detail="Summary has no content")
    if s.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Summary is not yet complete")

    filename = f"summary_{safe_filename(doc.title)}.docx"
    return build_docx_response(
        filename, s.content, title=doc.title, headings=["Summary"]
    )


@router.post("/{document_id}/summaries/{summary_id}/upload", response_model=SummaryListItem)
async def upload_summary(
    document_id: str,
    summary_id: str,
    file: UploadFile = File(..., description="Corrected summary as .txt or .docx"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Override summary content by uploading a corrected .txt or .docx file.
    """
    get_authorized_document(document_id, _user, db)
    text = await extract_text_from_upload(file)
    summary_repo = SummaryRepository(db)
    s = summary_repo.update(summary_id, document_id, text)
    if not s:
        raise HTTPException(status_code=404, detail="Summary not found")
    return SummaryListItem(
        id=s.id,
        document_id=s.document_id,
        summary_type=s.summary_type,
        content=s.content,
        status=s.status,
        created_at=s.created_at.isoformat() if s.created_at else None,
        updated_at=s.updated_at.isoformat() if s.updated_at else None,
    )
