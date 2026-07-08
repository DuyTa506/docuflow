"""Pipeline status for digest workflow (UI polling)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from data.db_models import Document, Task, User
from services.pipeline.constants import STAGE_LABELS

router = APIRouter(prefix="/api/v2/documents", tags=["pipeline"])


@router.get("/{document_id}/pipeline-status")
async def get_pipeline_status(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Aggregate digest pipeline status from documents.pipeline_* mirror."""
    get_authorized_document(document_id, _user, db)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    parent = (
        db.query(Task)
        .filter(
            Task.document_id == document_id,
            Task.task_type == "DIGEST_PIPELINE",
        )
        .order_by(Task.created_at.desc())
        .first()
    )

    stage = doc.pipeline_stage or ""
    return {
        "document_id": document_id,
        "workflow_id": doc.pipeline_workflow_id,
        "pipeline_id": doc.pipeline_workflow_id,
        "task_id": parent.id if parent else None,
        "state": doc.pipeline_state or "IDLE",
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "progress": doc.pipeline_progress or 0,
        "message": doc.pipeline_message or "",
        "quality_report": doc.quality_report,
        "parent_task": {
            "task_id": parent.id,
            "status": parent.status,
            "progress": parent.progress,
            "message": parent.message,
        } if parent else None,
    }
