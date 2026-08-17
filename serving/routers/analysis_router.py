"""
Full digest analysis pipeline — starts Temporal DigestPipelineWorkflow.

POST /api/v2/documents/{id}/analysis
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from data.db_models import User
from services.pipeline.admission import AdmissionRejected, http_exception
from services.pipeline.temporal_client import cancel_digest_workflow, start_digest_workflow

router = APIRouter(prefix="/api/v2/documents", tags=["analysis"])


@router.post("/{document_id}/analysis")
async def start_full_analysis(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Start durable digest pipeline via Temporal (BUILD_TREE → analysis stages).
    Returns single pipeline_id / task_id for UI polling.
    """
    doc = get_authorized_document(document_id, _user, db)
    if doc.processing_status != "EXTRACTED":
        # Manual stage: only valid once OCR/extraction has completed. A
        # premature start would just sit in the workflow's wait-gate showing
        # RUNNING, which users read as a stuck pipeline.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Tài liệu chưa OCR xong (trạng thái: {doc.processing_status}) — "
                "hãy đợi OCR hoàn thành trước khi chạy tổng thuật."
            ),
        )

    try:
        workflow_id, parent_task_id = await start_digest_workflow(
            document_id, fairness_key=_user.id
        )
    except AdmissionRejected as exc:
        raise http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Temporal pipeline unavailable: {exc}",
        ) from exc

    return {
        "pipeline_id": workflow_id,
        "workflow_id": workflow_id,
        "task_id": parent_task_id,
        "task_ids": [parent_task_id],
        "message": "Digest pipeline submitted (Temporal)",
    }


@router.delete("/{document_id}/analysis")
async def cancel_full_analysis(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Cancel a running digest pipeline."""
    get_authorized_document(document_id, _user, db)
    cancelled = await cancel_digest_workflow(document_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="No running digest workflow to cancel")

    from services.pipeline.mirror import update_pipeline_mirror

    update_pipeline_mirror(
        document_id,
        state="FAILED",
        message="Cancelled by user",
    )
    return {"cancelled": True, "document_id": document_id}
