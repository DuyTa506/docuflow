"""
Full digest analysis pipeline — starts Temporal DigestPipelineWorkflow.

POST /api/v2/documents/{id}/analysis
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from data.db_models import User
from services.pipeline.job_queue import kick_queue, submit_digest
from services.pipeline.temporal_client import cancel_digest_workflow

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
        workflow_id, parent_task_id, reused = await submit_digest(
            db, document_id, fairness_key=_user.id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "pipeline_id": workflow_id,
        "workflow_id": workflow_id,
        "task_id": parent_task_id,
        "task_ids": [parent_task_id],
        "message": ("Tổng thuật đang chạy" if reused else "Đã gửi tác vụ tổng thuật"),
    }


@router.delete("/{document_id}/analysis")
async def cancel_full_analysis(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Cancel digest: stop Temporal if running, always free OPEN parent Task."""
    get_authorized_document(document_id, _user, db)
    from services.pipeline.mirror import update_pipeline_mirror
    from services.task_manager import TaskManager

    cancelled_wf = await cancel_digest_workflow(document_id)
    task = TaskManager.fail_latest_open(db, document_id, "DIGEST_PIPELINE", commit=False)
    if not cancelled_wf and task is None:
        raise HTTPException(status_code=409, detail="Không có tác vụ tổng thuật đang chạy để hủy")
    db.commit()
    update_pipeline_mirror(
        document_id,
        state="FAILED",
        message="Đã hủy theo yêu cầu người dùng",
        parent_task_id=task.id if task else None,
    )
    from config.capacity import SLOT_DIGEST

    kick_queue(SLOT_DIGEST)
    return {"cancelled": True, "document_id": document_id}
