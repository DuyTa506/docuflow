"""
Full digest analysis pipeline — starts Temporal DigestPipelineWorkflow.

POST /api/v2/documents/{id}/analysis
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_authorized_document, get_current_user, get_db
from data.db_models import User
from services.pipeline.temporal_client import start_digest_workflow

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
    get_authorized_document(document_id, _user, db)

    try:
        workflow_id, parent_task_id = await start_digest_workflow(
            document_id, fairness_key=_user.id
        )
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
