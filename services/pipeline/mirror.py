"""Update documents.pipeline_* mirror for UI polling."""

from datetime import datetime
from typing import Optional

from data.database import get_db_manager
from data.db_models import Document, Task
from services.pipeline.constants import STAGE_WEIGHTS, aggregate_progress


def update_pipeline_mirror(
    document_id: str,
    *,
    workflow_id: Optional[str] = None,
    state: Optional[str] = None,
    stage: Optional[str] = None,
    stage_progress: Optional[int] = None,
    message: Optional[str] = None,
    parent_task_id: Optional[str] = None,
    completed_stages: Optional[dict[str, int]] = None,
    quality_report: Optional[dict] = None,
    task_result: Optional[dict] = None,
) -> None:
    """Persist pipeline status to documents + optional parent DIGEST_PIPELINE task."""
    db_manager = get_db_manager()
    with db_manager.session() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        if workflow_id is not None:
            doc.pipeline_workflow_id = workflow_id
        if state is not None:
            doc.pipeline_state = state
        if stage is not None:
            doc.pipeline_stage = stage
        if message is not None:
            doc.pipeline_message = message
        if quality_report is not None:
            doc.quality_report = quality_report

        if completed_stages is not None:
            if stage and stage_progress is not None:
                completed_stages = {**completed_stages, stage: stage_progress}
            doc.pipeline_progress = aggregate_progress(completed_stages)
        elif stage_progress is not None:
            stages = {stage: stage_progress} if stage else {}
            doc.pipeline_progress = aggregate_progress(stages)

        doc.updated_at = datetime.utcnow()

        if parent_task_id:
            task = db.query(Task).filter(Task.id == parent_task_id).first()
            if task:
                if state == "RUNNING":
                    task.status = "RUNNING"
                elif state == "DONE":
                    task.status = "COMPLETED"
                    task.progress = 100
                elif state == "FAILED":
                    task.status = "FAILED"
                else:
                    task.status = task.status or "RUNNING"
                if doc.pipeline_progress is not None:
                    task.progress = doc.pipeline_progress
                if state == "DONE":
                    task.progress = 100
                if task_result is not None:
                    task.result = task_result
                if message:
                    task.message = message
                task.updated_at = datetime.utcnow()

        db.commit()


def init_pipeline_run(
    document_id: str,
    workflow_id: str,
    parent_task_id: str,
) -> None:
    update_pipeline_mirror(
        document_id,
        workflow_id=workflow_id,
        state="RUNNING",
        stage="BUILD_TREE",
        stage_progress=0,
        message="Pipeline started",
        parent_task_id=parent_task_id,
        completed_stages={k: 0 for k in STAGE_WEIGHTS},
    )


def mark_stage_complete(
    document_id: str,
    stage: str,
    parent_task_id: Optional[str] = None,
    completed_stages: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    stages = dict(completed_stages or {k: 0 for k in STAGE_WEIGHTS})
    stages[stage] = 100
    update_pipeline_mirror(
        document_id,
        stage=stage,
        stage_progress=100,
        completed_stages=stages,
        parent_task_id=parent_task_id,
    )
    return stages
