"""Update documents.pipeline_* mirror for UI polling."""

from datetime import datetime
from typing import Optional

from data.database import get_db_manager
from data.db_models import Document, Task
from services.pipeline.constants import STAGE_WEIGHTS, aggregate_progress
from services.task_manager import TaskManager


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
    structured_progress: Optional[dict] = None,
    attempt: int = 1,
) -> None:
    """Persist pipeline status to documents + optional parent DIGEST_PIPELINE task."""
    db_manager = get_db_manager()
    with db_manager.session() as db:
        doc = db.query(Document).filter(Document.id == document_id).with_for_update().first()
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

        doc.updated_at = datetime.utcnow()

        if parent_task_id:
            task = db.query(Task).filter(Task.id == parent_task_id).with_for_update().first()
            if task:
                existing_meta = task.progress_meta if isinstance(task.progress_meta, dict) else {}
                stage_map = dict(existing_meta.get("stages") or {})
                weighted_stages = {name: 0 for name in STAGE_WEIGHTS}
                for name, value in stage_map.items():
                    weighted_stages[name] = int((value or {}).get("progress") or 0)
                for name, value in (completed_stages or {}).items():
                    weighted_stages[name] = max(weighted_stages.get(name, 0), int(value or 0))

                if stage:
                    previous = dict(stage_map.get(stage) or {})
                    incoming = dict(structured_progress or {})
                    stage_entry = {
                        **previous,
                        "phase": incoming.get("phase")
                        or ("exporting" if stage == "FINALIZE" else "active"),
                        "attempt": int(incoming.get("attempt") or attempt or 1),
                        "progress": max(
                            int(previous.get("progress") or 0),
                            int(stage_progress or 0),
                        ),
                    }
                    regressive_units = (
                        previous.get("attempt") == stage_entry["attempt"]
                        and incoming.get("units_done") is not None
                        and int(incoming["units_done"]) < int(previous.get("units_done") or 0)
                    )
                    if not regressive_units:
                        for key in ("unit_kind", "units_done", "units_total"):
                            if incoming.get(key) is not None:
                                stage_entry[key] = incoming[key]
                    stage_map[stage] = stage_entry
                    weighted_stages[stage] = max(
                        weighted_stages.get(stage, 0),
                        stage_entry["progress"],
                    )

                doc.pipeline_progress = aggregate_progress(weighted_stages)
                phase = (
                    "waiting_upstream"
                    if structured_progress
                    and structured_progress.get("phase") == "waiting_upstream"
                    else ("exporting" if stage == "FINALIZE" else "active")
                )
                current_stage = stage_map.get(stage or "", {})
                meta = {
                    "version": 1,
                    "pipeline": "digest",
                    "phase": phase,
                    "mode": "digest_pipeline",
                    "stage": stage or existing_meta.get("stage"),
                    "unit_kind": current_stage.get("unit_kind"),
                    "units_done": current_stage.get("units_done"),
                    "units_total": current_stage.get("units_total"),
                    "attempt": current_stage.get("attempt") or attempt or 1,
                    "stages": stage_map,
                }
                if state == "DONE":
                    TaskManager.mark_terminal(
                        db,
                        task.id,
                        status="COMPLETED",
                        result=task_result,
                        message=message or "Tổng thuật hoàn tất",
                        commit=False,
                    )
                elif state == "FAILED":
                    TaskManager.mark_terminal(
                        db,
                        task.id,
                        status="FAILED",
                        error=message,
                        message=message,
                        commit=False,
                    )
                else:
                    TaskManager.update_progress(
                        db,
                        task.id,
                        doc.pipeline_progress or 0,
                        message or task.message or "",
                        meta,
                        commit=False,
                    )
        elif completed_stages is not None:
            doc.pipeline_progress = aggregate_progress(completed_stages)
        elif stage_progress is not None:
            doc.pipeline_progress = aggregate_progress({stage: stage_progress} if stage else {})

        db.commit()

    if state in ("DONE", "FAILED"):
        from config.capacity import SLOT_DIGEST
        from services.pipeline.job_queue import kick_queue

        kick_queue(SLOT_DIGEST)


def make_stage_progress_sink(
    document_id: str,
    parent_task_id: str,
    stage: str,
    completed_stages: Optional[dict[str, int]] = None,
):
    """Forward fine-grained service units into the digest parent stage map."""

    def sink(progress: int, message: str, meta: dict) -> None:
        update_pipeline_mirror(
            document_id,
            state="RUNNING",
            stage=stage,
            stage_progress=progress,
            message=message,
            parent_task_id=parent_task_id,
            completed_stages=completed_stages,
            structured_progress=meta,
            attempt=int(meta.get("attempt") or 1),
        )

    return sink


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
        message="Đã khởi động tiến trình tổng thuật",
        parent_task_id=parent_task_id,
        completed_stages={k: 0 for k in STAGE_WEIGHTS},
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "waiting_upstream",
            "mode": "digest_pipeline",
            "stage": "BUILD_TREE",
            "attempt": 1,
        },
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
