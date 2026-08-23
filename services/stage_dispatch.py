"""Single-stage rerun dispatch: registry, durability policy, Temporal submit.

A standalone "rerun keywords / rerun summary" is the *same work* as the
corresponding digest stage — only the dispatch differed. The digest path ran it
as a Temporal activity (timeout, heartbeat, retry, resume, survives an API
restart); the standalone path ran it as a bare ``asyncio.create_task`` inside
uvicorn, which on a several-hundred-page book meant hours of work with no
timeout, no death detection, no retry and no resume, discarded by any deploy.

This module routes both through the same durability guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Awaitable, Callable, Optional

from temporalio.common import RetryPolicy

from services.pipeline import stage_runners
from workflows.timeouts import HEARTBEAT, LONG_RUN

# stage name (== Task.task_type) → runner(document_id, task_id) -> dict
STAGE_RUNNERS: dict[str, Callable[..., Awaitable[dict]]] = {
    "BUILD_TREE": stage_runners.run_build_tree,
    "BIBLIOGRAPHIC": stage_runners.run_bibliographic,
    "KEYWORDS": stage_runners.run_keywords,
    "RESEARCH_DIRECTIONS": stage_runners.run_research_directions,
    "USAGE_SCOPE": stage_runners.run_usage_scope,
    "HIERARCHICAL_SUMMARIZE": stage_runners.run_summarize,
    "MAIN_CONTENT": stage_runners.run_main_content,
}

# Stages that legitimately run for hours on book-length documents.
LONG_STAGES = {"BUILD_TREE", "HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"}


@dataclass(frozen=True)
class StagePolicy:
    start_to_close: timedelta
    retry: RetryPolicy
    # Set once the stage activity is RUNNING (Temporal started). Queued DB
    # waiters never reach this policy. Optional stall_timeout stays off —
    # heartbeat is process liveness, not a progress deadline.
    heartbeat: Optional[timedelta] = None
    stall_timeout: Optional[timedelta] = None


# Stall detection off by default — user-facing work should queue and finish,
# not FAIL after 45 quiet minutes during a slow LLM stretch.
STALL_TIMEOUTS: dict[str, timedelta] = {}


_LONG = StagePolicy(
    start_to_close=LONG_RUN,
    heartbeat=HEARTBEAT,
    retry=RetryPolicy(
        maximum_attempts=6,
        initial_interval=timedelta(seconds=30),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(minutes=5),
    ),
)

_SHORT = StagePolicy(
    start_to_close=LONG_RUN,
    heartbeat=HEARTBEAT,
    retry=RetryPolicy(maximum_attempts=2),
)


def stage_policy(stage: str) -> StagePolicy:
    """Durability settings for *stage*. Raises KeyError for unknown stages so a
    typo fails at submit time rather than starting a workflow nothing runs."""
    if stage not in STAGE_RUNNERS:
        raise KeyError(f"Unknown rerun stage: {stage}")
    base = _LONG if stage in LONG_STAGES else _SHORT
    stall = STALL_TIMEOUTS.get(stage)
    return replace(base, stall_timeout=stall) if stall else base


def stage_workflow_id(document_id: str, stage: str) -> str:
    return f"stage-{document_id}-{stage}"


def create_stage_task(
    db, document_id: str, stage: str, *, fairness_key: Optional[str] = None, options: Optional[dict] = None
) -> str:
    """Create the PENDING Task row a stage rerun reports into.

    LONG stages are marked ``queued`` so they share the digest capacity pool
    instead of HTTP 429 when pipelines are full.
    """
    from data.db_models import Task
    from data.id_generator import IdGenerator
    from services.pipeline.admission import mark_queued

    raw_id = IdGenerator.next_id(db, "tasks")
    task_id = f"{stage}_{raw_id.split('_')[-1]}"
    task = Task(
        id=task_id,
        document_id=document_id,
        task_type=stage,
        status="PENDING",
        progress=0,
        message="Đang chờ xử lý",
    )
    if stage in LONG_STAGES:
        mark_queued(
            task,
            extra={"fairness_key": fairness_key, "stage_options": options},
            message="Đang chờ máy rảnh…",
        )
    db.add(task)
    db.commit()
    return task_id


def close_superseded_stage_tasks(db, document_id: str, stage: str, keep_task_id: str) -> int:
    """Fail this document's still-open rows for *stage* other than the new one.

    Restarting a stage terminates the previous workflow; without this its Task
    row stays PENDING/RUNNING forever and the UI shows a run that no longer
    exists.
    """
    from data.db_models import Task
    from services.task_manager import TaskManager

    stale = (
        db.query(Task)
        .filter(
            Task.document_id == document_id,
            Task.task_type == stage,
            Task.status.in_(("PENDING", "RUNNING")),
            Task.id != keep_task_id,
        )
        .all()
    )
    for task in stale:
        TaskManager.mark_terminal(
            db,
            task.id,
            status="FAILED",
            error=((task.error or "") + "\nSuperseded by a newer run.").strip(),
            commit=False,
        )
    if stale:
        db.commit()
    return len(stale)


async def submit_stage_with_resource(
    db,
    document_id: str,
    stage: str,
    model,
    **row_kwargs,
) -> tuple[str, Optional[str], bool]:
    """Shared submit for the stage services: create/reuse the domain row, then
    start a durable rerun. Returns ``(task_id, resource_id, reused)``.

    Dedupe asks Temporal whether the previous run is still alive, unlike
    ``task_manager.get_active_task_id`` which consults an in-process dict that
    can never contain worker-side work. Queued LONG stages have no Temporal
    workflow yet — reuse + kick instead of treating them as dead.
    """
    from config.capacity import SLOT_DIGEST
    from data.db_models import Task
    from data.repositories import DocumentRepository
    from services.pipeline.admission import is_queued
    from services.pipeline.job_queue import kick_queue
    from services.pipeline.temporal_client import is_stage_running

    if not DocumentRepository(db).get(document_id):
        raise ValueError("Document not found")

    def _latest_resource_id():
        row = (
            db.query(model)
            .filter(model.document_id == document_id)
            .order_by(model.created_at.desc())
            .first()
        )
        return row.id if row else None

    active = (
        db.query(Task)
        .filter(
            Task.document_id == document_id,
            Task.task_type == stage,
            Task.status.in_(("PENDING", "RUNNING")),
        )
        .order_by(Task.created_at.desc())
        .first()
    )
    if active:
        if is_queued(active):
            kick_queue(SLOT_DIGEST)
            return active.id, _latest_resource_id(), True
        if await is_stage_running(document_id, stage):
            return active.id, _latest_resource_id(), True

    row = model(document_id=document_id, status="PENDING", **row_kwargs)
    db.add(row)
    db.commit()
    db.refresh(row)
    resource_id = row.id

    task_id = await submit_stage(db, document_id, stage)
    return task_id, resource_id, False


async def submit_stage(
    db,
    document_id: str,
    stage: str,
    *,
    options: Optional[dict] = None,
    fairness_key: Optional[str] = None,
) -> str:
    """Queue (LONG) or start (short) a durable rerun of *stage*; return task id.

    With ``stage_rerun_use_temporal`` off, callers keep their legacy
    ``task_manager.submit`` path — this function is only the Temporal branch.
    """
    from config.capacity import SLOT_DIGEST
    from services.pipeline.job_queue import kick_queue
    from services.pipeline.temporal_client import start_stage_workflow

    stage_policy(stage)  # fail fast on unknown stage
    task_id = create_stage_task(db, document_id, stage, fairness_key=fairness_key, options=options)
    close_superseded_stage_tasks(db, document_id, stage, task_id)
    if stage in LONG_STAGES:
        kick_queue(SLOT_DIGEST)
        return task_id
    await start_stage_workflow(
        document_id, stage, task_id, options=options, fairness_key=fairness_key
    )
    return task_id
