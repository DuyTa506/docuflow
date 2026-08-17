"""Activity for standalone single-stage reruns.

One generic activity dispatches on stage name, so adding a rerunnable stage is
a registry entry in services/stage_dispatch.py rather than a new activity plus
a new worker registration to forget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from temporalio import activity

from workflows.activities._common import _with_heartbeat


@dataclass
class StageRerunInput:
    document_id: str
    stage: str
    task_id: str
    # Per-request overrides (e.g. /tree-index build flags). Dropping these
    # would silently ignore what the caller asked for.
    options: Optional[dict] = None


def _finish_task(
    task_id: str, *, status: str, result: Optional[dict], error: Optional[str]
) -> None:
    from data.database import get_db_manager
    from services.task_manager import TaskManager

    with get_db_manager().session() as db:
        TaskManager.mark_terminal(
            db,
            task_id,
            status=status,
            result=result if isinstance(result, (dict, list)) else {"detail": str(result)},
            error=error,
        )


def _progress_probe(task_id: str):
    """Probe whose value advances whenever the stage reports progress.

    ``(progress, message)`` rather than ``updated_at``: the timestamp also
    moves on writes that aren't progress, which would mask a real stall.
    """

    def _probe():
        from data.database import get_db_manager
        from data.db_models import Task

        with get_db_manager().session() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            return (task.progress, task.message) if task else None

    return _probe


@activity.defn(name="run_stage")
async def run_stage_activity(inp: StageRerunInput) -> dict:
    """Run one stage, heartbeating so a genuine hang is detected in minutes.

    Terminal Task bookkeeping happens here because the Temporal path has no
    in-process ``_run_wrapper`` to do it. Only the final attempt writes FAILED:
    marking it on every retry would flash a failure the workflow is about to
    recover from.
    """
    from services.stage_dispatch import STAGE_RUNNERS, stage_policy

    runner = STAGE_RUNNERS.get(inp.stage)
    if runner is None:
        from temporalio.exceptions import ApplicationError

        # Non-retryable: an unknown stage never becomes known by retrying.
        raise ApplicationError(f"Unknown rerun stage: {inp.stage}", non_retryable=True)

    stall_timeout = stage_policy(inp.stage).stall_timeout
    try:
        result = await _with_heartbeat(
            runner(inp.document_id, inp.task_id, inp.options),
            stall_probe=_progress_probe(inp.task_id) if stall_timeout else None,
            stall_timeout=stall_timeout.total_seconds() if stall_timeout else None,
        )
    except Exception as exc:
        info = activity.info()
        is_last_attempt = (
            info.retry_policy is not None
            and info.retry_policy.maximum_attempts > 0
            and info.attempt >= info.retry_policy.maximum_attempts
        )
        if is_last_attempt:
            _finish_task(inp.task_id, status="FAILED", result=None, error=str(exc))
        raise

    _finish_task(inp.task_id, status="COMPLETED", result=result, error=None)
    return result if isinstance(result, dict) else {"detail": str(result)}


@activity.defn(name="fail_stage")
async def fail_stage_activity(inp: StageRerunInput) -> None:
    """Close the Task row when the workflow itself gives up (timeout, cancel,
    retries exhausted) — the run activity may never get a last attempt."""
    from data.database import get_db_manager
    from services.task_manager import TaskManager

    with get_db_manager().session() as db:
        TaskManager.mark_terminal(
            db,
            inp.task_id,
            status="FAILED",
            error="Stage run did not complete",
        )
