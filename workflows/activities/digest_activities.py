"""Shared activity input and helpers."""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from temporalio import activity

from config.settings import settings
from data.database import get_db_manager
from data.db_models import TreeIndex
from services.pipeline.constants import STAGE_WEIGHTS
from services.pipeline.mirror import mark_stage_complete, update_pipeline_mirror
from services.pipeline.stage_runners import ensure_extracted


@dataclass
class PipelineStageInput:
    document_id: str
    parent_task_id: str
    completed_stages: dict[str, int] = field(default_factory=dict)


def _stages_copy(completed: dict[str, int]) -> dict[str, int]:
    base = {k: 0 for k in STAGE_WEIGHTS}
    base.update(completed or {})
    return base


_HEARTBEAT_INTERVAL_SECONDS = 20


async def _heartbeat_forever() -> None:
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        activity.heartbeat()


async def _with_heartbeat(coro):
    """Run `coro` while periodically heartbeating, so activities with a
    `heartbeat_timeout` aren't killed just because the wrapped work doesn't
    report its own progress (e.g. long LLM chains on large documents)."""
    heartbeat_task = asyncio.create_task(_heartbeat_forever())
    try:
        return await coro
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


@activity.defn(name="ensure_extracted")
async def ensure_extracted_activity(inp: PipelineStageInput) -> dict[str, int]:
    await ensure_extracted(inp.document_id)
    stages = _stages_copy(inp.completed_stages)
    update_pipeline_mirror(
        inp.document_id,
        state="RUNNING",
        stage="BUILD_TREE",
        message="Document extracted — preparing tree",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
    )
    return stages


def _tree_index_fresh(document_id: str) -> bool:
    db_manager = get_db_manager()
    with db_manager.session() as db:
        row = (
            db.query(TreeIndex)
            .filter(TreeIndex.document_id == document_id)
            .order_by(TreeIndex.created_at.desc())
            .first()
        )
        if not row or not row.created_at:
            return False
        age = datetime.utcnow() - row.created_at
        return age < timedelta(hours=settings.tree_index_max_age_hours)


@activity.defn(name="build_tree")
async def build_tree_activity(inp: PipelineStageInput) -> dict[str, Any]:
    stages = _stages_copy(inp.completed_stages)
    update_pipeline_mirror(
        inp.document_id,
        stage="BUILD_TREE",
        stage_progress=5,
        message="Building tree index",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
    )

    tree_fallback = False
    if _tree_index_fresh(inp.document_id):
        activity.logger.info("Skipping BUILD_TREE — fresh TreeIndex exists")
        result = {"skipped": True, "reason": "fresh_tree_index"}
    else:
        from services.pipeline.stage_runners import run_build_tree

        try:
            result = await _with_heartbeat(run_build_tree(inp.document_id))
        except Exception as exc:
            activity.logger.warning("BUILD_TREE failed: %s", exc)
            tree_fallback = True
            result = {"error": str(exc), "tree_fallback": True}

    stages = mark_stage_complete(
        inp.document_id,
        "BUILD_TREE",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
    )
    return {"completed_stages": stages, "tree_result": result, "tree_fallback": tree_fallback}


async def _run_stage(
    inp: PipelineStageInput,
    stage: str,
    runner,
) -> dict[str, int]:
    stages = _stages_copy(inp.completed_stages)
    update_pipeline_mirror(
        inp.document_id,
        stage=stage,
        stage_progress=10,
        message=f"Running {stage}",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
    )
    await _with_heartbeat(runner(inp.document_id))
    return mark_stage_complete(
        inp.document_id,
        stage,
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
    )


@activity.defn(name="bibliographic")
async def bibliographic_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_bibliographic

    return await _run_stage(inp, "BIBLIOGRAPHIC", run_bibliographic)


@activity.defn(name="keywords")
async def keywords_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_keywords

    return await _run_stage(inp, "KEYWORDS", run_keywords)


@activity.defn(name="research_directions")
async def research_directions_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_research_directions

    return await _run_stage(inp, "RESEARCH_DIRECTIONS", run_research_directions)


@activity.defn(name="usage_scope")
async def usage_scope_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_usage_scope

    return await _run_stage(inp, "USAGE_SCOPE", run_usage_scope)


@activity.defn(name="summarize")
async def summarize_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_summarize

    return await _run_stage(inp, "HIERARCHICAL_SUMMARIZE", run_summarize)


@activity.defn(name="main_content")
async def main_content_activity(inp: PipelineStageInput) -> dict[str, int]:
    from services.pipeline.stage_runners import run_main_content

    return await _run_stage(inp, "MAIN_CONTENT", run_main_content)


@activity.defn(name="finalize_digest")
async def finalize_digest_activity(inp: PipelineStageInput) -> dict[str, Any]:
    from services.pipeline.quality import build_quality_report

    stages = _stages_copy(inp.completed_stages)
    report = build_quality_report(inp.document_id)
    update_pipeline_mirror(
        inp.document_id,
        state="RUNNING",
        stage="FINALIZE",
        stage_progress=99,
        message="Preparing download export…",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        quality_report=report,
    )
    await _cache_digest_export(inp.document_id)
    update_pipeline_mirror(
        inp.document_id,
        state="DONE",
        stage="FINALIZE",
        stage_progress=100,
        message="Pipeline completed",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        quality_report=report,
    )
    if inp.parent_task_id:
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.db_models import Task

            task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
            if task:
                task.status = "COMPLETED"
                task.progress = 100
                task.result = report
                task.message = "Digest pipeline completed"
                db.commit()
    return report


async def _cache_digest_export(document_id: str) -> None:
    from services.export_service import export_service

    await export_service.cache_digest_export(document_id)


@activity.defn(name="fail_pipeline")
async def fail_pipeline_activity(inp: PipelineStageInput, error: str) -> None:
    update_pipeline_mirror(
        inp.document_id,
        state="FAILED",
        message=error[:500],
        parent_task_id=inp.parent_task_id,
    )
    if inp.parent_task_id:
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.db_models import Task

            task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
            if task:
                task.status = "FAILED"
                task.error = error[:2000]
                db.commit()
