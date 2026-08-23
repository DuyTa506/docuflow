"""Shared activity input and helpers."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from temporalio import activity

from config.settings import settings
from data.database import get_db_manager
from data.db_models import TreeIndex
from services.pipeline.constants import STAGE_WEIGHTS
from services.pipeline.mirror import (
    make_stage_progress_sink,
    mark_stage_complete,
    update_pipeline_mirror,
)
from services.pipeline.stage_runners import ensure_extracted
from services.progress_reporting import progress_context


@dataclass
class PipelineStageInput:
    document_id: str
    parent_task_id: str
    completed_stages: dict[str, int] = field(default_factory=dict)


def _stages_copy(completed: dict[str, int]) -> dict[str, int]:
    base = {k: 0 for k in STAGE_WEIGHTS}
    base.update(completed or {})
    return base


def _activity_attempt() -> int:
    """Temporal attempt, with a deterministic value for direct unit calls."""

    try:
        return activity.info().attempt
    except RuntimeError:
        return 1


from workflows.activities._common import _with_heartbeat  # noqa: E402


_DIGEST_STAGE_LABELS = {
    "BUILD_TREE": "xây dựng cây mục lục",
    "BIBLIOGRAPHIC": "trích xuất thư mục",
    "KEYWORDS": "trích xuất từ khóa",
    "RESEARCH_DIRECTIONS": "phân tích hướng nghiên cứu",
    "USAGE_SCOPE": "xác định phạm vi ứng dụng",
    "HIERARCHICAL_SUMMARIZE": "tóm tắt phân cấp",
    "MAIN_CONTENT": "trích nội dung chính",
    "FINALIZE": "hoàn tất",
}


def _digest_stage_message(stage: str) -> str:
    return f"Đang {_DIGEST_STAGE_LABELS.get(stage, stage.lower())}"


@activity.defn(name="ensure_extracted")
async def ensure_extracted_activity(inp: PipelineStageInput) -> dict[str, int]:
    try:
        await ensure_extracted(inp.document_id)
    except ValueError:
        # Still waiting on OCR — the UI polls pipeline-status, and showing
        # "Pipeline started"/RUNNING while OCR is PENDING reads as a stuck
        # pipeline. Tell the truth on every retry of the wait loop.
        update_pipeline_mirror(
            inp.document_id,
            state="RUNNING",
            stage="BUILD_TREE",
            message="Chờ trích xuất (OCR) hoàn thành trước khi tổng thuật…",
            parent_task_id=inp.parent_task_id,
            structured_progress={
                "version": 1,
                "pipeline": "digest",
                "phase": "waiting_upstream",
                "mode": "digest_pipeline",
                "stage": "BUILD_TREE",
                "attempt": _activity_attempt(),
            },
        )
        raise
    stages = _stages_copy(inp.completed_stages)
    update_pipeline_mirror(
        inp.document_id,
        state="RUNNING",
        stage="BUILD_TREE",
        message="Đã trích xuất xong, đang chuẩn bị cây mục lục",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "active",
            "mode": "digest_pipeline",
            "stage": "BUILD_TREE",
            "attempt": _activity_attempt(),
        },
    )
    return stages


def _tree_index_fresh(document_id: str) -> bool:
    from utils.tree_quality import TREE_SCHEMA_VERSION

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
        config = row.config or {}
        if config.get("tree_schema_version", 0) < TREE_SCHEMA_VERSION:
            return False
        quality = config.get("tree_quality") or {}
        if quality.get("ok") is False:
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
        message="Đang xây dựng cây mục lục",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "active",
            "mode": "digest_pipeline",
            "stage": "BUILD_TREE",
            "attempt": _activity_attempt(),
        },
    )

    tree_fallback = False
    if _tree_index_fresh(inp.document_id):
        activity.logger.info("Skipping BUILD_TREE — fresh TreeIndex exists")
        result = {"skipped": True, "reason": "fresh_tree_index"}
    else:
        from services.pipeline.stage_runners import run_build_tree

        try:
            with progress_context(
                sink=make_stage_progress_sink(
                    inp.document_id,
                    inp.parent_task_id,
                    "BUILD_TREE",
                    stages,
                ),
                defaults={
                    "pipeline": "digest",
                    "phase": "active",
                    "mode": "digest_pipeline",
                    "stage": "BUILD_TREE",
                    "attempt": _activity_attempt(),
                },
            ):
                result = await _with_heartbeat(run_build_tree(inp.document_id))
                if result.get("tree_fallback") or result.get("skipped_persist"):
                    tree_fallback = True
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
        message=_digest_stage_message(stage),
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "active",
            "mode": "digest_pipeline",
            "stage": stage,
            "attempt": _activity_attempt(),
        },
    )
    with progress_context(
        sink=make_stage_progress_sink(
            inp.document_id,
            inp.parent_task_id,
            stage,
            stages,
        ),
        defaults={
            "pipeline": "digest",
            "phase": "active",
            "mode": "digest_pipeline",
            "stage": stage,
            "attempt": _activity_attempt(),
        },
    ):
        from services.stage_dispatch import STALL_TIMEOUTS
        from workflows.activities.stage_rerun_activities import _progress_probe

        stall = STALL_TIMEOUTS.get(stage)
        await _with_heartbeat(
            runner(inp.document_id),
            stall_probe=_progress_probe(inp.parent_task_id) if stall else None,
            stall_timeout=stall.total_seconds() if stall else None,
        )
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
async def finalize_digest_activity(
    inp: PipelineStageInput,
    stage_failures: Optional[dict[str, str]] = None,
    tree_fallback: bool = False,
) -> dict[str, Any]:
    from services.pipeline.quality import build_quality_report

    stages = _stages_copy(inp.completed_stages)
    report = build_quality_report(
        inp.document_id,
        stage_failures=stage_failures,
        tree_fallback=tree_fallback,
    )
    update_pipeline_mirror(
        inp.document_id,
        state="RUNNING",
        stage="FINALIZE",
        stage_progress=99,
        message="Đang chuẩn bị file tải xuống…",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        quality_report=report,
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "exporting",
            "mode": "digest_pipeline",
            "stage": "FINALIZE",
            "unit_kind": "export",
            "units_done": 0,
            "units_total": 1,
            "attempt": _activity_attempt(),
        },
    )
    await _cache_digest_export(inp.document_id)
    # Single mirror call commits doc DONE + task COMPLETED/result atomically —
    # a crash here must never leave the doc DONE with the task still RUNNING.
    update_pipeline_mirror(
        inp.document_id,
        state="DONE",
        stage="FINALIZE",
        stage_progress=100,
        message="Tổng thuật hoàn tất",
        parent_task_id=inp.parent_task_id,
        completed_stages=stages,
        quality_report=report,
        task_result=report,
        structured_progress={
            "version": 1,
            "pipeline": "digest",
            "phase": "exporting",
            "mode": "digest_pipeline",
            "stage": "FINALIZE",
            "unit_kind": "export",
            "units_done": 1,
            "units_total": 1,
            "attempt": _activity_attempt(),
        },
    )
    return report


async def _cache_digest_export(document_id: str) -> None:
    from services.export_service import export_service
    from workflows.activities._common import _with_heartbeat

    # Book-length digest DOCX/PDF can exceed BOOKKEEPING; keep the activity
    # alive the same way translation export does.
    await _with_heartbeat(export_service.cache_digest_export(document_id))


@activity.defn(name="fail_pipeline")
async def fail_pipeline_activity(inp: PipelineStageInput, error: str) -> None:
    update_pipeline_mirror(
        inp.document_id,
        state="FAILED",
        message=error[:500],
        parent_task_id=inp.parent_task_id,
    )
