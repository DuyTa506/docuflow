"""Digest pipeline Temporal workflow."""
import asyncio
from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows.activities.digest_activities import (
        PipelineStageInput,
        bibliographic_activity,
        build_tree_activity,
        ensure_extracted_activity,
        finalize_digest_activity,
        fail_pipeline_activity,
        keywords_activity,
        main_content_activity,
        research_directions_activity,
        summarize_activity,
        usage_scope_activity,
    )
    from services.pipeline.constants import STAGE_WEIGHTS


@dataclass
class DigestPipelineInput:
    document_id: str
    parent_task_id: str
    workflow_id: str = ""


@workflow.defn(name="DigestPipelineWorkflow")
class DigestPipelineWorkflow:
    @workflow.run
    async def run(self, inp: DigestPipelineInput) -> dict:
        completed: dict[str, int] = {k: 0 for k in STAGE_WEIGHTS}
        stage_inp = lambda: PipelineStageInput(
            document_id=inp.document_id,
            parent_task_id=inp.parent_task_id,
            completed_stages=dict(completed),
        )

        short_retry = RetryPolicy(maximum_attempts=2)
        tree_retry = RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=30))

        try:
            completed = await workflow.execute_activity(
                ensure_extracted_activity,
                stage_inp(),
                start_to_close_timeout=timedelta(minutes=5),
            )

            tree_out = await workflow.execute_activity(
                build_tree_activity,
                PipelineStageInput(
                    document_id=inp.document_id,
                    parent_task_id=inp.parent_task_id,
                    completed_stages=completed,
                ),
                start_to_close_timeout=timedelta(hours=2),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=tree_retry,
            )
            completed = tree_out.get("completed_stages", completed)

            group_a = await asyncio.gather(
                workflow.execute_activity(
                    bibliographic_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=short_retry,
                ),
                workflow.execute_activity(
                    keywords_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=short_retry,
                ),
                workflow.execute_activity(
                    research_directions_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=short_retry,
                ),
                workflow.execute_activity(
                    usage_scope_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=short_retry,
                ),
            )
            for stages in group_a:
                completed.update({k: max(completed.get(k, 0), v) for k, v in stages.items()})

            group_b = await asyncio.gather(
                workflow.execute_activity(
                    summarize_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=5),
                    retry_policy=short_retry,
                ),
                workflow.execute_activity(
                    main_content_activity,
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    start_to_close_timeout=timedelta(hours=4),
                    heartbeat_timeout=timedelta(minutes=5),
                    retry_policy=short_retry,
                ),
            )
            for stages in group_b:
                completed.update({k: max(completed.get(k, 0), v) for k, v in stages.items()})

            report = await workflow.execute_activity(
                finalize_digest_activity,
                PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                start_to_close_timeout=timedelta(minutes=10),
            )
            return report

        except Exception as exc:
            await workflow.execute_activity(
                fail_pipeline_activity,
                PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                str(exc),
                start_to_close_timeout=timedelta(minutes=2),
            )
            raise
