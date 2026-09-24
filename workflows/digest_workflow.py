"""Digest pipeline Temporal workflow."""

import asyncio
from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from config.capacity import batched, capacity_profile
    from services.pipeline.constants import CRITICAL_STAGES, STAGE_WEIGHTS
    from workflows.activities.digest_activities import (
        PipelineStageInput,
        bibliographic_activity,
        build_tree_activity,
        ensure_extracted_activity,
        fail_pipeline_activity,
        finalize_digest_activity,
        keywords_activity,
        main_content_activity,
        research_directions_activity,
        summarize_activity,
        usage_scope_activity,
    )
    from workflows.timeouts import BOOKKEEPING, HEARTBEAT, LONG_RUN, WAIT_GATE


@dataclass
class DigestPipelineInput:
    document_id: str
    parent_task_id: str
    workflow_id: str = ""


def _root_error(exc: BaseException) -> str:
    """Unwrap ActivityError/ApplicationError chains to the original message —
    str(ActivityError) is just 'Activity task failed'."""
    cur = exc
    while True:
        nxt = getattr(cur, "cause", None) or cur.__cause__
        if nxt is None:
            return str(cur)
        cur = nxt


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
        # Book-length summarize/main_content can run many hours; keep retries
        # but do not kill healthy work with heartbeat / short start_to_close.
        long_stage_retry = RetryPolicy(
            maximum_attempts=6,
            initial_interval=timedelta(seconds=30),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=5),
        )

        try:
            completed = await workflow.execute_activity(
                ensure_extracted_activity,
                stage_inp(),
                start_to_close_timeout=WAIT_GATE,
            )

            tree_out = await workflow.execute_activity(
                build_tree_activity,
                PipelineStageInput(
                    document_id=inp.document_id,
                    parent_task_id=inp.parent_task_id,
                    completed_stages=completed,
                ),
                start_to_close_timeout=LONG_RUN,
                heartbeat_timeout=HEARTBEAT,
                retry_policy=tree_retry,
            )
            completed = tree_out.get("completed_stages", completed)
            tree_fallback = bool(tree_out.get("tree_fallback"))
            stage_failures: dict[str, str] = {}

            cap = capacity_profile()
            group_a_stages = (
                ("BIBLIOGRAPHIC", bibliographic_activity),
                ("KEYWORDS", keywords_activity),
                ("RESEARCH_DIRECTIONS", research_directions_activity),
                ("USAGE_SCOPE", usage_scope_activity),
            )
            group_a = []
            for batch in batched(group_a_stages, cap.digest_group_a_parallelism):
                batch_out = await asyncio.gather(
                    *(
                        workflow.execute_activity(
                            act,
                            PipelineStageInput(
                                inp.document_id, inp.parent_task_id, dict(completed)
                            ),
                            start_to_close_timeout=LONG_RUN,
                            heartbeat_timeout=HEARTBEAT,
                            retry_policy=short_retry,
                        )
                        for _, act in batch
                    ),
                    return_exceptions=True,
                )
                group_a.extend(batch_out)
            for (stage_name, _), stages in zip(group_a_stages, group_a):
                if isinstance(stages, BaseException):
                    if stage_name in CRITICAL_STAGES:
                        raise stages
                    workflow.logger.warning(
                        "Stage %s failed — continuing without it: %s",
                        stage_name,
                        stages,
                    )
                    stage_failures[stage_name] = _root_error(stages)
                    # Non-critical failures must not freeze the parallel-group
                    # bottleneck at 0% — treat the stage as accounted so the
                    # bar can advance with its siblings.
                    completed[stage_name] = 100
                    continue
                completed.update({k: max(completed.get(k, 0), v) for k, v in stages.items()})

            summarize_call = workflow.execute_activity(
                summarize_activity,
                PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                start_to_close_timeout=LONG_RUN,
                heartbeat_timeout=HEARTBEAT,
                retry_policy=long_stage_retry,
            )
            main_content_call = workflow.execute_activity(
                main_content_activity,
                PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                start_to_close_timeout=LONG_RUN,
                heartbeat_timeout=HEARTBEAT,
                retry_policy=long_stage_retry,
            )
            if cap.digest_group_b_parallel:
                group_b = await asyncio.gather(summarize_call, main_content_call)
            else:
                group_b = [
                    await summarize_call,
                    await main_content_call,
                ]
            for stages in group_b:
                completed.update({k: max(completed.get(k, 0), v) for k, v in stages.items()})

            report = await workflow.execute_activity(
                finalize_digest_activity,
                args=[
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    stage_failures,
                    tree_fallback,
                ],
                start_to_close_timeout=LONG_RUN,
                heartbeat_timeout=HEARTBEAT,
            )
            return report

        except Exception as exc:
            await workflow.execute_activity(
                fail_pipeline_activity,
                args=[
                    PipelineStageInput(inp.document_id, inp.parent_task_id, dict(completed)),
                    str(exc),
                ],
                start_to_close_timeout=BOOKKEEPING,
            )
            raise
