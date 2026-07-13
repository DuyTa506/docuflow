"""When any pipeline stage raises, DigestPipelineWorkflow's except-block must
call fail_pipeline_activity successfully so the workflow terminates as FAILED
instead of looping forever on a bad execute_activity() call (see
workflows/digest_workflow.py). A broken call here masks the real error and
leaves documents stuck in RUNNING indefinitely.
"""

from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.activities.digest_activities import PipelineStageInput
from workflows.digest_workflow import DigestPipelineInput, DigestPipelineWorkflow


@activity.defn(name="ensure_extracted")
async def fake_ensure_extracted(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="build_tree")
async def fake_build_tree(inp: PipelineStageInput) -> dict:
    raise ValueError("boom: tree build failed")


@activity.defn(name="fail_pipeline")
async def fake_fail_pipeline(inp: PipelineStageInput, error: str) -> None:
    fake_fail_pipeline.calls.append((inp, error))


fake_fail_pipeline.calls = []


@pytest.mark.asyncio
async def test_workflow_failure_path_calls_fail_pipeline_activity_and_fails_cleanly():
    fake_fail_pipeline.calls.clear()

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-digest-queue",
            workflows=[DigestPipelineWorkflow],
            activities=[
                fake_ensure_extracted,
                fake_build_tree,
                fake_fail_pipeline,
            ],
        ):
            with pytest.raises(WorkflowFailureError) as excinfo:
                await env.client.execute_workflow(
                    DigestPipelineWorkflow.run,
                    DigestPipelineInput(document_id="DOC_TEST", parent_task_id="TASK_TEST"),
                    id="digest-DOC_TEST",
                    task_queue="test-digest-queue",
                    execution_timeout=timedelta(seconds=30),
                )

    # fail_pipeline_activity must have actually run with the original error
    # (proves the except-block's execute_activity() call didn't itself blow
    # up with a TypeError, which is what left real documents stuck RUNNING).
    assert len(fake_fail_pipeline.calls) == 1
    inp, error = fake_fail_pipeline.calls[0]
    assert inp.document_id == "DOC_TEST"
    assert error

    # The workflow must fail promptly, not time out from an endlessly
    # retried workflow task.
    assert "timed out" not in str(excinfo.value.cause).lower()


# ── Partial success: a non-critical stage failure must not sink the run ──

_OK_STAGES: dict = {}


@activity.defn(name="build_tree")
async def fake_build_tree_ok(inp: PipelineStageInput) -> dict:
    return {
        "completed_stages": dict(inp.completed_stages),
        "tree_result": {},
        "tree_fallback": True,
    }


@activity.defn(name="bibliographic")
async def fake_biblio(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="keywords")
async def fake_keywords_fails(inp: PipelineStageInput) -> dict:
    raise ValueError("keywords LLM exploded")


@activity.defn(name="research_directions")
async def fake_research(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="usage_scope")
async def fake_usage(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="summarize")
async def fake_summarize(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="main_content")
async def fake_main_content(inp: PipelineStageInput) -> dict:
    return dict(inp.completed_stages)


@activity.defn(name="finalize_digest")
async def fake_finalize(
    inp: PipelineStageInput, stage_failures: dict = None, tree_fallback: bool = False
) -> dict:
    fake_finalize.calls.append((inp, stage_failures, tree_fallback))
    return {"ok": True, "stage_failures": stage_failures or {}}


fake_finalize.calls = []


@pytest.mark.asyncio
async def test_noncritical_stage_failure_still_finalizes():
    fake_finalize.calls.clear()
    fake_fail_pipeline.calls.clear()

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-digest-queue",
            workflows=[DigestPipelineWorkflow],
            activities=[
                fake_ensure_extracted,
                fake_build_tree_ok,
                fake_biblio,
                fake_keywords_fails,
                fake_research,
                fake_usage,
                fake_summarize,
                fake_main_content,
                fake_finalize,
                fake_fail_pipeline,
            ],
        ):
            report = await env.client.execute_workflow(
                DigestPipelineWorkflow.run,
                DigestPipelineInput(document_id="DOC_PS", parent_task_id="TASK_PS"),
                id="digest-DOC_PS",
                task_queue="test-digest-queue",
                execution_timeout=timedelta(seconds=60),
            )

    # Workflow completed instead of failing, finalize saw the failed stage
    # and the tree fallback flag, and the failure path never ran.
    assert len(fake_finalize.calls) == 1
    _, stage_failures, tree_fallback = fake_finalize.calls[0]
    assert "KEYWORDS" in stage_failures
    assert "keywords LLM exploded" in stage_failures["KEYWORDS"]
    assert tree_fallback is True
    assert not fake_fail_pipeline.calls
    assert report["stage_failures"] == stage_failures
