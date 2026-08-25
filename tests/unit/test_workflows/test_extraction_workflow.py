"""ExtractionWorkflow orchestration: run → finalize on success; failure path
invokes fail_extraction with args=[...] and fails the workflow. Retries of
the run activity pass resume=True (attempt > 1) so extraction skips pages
already persisted by the crashed attempt.
"""

from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.activities.extraction_activities import ExtractionRunInput
from workflows.extraction_workflow import ExtractionWorkflow

INP = dict(document_id="DOC_E", parent_task_id="EXTRACT_042")


@activity.defn(name="run_extraction")
async def fake_run_ok(inp: ExtractionRunInput) -> dict:
    return {"pages_processed": 700, "element_count": 9000}


@activity.defn(name="run_extraction")
async def fake_run_flaky(inp: ExtractionRunInput) -> dict:
    fake_run_flaky.attempts.append(activity.info().attempt)
    if activity.info().attempt == 1:
        raise RuntimeError("GPU hiccup at page 650")
    return {"pages_processed": 700, "element_count": 9000}


fake_run_flaky.attempts = []


@activity.defn(name="finalize_extraction")
async def fake_finalize(inp: ExtractionRunInput, meta: dict = None) -> dict:
    return dict(meta or {}, finalized=True)


@activity.defn(name="fail_extraction")
async def fake_fail(inp: ExtractionRunInput, error: str) -> None:
    fake_fail.calls.append((inp, error))


fake_fail.calls = []


async def _run(activities):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-extraction-queue",
            workflows=[ExtractionWorkflow],
            activities=activities,
        ):
            return await env.client.execute_workflow(
                ExtractionWorkflow.run,
                ExtractionRunInput(**INP),
                id="extraction-DOC_E",
                task_queue="test-extraction-queue",
                execution_timeout=timedelta(minutes=15),
            )


@pytest.mark.asyncio
async def test_happy_path_runs_then_finalizes():
    fake_fail.calls.clear()
    report = await _run([fake_run_ok, fake_finalize, fake_fail])
    assert report["finalized"] is True
    assert report["pages_processed"] == 700
    assert not fake_fail.calls


@pytest.mark.asyncio
async def test_transient_failure_retries_and_completes():
    fake_run_flaky.attempts.clear()
    fake_fail.calls.clear()
    report = await _run([fake_run_flaky, fake_finalize, fake_fail])
    assert report["finalized"] is True
    # first attempt failed, second (the resume attempt) succeeded
    assert fake_run_flaky.attempts == [1, 2]
    assert not fake_fail.calls


@pytest.mark.asyncio
async def test_exhausted_failure_marks_failed():
    fake_fail.calls.clear()

    @activity.defn(name="run_extraction")
    async def always_fails(inp: ExtractionRunInput) -> dict:
        raise RuntimeError("vLLM down hard")

    with pytest.raises(WorkflowFailureError):
        await _run([always_fails, fake_finalize, fake_fail])

    assert len(fake_fail.calls) == 1
    assert "vLLM down hard" in fake_fail.calls[0][1]


@pytest.mark.asyncio
async def test_degenerate_ocr_error_is_not_retried():
    fake_fail.calls.clear()
    attempts = []

    @activity.defn(name="run_extraction")
    async def degenerate(inp: ExtractionRunInput) -> dict:
        from temporalio.exceptions import ApplicationError

        attempts.append(activity.info().attempt)
        raise ApplicationError(
            "Degenerate OCR output detected (repetition loop)",
            type="DegenerateOcrError",
        )

    with pytest.raises(WorkflowFailureError):
        await _run([degenerate, fake_finalize, fake_fail])

    assert attempts == [1]
    assert len(fake_fail.calls) == 1
    assert "Degenerate" in fake_fail.calls[0][1]
