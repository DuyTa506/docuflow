"""TranslationWorkflow orchestration: run → export on success; a run that
exhausts retries must invoke fail_translation with args=[...] (multi-arg
activities called positionally was the bug class that left digest documents
stuck RUNNING forever) and fail the workflow promptly.
"""

from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.activities.translation_activities import TranslationRunInput
from workflows.translation_workflow import TranslationWorkflow

INP = dict(
    document_id="DOC_T",
    translation_id="TRN_T",
    parent_task_id="TRANSLATE_042",
    target_language="vi",
    domain="general",
)


@activity.defn(name="run_translation")
async def fake_run_ok(inp: TranslationRunInput) -> dict:
    return {"translation_mode": "block_based", "translation_length": 123}


@activity.defn(name="run_translation")
async def fake_run_fails(inp: TranslationRunInput) -> dict:
    raise ValueError("boom: LLM died")


@activity.defn(name="export_translation")
async def fake_export(inp: TranslationRunInput, meta: dict = None) -> dict:
    fake_export.calls.append((inp, meta))
    return dict(meta or {}, exported=True)


fake_export.calls = []


@activity.defn(name="fail_translation")
async def fake_fail(inp: TranslationRunInput, error: str) -> None:
    fake_fail.calls.append((inp, error))


fake_fail.calls = []


async def _run(activities):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-translation-queue",
            workflows=[TranslationWorkflow],
            activities=activities,
        ):
            return await env.client.execute_workflow(
                TranslationWorkflow.run,
                TranslationRunInput(**INP),
                id="translation-DOC_T-vi",
                task_queue="test-translation-queue",
                execution_timeout=timedelta(seconds=60),
            )


@pytest.mark.asyncio
async def test_happy_path_runs_then_exports():
    fake_export.calls.clear()
    fake_fail.calls.clear()

    report = await _run([fake_run_ok, fake_export, fake_fail])

    assert report["exported"] is True
    assert report["translation_mode"] == "block_based"
    assert len(fake_export.calls) == 1
    _, meta = fake_export.calls[0]
    assert meta["translation_length"] == 123
    assert not fake_fail.calls


@pytest.mark.asyncio
async def test_run_failure_marks_failed_and_raises():
    fake_export.calls.clear()
    fake_fail.calls.clear()

    with pytest.raises(WorkflowFailureError):
        await _run([fake_run_fails, fake_export, fake_fail])

    assert len(fake_fail.calls) == 1
    inp, error = fake_fail.calls[0]
    assert inp.translation_id == "TRN_T"
    assert "boom: LLM died" in error
    assert not fake_export.calls
