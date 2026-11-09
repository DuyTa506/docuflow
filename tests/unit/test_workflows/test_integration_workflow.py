"""IntegrationJobWorkflow: poll until final, mirror changes, deliver once.

The workflow decides what to deliver from what *it* last delivered, not from
whether the refresh changed the row: the API's GET /jobs/{id} refreshes the
same row, so a change can already be consumed when the workflow looks. A
delivery channel that keeps failing must neither fail the workflow nor hold up
the other channel — the result is still available over the pull API.
"""

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.activities.integration_activities import DeliverInput, IntegrationJobInput
from workflows.integration_workflow import DELIVERY_RETRY, STATUS_RETRY, IntegrationJobWorkflow

RUNNING = {"final": False, "status": "PROCESSING", "stages": {"ocr": "RUNNING"}}
DONE = {"final": True, "status": "COMPLETED", "stages": {"ocr": "COMPLETED"}}


def _activities(calls, states, *, mssql_fails=False, callback_fails=False):
    states = iter(states)

    @activity.defn(name="refresh_integration_job")
    async def refresh(job_id: str) -> dict:
        calls.append(("refresh", job_id))
        return next(states)

    @activity.defn(name="deliver_integration_mssql")
    async def mssql(inp: DeliverInput) -> str:
        calls.append(("mssql", inp.final))
        if mssql_fails:
            raise RuntimeError("Login failed")
        return "DELIVERED"

    @activity.defn(name="deliver_integration_callback")
    async def callback(job_id: str) -> str:
        calls.append(("callback", job_id))
        if callback_fails:
            raise RuntimeError("connection refused")
        return "DELIVERED"

    return [refresh, mssql, callback]


async def _run(activities, inp=None):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-integration",
            workflows=[IntegrationJobWorkflow],
            activities=activities,
        ):
            return await env.client.execute_workflow(
                IntegrationJobWorkflow.run,
                inp or IntegrationJobInput(job_id="JOB_001", poll_seconds=30),
                id=f"integration-test-{uuid.uuid4().hex}",
                task_queue="test-integration",
            )


@pytest.mark.asyncio
async def test_mirrors_each_new_state_then_delivers_once():
    calls = []
    result = await _run(_activities(calls, [RUNNING, RUNNING, DONE]))

    assert result["status"] == "COMPLETED"
    assert calls.count(("refresh", "JOB_001")) == 3
    assert [c for c in calls if c[0] == "mssql"] == [("mssql", False), ("mssql", True)]
    assert calls.count(("callback", "JOB_001")) == 1


@pytest.mark.asyncio
async def test_final_state_is_delivered_even_if_already_seen():
    """After continue-as-new, or when the API refreshed the row first."""
    calls = []
    inp = IntegrationJobInput(job_id="JOB_001", poll_seconds=30,
                              last_state={"status": "COMPLETED", "stages": {"ocr": "COMPLETED"}})
    await _run(_activities(calls, [DONE]), inp)

    assert ("mssql", True) in calls
    assert calls.count(("callback", "JOB_001")) == 1


@pytest.mark.asyncio
async def test_failing_callback_does_not_fail_the_job():
    calls = []
    result = await _run(_activities(calls, [DONE], callback_fails=True))

    assert result["status"] == "COMPLETED"
    assert calls.count(("callback", "JOB_001")) == DELIVERY_RETRY.maximum_attempts
    assert calls.count(("mssql", True)) == 1


@pytest.mark.asyncio
async def test_status_writes_give_up_quickly():
    """MSSQL down while the job runs: the poll loop moves on after a few tries."""
    calls = []
    await _run(_activities(calls, [RUNNING, DONE], mssql_fails=True))

    assert calls.count(("mssql", False)) == STATUS_RETRY.maximum_attempts
    assert calls.count(("callback", "JOB_001")) == 1


@pytest.mark.asyncio
async def test_failing_mssql_does_not_hold_up_the_callback():
    calls = []
    await _run(_activities(calls, [DONE], mssql_fails=True))

    # The callback went out on the first try, long before the MSSQL retries ran out.
    first_callback = calls.index(("callback", "JOB_001"))
    assert first_callback < calls.index(("mssql", True)) + 3
    assert calls.count(("mssql", True)) == DELIVERY_RETRY.maximum_attempts
    assert calls.count(("callback", "JOB_001")) == 1
