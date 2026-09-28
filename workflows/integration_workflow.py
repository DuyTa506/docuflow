"""IntegrationJobWorkflow — follow one library-system job to its end, then deliver.

The stages themselves run in their own workflows (extraction, digest,
translation), started when the job was created. This one only re-reads the
job's stages every ``poll_seconds``, mirrors each change into the library's
MSSQL tables, and when every stage is final writes the full result and sends
the signed callback, side by side. A delivery that keeps failing is given up
on without failing the job: the library can always pull the result over the API.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from workflows.activities.integration_activities import (
        DeliverInput,
        IntegrationJobInput,
        deliver_callback_activity,
        deliver_mssql_activity,
        refresh_integration_job_activity,
    )

REFRESH_RETRY = RetryPolicy(initial_interval=timedelta(seconds=5), maximum_interval=timedelta(minutes=2))
DELIVERY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=10),
    maximum_attempts=8,
)
# A status row is superseded by the next one, so it is not worth stalling the poll loop for.
STATUS_RETRY = RetryPolicy(initial_interval=timedelta(seconds=5), maximum_attempts=3)


@workflow.defn(name="IntegrationJobWorkflow")
class IntegrationJobWorkflow:
    @workflow.run
    async def run(self, inp: IntegrationJobInput) -> dict:
        last = inp.last_state
        while True:
            state = await workflow.execute_activity(
                refresh_integration_job_activity,
                inp.job_id,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=REFRESH_RETRY,
            )
            if state["final"]:
                break
            # Compared with what this workflow delivered, not with the row: the
            # API's GET /jobs/{id} refreshes the same row and may see a change first.
            current = {"status": state["status"], "stages": state["stages"]}
            if current != last:
                await self._deliver(
                    deliver_mssql_activity, DeliverInput(inp.job_id, final=False), STATUS_RETRY
                )
                last = current
            await workflow.sleep(timedelta(seconds=inp.poll_seconds))
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(
                    IntegrationJobInput(inp.job_id, inp.poll_seconds, last_state=last)
                )

        # Independent channels: a library whose MSSQL is down still gets its callback now.
        await asyncio.gather(
            self._deliver(deliver_mssql_activity, DeliverInput(inp.job_id, final=True)),
            self._deliver(deliver_callback_activity, inp.job_id),
        )
        return state

    async def _deliver(self, activity_fn, arg, retry: RetryPolicy = DELIVERY_RETRY) -> None:
        try:
            await workflow.execute_activity(
                activity_fn,
                arg,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry,
            )
        except ActivityError as exc:
            workflow.logger.warning("Delivery %s gave up: %s", activity_fn.__name__, exc)
