"""Workflow for a standalone single-stage rerun.

Deliberately thin: one activity, whose timeout/heartbeat/retry come from the
same policy table the digest stages use. The value is durability, not
orchestration — the stage itself is already one unit of work.
"""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from services.stage_dispatch import stage_policy
    from workflows.activities.stage_rerun_activities import (
        StageRerunInput,
        fail_stage_activity,
        run_stage_activity,
    )
    from workflows.timeouts import BOOKKEEPING


@workflow.defn(name="StageRerunWorkflow")
class StageRerunWorkflow:
    @workflow.run
    async def run(self, inp: StageRerunInput) -> dict:
        policy = stage_policy(inp.stage)
        activity_kwargs = {
            "start_to_close_timeout": policy.start_to_close,
            "retry_policy": policy.retry,
        }
        if policy.heartbeat is not None:
            activity_kwargs["heartbeat_timeout"] = policy.heartbeat
        try:
            return await workflow.execute_activity(
                run_stage_activity,
                inp,
                **activity_kwargs,
            )
        except BaseException:
            # Timeout / cancellation / retries exhausted: the run activity may
            # never have reached a final attempt, so the Task row would sit
            # RUNNING forever without this.
            await workflow.execute_activity(
                fail_stage_activity,
                inp,
                start_to_close_timeout=BOOKKEEPING,
            )
            raise
