"""OCR/extraction Temporal workflow.

One heartbeated extraction activity (page-level persistence makes each page
a checkpoint; retries pass resume=True and skip stored pages) followed by an
atomic finalize. GPU/vLLM outages surface as retries with backoff rather
than a silently dead in-process coroutine.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows.activities.extraction_activities import (
        ExtractionRunInput,
        fail_extraction_activity,
        finalize_extraction_activity,
        run_extraction_activity,
    )
    from workflows.timeouts import BOOKKEEPING, HEARTBEAT, LONG_RUN


def _root_error(exc: BaseException) -> str:
    cur = exc
    while True:
        nxt = getattr(cur, "cause", None) or cur.__cause__
        if nxt is None:
            return str(cur)
        cur = nxt


@workflow.defn(name="ExtractionWorkflow")
class ExtractionWorkflow:
    @workflow.run
    async def run(self, inp: ExtractionRunInput) -> dict:
        try:
            meta = await workflow.execute_activity(
                run_extraction_activity,
                inp,
                start_to_close_timeout=LONG_RUN,
                heartbeat_timeout=HEARTBEAT,
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(minutes=1),
                    backoff_coefficient=2.0,
                    # Data-dependent OCR loop: retrying the whole book
                    # re-dies on the same page. Per-page skip handles it.
                    non_retryable_error_types=["DegenerateOcrError"],
                ),
            )
            return await workflow.execute_activity(
                finalize_extraction_activity,
                args=[inp, meta],
                start_to_close_timeout=BOOKKEEPING,
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception as exc:
            await workflow.execute_activity(
                fail_extraction_activity,
                args=[inp, _root_error(exc)],
                start_to_close_timeout=BOOKKEEPING,
            )
            raise
