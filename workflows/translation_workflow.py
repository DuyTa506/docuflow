"""Translation Temporal workflow.

Durability model: the routed translation runs as one long heartbeated
activity; a worker crash or retry re-runs it, but the persistent unit cache
(services/translators/_cache.py) turns the re-run into a resume — only
units missing from MinIO are re-translated. Export + completion bookkeeping
is a separate activity so its retries never re-translate anything.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows.activities.translation_activities import (
        TranslationRunInput,
        ensure_extracted_translation_activity,
        export_translation_activity,
        fail_translation_activity,
        run_translation_activity,
    )


def _root_error(exc: BaseException) -> str:
    """Unwrap ActivityError/ApplicationError chains to the original message."""
    cur = exc
    while True:
        nxt = getattr(cur, "cause", None) or cur.__cause__
        if nxt is None:
            return str(cur)
        cur = nxt


@workflow.defn(name="TranslationWorkflow")
class TranslationWorkflow:
    @workflow.run
    async def run(self, inp: TranslationRunInput) -> dict:
        try:
            # Wait-gate: no retry_policy → Temporal's default (unlimited
            # attempts) turns the retryable "not extracted yet" error into a
            # poll loop; a FAILED extraction raises non-retryable and falls
            # straight to the error path below.
            await workflow.execute_activity(
                ensure_extracted_translation_activity,
                inp,
                start_to_close_timeout=timedelta(minutes=5),
            )
            meta = await workflow.execute_activity(
                run_translation_activity,
                inp,
                start_to_close_timeout=timedelta(hours=6),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=30),
                    backoff_coefficient=2.0,
                ),
            )
            return await workflow.execute_activity(
                export_translation_activity,
                args=[inp, meta],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception as exc:
            await workflow.execute_activity(
                fail_translation_activity,
                args=[inp, _root_error(exc)],
                start_to_close_timeout=timedelta(minutes=2),
            )
            raise
