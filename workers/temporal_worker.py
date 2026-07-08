"""Temporal worker — run separately from uvicorn API."""
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporalio.worker import Worker

from config.settings import settings
from services.pipeline.temporal_client import get_temporal_client
from workflows.activities.digest_activities import (
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
from workflows.digest_workflow import DigestPipelineWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Every activity the workflow can schedule. Kept as an importable list so a unit
# test can assert it stays in sync with what workflows/digest_workflow.py
# actually calls (a missing entry here means the workflow hangs forever waiting
# on an activity no worker will ever execute).
PIPELINE_ACTIVITIES = [
    ensure_extracted_activity,
    build_tree_activity,
    bibliographic_activity,
    keywords_activity,
    research_directions_activity,
    usage_scope_activity,
    summarize_activity,
    main_content_activity,
    finalize_digest_activity,
    fail_pipeline_activity,
]


async def main() -> None:
    from data.database import init_database

    init_database()
    client = await get_temporal_client()
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[DigestPipelineWorkflow],
        activities=PIPELINE_ACTIVITIES,
        max_concurrent_activities=settings.max_concurrent_pipelines,
    )
    logger.info(
        "Temporal worker listening on queue=%s (max_concurrent=%s)",
        settings.temporal_task_queue,
        settings.max_concurrent_pipelines,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
