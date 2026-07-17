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
    fail_pipeline_activity,
    finalize_digest_activity,
    keywords_activity,
    main_content_activity,
    research_directions_activity,
    summarize_activity,
    usage_scope_activity,
)
from workflows.activities.extraction_activities import (
    fail_extraction_activity,
    finalize_extraction_activity,
    run_extraction_activity,
)
from workflows.activities.translation_activities import (
    ensure_extracted_translation_activity,
    export_translation_activity,
    fail_translation_activity,
    run_translation_activity,
)
from workflows.digest_workflow import DigestPipelineWorkflow
from workflows.extraction_workflow import ExtractionWorkflow
from workflows.translation_workflow import TranslationWorkflow

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


# Same sync-with-workflow contract as PIPELINE_ACTIVITIES.
TRANSLATION_ACTIVITIES = [
    ensure_extracted_translation_activity,
    run_translation_activity,
    export_translation_activity,
    fail_translation_activity,
]

EXTRACTION_ACTIVITIES = [
    run_extraction_activity,
    finalize_extraction_activity,
    fail_extraction_activity,
]


def _worker_config() -> dict:
    return {
        "task_queue": settings.temporal_task_queue,
        "workflows": [DigestPipelineWorkflow],
        "activities": PIPELINE_ACTIVITIES,
        "max_concurrent_activities": settings.temporal_max_concurrent_activities,
    }


def _translation_worker_config() -> dict:
    # Separate queue so a running digest can't starve translations (and vice
    # versa) — same process/systemd unit, independently tunable concurrency.
    return {
        "task_queue": settings.temporal_translation_task_queue,
        "workflows": [TranslationWorkflow],
        "activities": TRANSLATION_ACTIVITIES,
        "max_concurrent_activities": settings.temporal_max_concurrent_activities,
    }


def _extraction_worker_config() -> dict:
    return {
        "task_queue": settings.temporal_extraction_task_queue,
        "workflows": [ExtractionWorkflow],
        "activities": EXTRACTION_ACTIVITIES,
        "max_concurrent_activities": settings.extraction_max_concurrent,
    }


async def main() -> None:
    from data.database import init_database

    init_database()
    client = await get_temporal_client()
    configs = [_worker_config(), _translation_worker_config(), _extraction_worker_config()]
    workers = [Worker(client, **cfg) for cfg in configs]
    logger.info(
        "Temporal workers listening on queues=%s",
        ",".join(cfg["task_queue"] for cfg in configs),
    )
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    asyncio.run(main())
