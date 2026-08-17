"""Temporal worker — run separately from uvicorn API."""

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import timedelta

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
from workflows.activities.stage_rerun_activities import (
    fail_stage_activity,
    run_stage_activity,
)
from workflows.activities.translation_activities import (
    ensure_extracted_translation_activity,
    export_translation_activity,
    fail_translation_activity,
    run_translation_activity,
)
from workflows.digest_workflow import DigestPipelineWorkflow
from workflows.extraction_workflow import ExtractionWorkflow
from workflows.stage_rerun_workflow import StageRerunWorkflow
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

STAGE_ACTIVITIES = [
    run_stage_activity,
    fail_stage_activity,
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


def _stage_worker_config() -> dict:
    # Own queue: a multi-hour summary rerun must not sit in front of a queued
    # digest (and vice versa) — same rationale as the translation queue.
    return {
        "task_queue": settings.temporal_stage_task_queue,
        "workflows": [StageRerunWorkflow],
        "activities": STAGE_ACTIVITIES,
        "max_concurrent_activities": settings.temporal_max_concurrent_activities,
    }


# Extraction can run in its own process because it is the only one that loads
# Docling (layout, TableFormer, CodeFormula) onto the GPU. Sharing a process means
# the digest worker holds those three models too — which starved vLLM OCR of the
# memory it needed to start. `all` keeps the old behaviour so upgrading breaks no
# existing deployment.
ROLES = ("all", "pipeline", "extraction")

_ROLE_QUEUES = {
    "all": ("pipeline", "translation", "extraction", "stage"),
    "pipeline": ("pipeline", "translation", "stage"),
    "extraction": ("extraction",),
}


def worker_configs(role: str | None = None) -> list[dict]:
    """Worker configuration for *role*. `None` means "all"."""
    role = role or "all"
    if role not in _ROLE_QUEUES:
        raise ValueError(f"invalid role: {role!r} — choose one of {', '.join(ROLES)}")

    builders = {
        "pipeline": _worker_config,
        "translation": _translation_worker_config,
        "extraction": _extraction_worker_config,
        "stage": _stage_worker_config,
    }
    return [builders[name]() for name in _ROLE_QUEUES[role]]


async def main(role: str | None = None) -> None:
    from data.database import init_database

    init_database()

    # The worker is what actually shells out to soffice/pandoc during
    # extraction and export, so a missing binary matters most here.
    from utils.native_deps import log_native_dependency_warnings

    log_native_dependency_warnings()

    client = await get_temporal_client()

    # Rows left open by a worker that died mid-activity are only detectable by
    # asking Temporal — the API's startup sweep deliberately skips them.
    try:
        from services.pipeline.reconcile import reconcile_all_open_tasks

        reconciled = await reconcile_all_open_tasks()
        if reconciled:
            logger.info("Reconciled %d stale task row(s) at startup", reconciled)
    except Exception as exc:
        logger.warning("Startup reconcile skipped: %s", exc)

    configs = worker_configs(role)
    grace = timedelta(seconds=max(30, settings.worker_graceful_shutdown_seconds))
    workers = [Worker(client, graceful_shutdown_timeout=grace, **cfg) for cfg in configs]
    logger.info(
        "Temporal workers (role=%s) listening on queues=%s drain=%ss",
        role or "all",
        ",".join(cfg["task_queue"] for cfg in configs),
        int(grace.total_seconds()),
    )
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=ROLES,
        default="all",
        help="Which queues this process serves. 'extraction' is split out so the "
        "digest side never loads Docling onto the GPU. Default 'all' is the old "
        "behaviour.",
    )
    asyncio.run(main(parser.parse_args().role))
