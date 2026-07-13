"""Temporal client helpers for starting digest workflows."""

import logging
from typing import Optional

from temporalio.client import Client
from temporalio.common import Priority
from temporalio.service import RPCError, RPCStatusCode

from config.settings import settings
from data.database import get_db_manager
from services.pipeline.mirror import init_pipeline_run
from services.pipeline.stage_runners import create_parent_task
from workflows.digest_workflow import DigestPipelineInput, DigestPipelineWorkflow

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


async def get_temporal_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(
            settings.temporal_host,
            namespace=settings.temporal_namespace,
        )
    return _client


def _fairness(fairness_key) -> Priority:
    """Per-user fair dispatch: one user queueing 50 documents must not starve
    another user's single click. Requires a Temporal server with fairness
    support; older servers silently ignore the field (proto3 semantics)."""
    return Priority(fairness_key=str(fairness_key)) if fairness_key else Priority()


def workflow_id_for_document(document_id: str) -> str:
    return f"digest-{document_id}"


async def terminate_running_digest(document_id: str) -> None:
    client = await get_temporal_client()
    wf_id = workflow_id_for_document(document_id)
    handle = client.get_workflow_handle(wf_id)
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.terminate("Replaced by new digest pipeline run")
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise


async def start_digest_workflow(document_id: str, fairness_key: str | None = None) -> tuple[str, str]:
    """
    Start DigestPipelineWorkflow. Returns (workflow_id, parent_task_id).
    """
    from data.db_models import Document

    db_manager = get_db_manager()
    with db_manager.session() as db:
        parent_task_id = create_parent_task(db, document_id)
        doc = db.query(Document).filter(Document.id == document_id).first()
        prior_state = doc.pipeline_state if doc else None

    wf_id = workflow_id_for_document(document_id)
    # Only pay for the describe()+terminate() Temporal RPC round-trip when a
    # prior run is actually known-running — this cheap DB read avoids two
    # unconditional Temporal RPCs on every trigger, which was slow enough to
    # let the FE's polling hit a transient error before start_workflow below
    # even returned, surfacing a false "task failed" toast.
    if prior_state == "RUNNING":
        await terminate_running_digest(document_id)

    init_pipeline_run(document_id, wf_id, parent_task_id)

    client = await get_temporal_client()
    await client.start_workflow(
        DigestPipelineWorkflow.run,
        DigestPipelineInput(
            document_id=document_id,
            parent_task_id=parent_task_id,
            workflow_id=wf_id,
        ),
        id=wf_id,
        task_queue=settings.temporal_task_queue,
        priority=_fairness(fairness_key),
    )
    logger.info("Started digest workflow %s for %s", wf_id, document_id)
    return wf_id, parent_task_id


def translation_workflow_id(document_id: str, target_language: str) -> str:
    return f"translation-{document_id}-{target_language}"


async def terminate_running_translation(document_id: str, target_language: str) -> None:
    client = await get_temporal_client()
    handle = client.get_workflow_handle(translation_workflow_id(document_id, target_language))
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.terminate("Replaced by new translation run")
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise


async def start_translation_workflow(
    document_id: str,
    translation_id: str,
    parent_task_id: str,
    target_language: str,
    domain: str = "general",
    fairness_key: str | None = None,
) -> str:
    """Start TranslationWorkflow. Returns the workflow id."""
    from workflows.activities.translation_activities import TranslationRunInput
    from workflows.translation_workflow import TranslationWorkflow

    await terminate_running_translation(document_id, target_language)

    wf_id = translation_workflow_id(document_id, target_language)
    client = await get_temporal_client()
    await client.start_workflow(
        TranslationWorkflow.run,
        TranslationRunInput(
            document_id=document_id,
            translation_id=translation_id,
            parent_task_id=parent_task_id,
            target_language=target_language,
            domain=domain,
        ),
        id=wf_id,
        task_queue=settings.temporal_translation_task_queue,
        priority=_fairness(fairness_key),
    )
    logger.info("Started translation workflow %s", wf_id)
    return wf_id


def extraction_workflow_id(document_id: str) -> str:
    return f"extraction-{document_id}"


async def start_extraction_workflow(
    document_id: str, parent_task_id: str, fairness_key: str | None = None
) -> str:
    """Start ExtractionWorkflow. Returns the workflow id."""
    from workflows.activities.extraction_activities import ExtractionRunInput
    from workflows.extraction_workflow import ExtractionWorkflow

    wf_id = extraction_workflow_id(document_id)
    client = await get_temporal_client()
    handle = client.get_workflow_handle(wf_id)
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.terminate("Replaced by new extraction run")
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise

    await client.start_workflow(
        ExtractionWorkflow.run,
        ExtractionRunInput(document_id=document_id, parent_task_id=parent_task_id),
        id=wf_id,
        task_queue=settings.temporal_extraction_task_queue,
        priority=_fairness(fairness_key),
    )
    logger.info("Started extraction workflow %s", wf_id)
    return wf_id


async def cancel_translation_workflow(document_id: str, target_language: str) -> bool:
    """Request cancellation of a running translation. Returns True if one
    was found and cancelled."""
    client = await get_temporal_client()
    handle = client.get_workflow_handle(translation_workflow_id(document_id, target_language))
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.cancel()
            return True
        return False
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return False
        raise
