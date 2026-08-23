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


async def start_digest_workflow(
    document_id: str,
    fairness_key: str | None = None,
    parent_task_id: str | None = None,
) -> tuple[str, str]:
    """
    Start DigestPipelineWorkflow. Returns (workflow_id, parent_task_id).

    Pass ``parent_task_id`` when the HTTP layer already inserted a
    DIGEST_PIPELINE row (direct start or overflow unqueue). Soft admission
    happens at submit / queue claim — not here.
    """
    from temporalio.exceptions import WorkflowAlreadyStartedError

    from data.db_models import Document, Task
    from services.pipeline.admission import mark_dispatched

    db_manager = get_db_manager()
    with db_manager.session() as db:
        if parent_task_id:
            task = db.query(Task).filter(Task.id == parent_task_id).first()
            if task is not None:
                mark_dispatched(task)
            db.commit()
        else:
            parent_task_id = create_parent_task(db, document_id)
        doc = db.query(Document).filter(Document.id == document_id).first()
        prior_state = doc.pipeline_state if doc else None

    wf_id = workflow_id_for_document(document_id)
    # Cheap DB shortcut avoids two Temporal RPCs on a cold start. If the
    # mirror is stale and start still hits AlreadyStarted, terminate+retry once.
    if prior_state == "RUNNING":
        await terminate_running_digest(document_id)

    init_pipeline_run(document_id, wf_id, parent_task_id)

    client = await get_temporal_client()

    async def _start() -> None:
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

    try:
        await _start()
    except WorkflowAlreadyStartedError:
        await terminate_running_digest(document_id)
        await _start()
    logger.info("Started digest workflow %s for %s", wf_id, document_id)
    return wf_id, parent_task_id


async def is_stage_running(document_id: str, stage: str) -> bool:
    """Whether a stage rerun is genuinely in flight, per Temporal.

    Liveness for Temporal-owned work lives in Temporal, not in the API's
    in-process ``_running_tasks`` dict — asking the dict (as the old dedupe
    did) reports "dead" for every healthy worker-side run.
    """
    from services.stage_dispatch import stage_workflow_id

    client = await get_temporal_client()
    handle = client.get_workflow_handle(stage_workflow_id(document_id, stage))
    try:
        desc = await handle.describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return False
        raise
    return desc.status is not None and desc.status.name in ("RUNNING", "CONTINUED_AS_NEW")


async def start_stage_workflow(
    document_id: str,
    stage: str,
    task_id: str,
    options: dict | None = None,
    fairness_key: str | None = None,
) -> str:
    """Start (or restart) a durable single-stage rerun. Returns workflow id.

    Soft admission is done at submit / overflow claim — this path only clears
    a queued flag if present and starts Temporal.
    """
    from data.db_models import Task
    from services.pipeline.admission import mark_dispatched
    from services.stage_dispatch import stage_workflow_id
    from workflows.activities.stage_rerun_activities import StageRerunInput
    from workflows.stage_rerun_workflow import StageRerunWorkflow

    wf_id = stage_workflow_id(document_id, stage)

    db_manager = get_db_manager()
    with db_manager.session() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is not None:
            mark_dispatched(task)
            db.commit()

    # Explicit rerun means "replace whatever is running", same contract as
    # start_digest_workflow — otherwise start_workflow rejects the duplicate id.
    client = await get_temporal_client()
    handle = client.get_workflow_handle(wf_id)
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.terminate("Replaced by a newer stage rerun")
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise

    await client.start_workflow(
        StageRerunWorkflow.run,
        StageRerunInput(document_id=document_id, stage=stage, task_id=task_id, options=options),
        id=wf_id,
        task_queue=settings.temporal_stage_task_queue,
        priority=_fairness(fairness_key),
    )
    logger.info("Started stage rerun %s for %s (task %s)", wf_id, document_id, task_id)
    return wf_id


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


async def terminate_running_extraction(document_id: str) -> None:
    client = await get_temporal_client()
    handle = client.get_workflow_handle(extraction_workflow_id(document_id))
    try:
        desc = await handle.describe()
        if desc.status.name in ("RUNNING", "CONTINUED_AS_NEW"):
            await handle.terminate("Document deleted")
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise


async def terminate_document_workflows(document_id: str) -> None:
    """Best-effort terminate of every workflow tied to a document.

    Called on document delete — without this, a running OCR keeps burning
    GPU on a nonexistent document and orphaned digest/translation workflows
    sit in their wait-gates. One unreachable workflow must not block the
    delete or the remaining terminations.
    """
    from data.db_models import Translation

    db_manager = get_db_manager()
    with db_manager.session() as db:
        langs = [
            row[0]
            for row in db.query(Translation.target_language)
            .filter(Translation.document_id == document_id)
            .distinct()
            .all()
        ]

    async def _quiet(coro) -> None:
        try:
            await coro
        except Exception:
            logger.warning(
                "Terminate workflow failed for %s (continuing delete)",
                document_id,
                exc_info=True,
            )

    await _quiet(terminate_running_extraction(document_id))
    await _quiet(terminate_running_digest(document_id))
    for lang in langs:
        await _quiet(terminate_running_translation(document_id, lang))


async def start_translation_workflow(
    document_id: str,
    translation_id: str,
    parent_task_id: str,
    target_language: str,
    domain: str = "general",
    fairness_key: str | None = None,
) -> str:
    """Start TranslationWorkflow. Returns the workflow id."""
    from data.db_models import Task
    from services.pipeline.admission import mark_dispatched
    from workflows.activities.translation_activities import TranslationRunInput
    from workflows.translation_workflow import TranslationWorkflow

    db_manager = get_db_manager()
    with db_manager.session() as db:
        task = db.query(Task).filter(Task.id == parent_task_id).first()
        if task is not None:
            mark_dispatched(task)
            db.commit()

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
    from data.db_models import Task
    from services.pipeline.admission import mark_dispatched
    from workflows.activities.extraction_activities import ExtractionRunInput
    from workflows.extraction_workflow import ExtractionWorkflow

    db_manager = get_db_manager()
    with db_manager.session() as db:
        task = db.query(Task).filter(Task.id == parent_task_id).first()
        if task is not None:
            mark_dispatched(task)
            db.commit()

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


async def cancel_digest_workflow(document_id: str) -> bool:
    """Request cancellation of a running digest. Returns True if one was found."""
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id_for_document(document_id))
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


async def cancel_extraction_workflow(document_id: str) -> bool:
    """Request cancellation of a running extraction. Returns True if one was found."""
    client = await get_temporal_client()
    handle = client.get_workflow_handle(extraction_workflow_id(document_id))
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
