"""Temporal activities for the translation workflow.

The whole routed translation runs as one heartbeated activity; resume
granularity comes from the MinIO-backed TranslationUnitCache inside
StructuredTranslator — on retry/worker-restart, already-translated units are
cache hits and only the missing remainder is re-translated. This reuses every
existing translation mode unchanged instead of re-implementing per-mode unit
extraction/assembly in activity code.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from temporalio import activity

from data.database import get_db_manager
from workflows.activities._common import _with_heartbeat


@dataclass
class TranslationRunInput:
    document_id: str
    translation_id: str
    parent_task_id: str
    target_language: str
    domain: str = "general"


def _set_translation_status(translation_id: str, status: str) -> None:
    from data.db_models import Translation

    with get_db_manager().session() as db:
        t = db.query(Translation).filter(Translation.id == translation_id).first()
        if t:
            t.status = status


@activity.defn(name="run_translation")
async def run_translation_activity(inp: TranslationRunInput) -> dict[str, Any]:
    from services.translation_service import TranslationService
    from services.translators._cache import TranslationUnitCache

    cache = TranslationUnitCache(
        inp.document_id, inp.translation_id, target_lang=inp.target_language
    )
    warm = cache.load()
    if warm:
        activity.logger.info(
            "Translation cache warm for %s: %d unit(s) resume from checkpoint",
            inp.translation_id,
            warm,
        )

    _set_translation_status(inp.translation_id, "IN_PROGRESS")

    svc = TranslationService()
    result, target_language = await _with_heartbeat(
        svc._run_translation(
            inp.document_id,
            inp.target_language,
            inp.domain,
            inp.translation_id,
            task_id=inp.parent_task_id,
            unit_cache=cache,
        )
    )

    meta = {
        "translation_length": len(result.get("translated_content") or ""),
        "target_language": target_language,
        "translation_mode": result.get("translation_mode"),
    }
    if result.get("degraded_paragraphs"):
        meta["degraded_paragraphs"] = result["degraded_paragraphs"]
    if result.get("degraded_units"):
        meta["degraded_units"] = result["degraded_units"]
    return meta


@activity.defn(name="export_translation")
async def export_translation_activity(
    inp: TranslationRunInput, meta: Optional[dict] = None
) -> dict[str, Any]:
    from data.db_models import Task, Translation
    from services.export_service import export_service

    db_manager = get_db_manager()
    with db_manager.session() as db:
        await _with_heartbeat(
            export_service.cache_translation_exports(db, inp.document_id, inp.translation_id)
        )

    # Translation COMPLETED + parent Task COMPLETED/result in ONE commit —
    # a crash between them must never leave a half-finished state.
    with db_manager.session() as db:
        t = db.query(Translation).filter(Translation.id == inp.translation_id).first()
        if t:
            t.status = "COMPLETED"
        task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
        if task:
            task.status = "COMPLETED"
            task.progress = 100
            task.result = meta or {}
            task.message = "Translation completed"
            task.updated_at = datetime.utcnow()
    return meta or {}


@activity.defn(name="fail_translation")
async def fail_translation_activity(inp: TranslationRunInput, error: str) -> None:
    from data.db_models import Task, Translation

    with get_db_manager().session() as db:
        t = db.query(Translation).filter(Translation.id == inp.translation_id).first()
        if t:
            t.status = "FAILED"
        task = db.query(Task).filter(Task.id == inp.parent_task_id).first()
        if task:
            task.status = "FAILED"
            task.error = error[:2000]
            task.updated_at = datetime.utcnow()
