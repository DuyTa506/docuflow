"""Thin adapters: run digest stages inside Temporal activities (no task_manager.submit)."""

from typing import Any, Optional

from data.database import get_db_manager
from data.db_models import Task
from data.id_generator import IdGenerator
from services.bibliographic_service import BibliographicService
from services.keyword_service import KeywordService
from services.main_content_service import MainContentService
from services.research_direction_service import ResearchDirectionService
from services.summarization_service import SummarizationService
from services.usage_scope_service import UsageScopeService
from serving.tree_indexing_service import TreeIndexingService


async def ensure_extracted(document_id: str) -> None:
    db_manager = get_db_manager()
    with db_manager.session() as db:
        from data.repositories import DocumentRepository

        doc = DocumentRepository(db).get(document_id)
        if not doc:
            # Non-retryable: a missing document never comes back (deleted
            # mid-run) — the wait loop must not poll for it forever.
            from temporalio.exceptions import ApplicationError

            raise ApplicationError(f"Document {document_id} not found", non_retryable=True)
        if doc.processing_status == "FAILED":
            # Non-retryable: the gate's unlimited-retry wait loop must not
            # park the workflow forever on a document whose OCR already died.
            from temporalio.exceptions import ApplicationError

            raise ApplicationError(
                f"OCR/extraction failed for {document_id} — retry OCR first.",
                non_retryable=True,
            )
        if doc.processing_status != "EXTRACTED":
            # Retryable on purpose: with Temporal's default retry policy the
            # calling activity becomes a wait-for-extraction poll loop.
            raise ValueError(
                f"Document not extracted (status={doc.processing_status}). Run OCR first."
            )


# Digest defaults; a standalone /tree-index call overrides these per request.
BUILD_TREE_DEFAULTS: dict[str, Any] = {
    "use_spatial_metadata": True,
    "if_add_node_summary": "no",
    "if_thinning": True,
}


async def run_build_tree(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict[str, Any]:
    kwargs = {**BUILD_TREE_DEFAULTS, **(options or {})}
    db_manager = get_db_manager()
    with db_manager.session() as db:
        svc = TreeIndexingService(db)
        result = await svc.build_enhanced_tree_index(document_id=document_id, **kwargs)
    return result or {}


async def run_bibliographic(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await BibliographicService().run_for_pipeline(document_id, task_id=task_id)


async def run_keywords(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await KeywordService().run_for_pipeline(document_id, task_id=task_id)


async def run_research_directions(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await ResearchDirectionService().run_for_pipeline(document_id, task_id=task_id)


async def run_usage_scope(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await UsageScopeService().run_for_pipeline(document_id, task_id=task_id)


async def run_summarize(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await SummarizationService().run_for_pipeline(document_id, task_id=task_id)


async def run_main_content(
    document_id: str, task_id: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    return await MainContentService().run_for_pipeline(document_id, task_id=task_id)


def create_parent_task(db, document_id: str) -> str:
    """Create DIGEST_PIPELINE parent task row for UI compatibility."""
    raw_id = IdGenerator.next_id(db, "tasks")
    seq_num = raw_id.split("_")[-1]
    task_id = f"DIGEST_PIPELINE_{seq_num}"
    db.add(
        Task(
            id=task_id,
            document_id=document_id,
            task_type="DIGEST_PIPELINE",
            status="PENDING",
            progress=0,
            message="Đã đưa tổng thuật vào hàng chờ",
        )
    )
    db.commit()
    return task_id
