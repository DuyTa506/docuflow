"""
Tree Index endpoints.

POST /api/v2/documents/{id}/tree-index  — Build tree index (background task)
GET  /api/v2/documents/{id}/tree-index  — Get latest tree index metadata
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, get_authorized_document
from api.schemas import TaskSubmittedResponse, TreeIndexRequest
from data.db_models import User
from data.repositories import DocumentRepository
from services.task_manager import task_manager, TaskManager
from data.database import get_db_manager

router = APIRouter(prefix="/api/v2/documents", tags=["tree-index"])


@router.post("/{document_id}/tree-index", response_model=TaskSubmittedResponse)
async def build_tree_index(
    document_id: str,
    body: TreeIndexRequest = TreeIndexRequest(),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Build a tree index for a document as a background task."""
    get_authorized_document(document_id, _user, db)

    from serving.tree_indexing_service import TreeIndexingService

    async def _run():
        db_manager = get_db_manager()
        with db_manager.session() as db2:
            svc = TreeIndexingService(db2)
            result = await svc.build_enhanced_tree_index(
                document_id=document_id,
                use_spatial_metadata=body.use_spatial_metadata,
                discover_implicit_sections=body.discover_implicit_sections,
                if_thinning=body.if_thinning,
                if_add_node_summary=body.if_add_node_summary,
                summary_token_threshold=body.summary_token_threshold,
                model=body.model,
                llm_provider=body.llm_provider,
                ollama_base_url=body.ollama_base_url,
                ollama_timeout=body.ollama_timeout,
            )
        return result

    task_id = task_manager.submit(
        db,
        document_id=document_id,
        task_type="BUILD_TREE",
        coro=_run(),
    )

    return TaskSubmittedResponse(
        task_id=task_id,
        message="Tree index build task submitted",
    )


@router.get("/{document_id}/tree-index")
async def get_tree_index(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get the latest tree index for a document."""
    get_authorized_document(document_id, _user, db)

    from serving.tree_indexing_service import TreeIndexingService

    svc = TreeIndexingService(db)
    tree = svc.get_tree_index(document_id)

    if not tree:
        raise HTTPException(
            status_code=404,
            detail="No tree index found for this document. Run POST /tree-index first.",
        )

    return tree
