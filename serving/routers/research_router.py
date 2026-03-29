"""
Research direction endpoints.

POST /api/v2/documents/{id}/research-directions    — Extract (background)
GET  /api/v2/documents/{id}/research-directions    — Get directions
POST /api/v2/research-directions/catalog           — Add to catalog (ADMIN)
GET  /api/v2/research-directions/catalog           — List catalog
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, require_role
from api.schemas import (
    CatalogDirectionRequest,
    CatalogDirectionResponse,
    CatalogListItem,
    ResearchDirectionsResponse,
    ResearchListItem,
    TaskSubmittedResponse,
)
from data.db_models import User
from data.repositories import DocumentRepository, ResearchRepository
from services.research_direction_service import ResearchDirectionService

router = APIRouter(tags=["research-directions"])
_svc = ResearchDirectionService()


# ── Document-level endpoints ────────────────────────────────────────

@router.post(
    "/api/v2/documents/{document_id}/research-directions",
    response_model=TaskSubmittedResponse,
)
async def start_research_direction_extraction(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Identify research directions as a background task."""
    try:
        task_id = _svc.submit(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, message="Research direction extraction task submitted")


@router.get(
    "/api/v2/documents/{document_id}/research-directions",
    response_model=ResearchDirectionsResponse,
)
async def get_research_directions(
    document_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get identified research directions for a document."""
    doc_repo = DocumentRepository(db)
    if not doc_repo.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    research_repo = ResearchRepository(db)
    assocs = research_repo.get_directions(document_id)

    return ResearchDirectionsResponse(
        document_id=document_id,
        directions=[
            ResearchListItem(
                direction_name=rd.direction_name,
                confidence=assoc.confidence,
                reasoning=assoc.reasoning,
                is_predefined=rd.is_predefined,
            )
            for assoc, rd in assocs
        ],
    )


# ── Catalog endpoints ───────────────────────────────────────────────

@router.post(
    "/api/v2/research-directions/catalog",
    response_model=CatalogDirectionResponse,
    status_code=201,
)
async def add_catalog_direction(
    body: CatalogDirectionRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("ADMIN")),
):
    """Add a predefined research direction to the catalog (ADMIN only)."""
    research_repo = ResearchRepository(db)
    existing = research_repo.get_by_name(body.direction_name)
    if existing:
        raise HTTPException(status_code=400, detail="Direction already exists")

    rd = research_repo.add_catalog(body.direction_name)
    return CatalogDirectionResponse(
        id=rd.id,
        direction_name=rd.direction_name,
        is_predefined=rd.is_predefined,
        created_at=rd.created_at.isoformat() if rd.created_at else None,
    )


@router.get(
    "/api/v2/research-directions/catalog",
    response_model=List[CatalogListItem],
)
async def list_catalog(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all research directions in the catalog."""
    research_repo = ResearchRepository(db)
    dirs = research_repo.get_catalog()
    return [
        CatalogListItem(
            id=rd.id,
            direction_name=rd.direction_name,
            is_predefined=rd.is_predefined,
            created_at=rd.created_at.isoformat() if rd.created_at else None,
        )
        for rd in dirs
    ]
