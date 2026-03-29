"""
Search endpoint.

GET /api/v2/search?q=...&search_in=title,content,keywords&language=en&limit=20
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from data.db_models import User
from services.search_service import SearchService

router = APIRouter(prefix="/api/v2", tags=["search"])
_svc = SearchService()


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    search_in: Optional[str] = Query(
        None,
        description="Comma-separated fields: title,content,keywords,translations",
    ),
    language: Optional[str] = Query(None, description="Filter translations by language"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Full-text search across the document library."""
    fields = search_in.split(",") if search_in else None
    return _svc.search(
        db,
        query=q,
        search_in=fields,
        language=language,
        limit=limit,
        offset=offset,
    )
