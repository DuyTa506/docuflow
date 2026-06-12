"""
Search endpoint.

GET /api/v2/search?q=...&search_in=title,content,keywords&language=en&page=1&limit=20

Response envelope matches GET /api/v2/documents (items + pagination) plus ``query``.
"""
import math
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user
from api.schemas import SearchResponse
from data.db_models import User
from services.search_service import SearchService

router = APIRouter(prefix="/api/v2", tags=["search"])
_svc = SearchService()


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    search_in: Optional[str] = Query(
        None,
        description="Comma-separated fields: title,content,keywords,translations",
    ),
    language: Optional[str] = Query(None, description="Filter translations by language"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Full-text search across the document library."""
    fields = (
        [f.strip() for f in search_in.split(",") if f.strip()]
        if search_in
        else None
    )
    offset = (page - 1) * limit
    result = _svc.search(
        db,
        query=q,
        search_in=fields,
        language=language,
        limit=limit,
        offset=offset,
        user_id=user.id,
        is_admin=(user.role == "ADMIN"),
    )
    total = result.get("total", 0)
    total_pages = math.ceil(total / limit) if total > 0 else 1
    return SearchResponse(
        items=result["items"],
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        query=result["query"],
    )
