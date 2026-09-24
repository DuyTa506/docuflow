"""Server-Sent Events: push task status changes to the UI instantly.

GET /api/v2/events?token=<jwt>

A Postgres trigger (scripts/migrate_task_notify.py) NOTIFYs channel
'task_events' on every tasks INSERT/UPDATE — from ANY process (API or the
Temporal worker). This endpoint LISTENs and streams the payloads as SSE, so
the FE flips button states in <100ms instead of waiting for a polling tick.

Auth: EventSource cannot set headers, so the JWT rides a query parameter.
Non-admin users only receive events for documents they own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import select as _select

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["events"])

_KEEPALIVE_SECONDS = 15


def _authenticate(token: str) -> dict:
    from services.auth_service import AuthService

    payload = AuthService.decode_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def _event_visible_to(payload: dict, user_id: str, role: str, owner_cache: dict) -> bool:
    """ADMINs see everything; others only tasks on their own documents."""
    if role == "ADMIN":
        return True
    document_id = payload.get("document_id")
    if not document_id:
        return False
    if document_id not in owner_cache:
        from data.database import get_db_manager
        from data.db_models import Document

        with get_db_manager().session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            owner_cache[document_id] = doc.user_id if doc else None
    return owner_cache[document_id] == user_id


def _sanitize_event(payload: dict) -> dict:
    from services.eta import sanitize_eta, sanitize_progress_meta

    allowed = {
        key: payload.get(key)
        for key in (
            "task_id",
            "document_id",
            "task_type",
            "status",
            "progress",
            "message",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        )
    }
    allowed["message"] = str(allowed.get("message") or "")[:500]
    allowed["progress_meta"] = sanitize_progress_meta(payload.get("progress_meta"))
    allowed["eta"] = sanitize_eta(payload.get("eta"))
    return allowed


def _pg_dsn() -> str:
    # SQLAlchemy URL → libpq DSN (strip the +driver suffix).
    return settings.database_url.replace("postgresql+psycopg2://", "postgresql://", 1)


async def _task_event_stream(user_id: str, role: str):
    import psycopg2
    import psycopg2.extensions

    conn = psycopg2.connect(_pg_dsn())
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    owner_cache: dict = {}
    loop = asyncio.get_running_loop()
    try:
        cur = conn.cursor()
        cur.execute("LISTEN task_events;")
        yield ": connected\n\n"
        while True:
            ready = await loop.run_in_executor(
                None, _select.select, [conn], [], [], _KEEPALIVE_SECONDS
            )
            if not ready[0]:
                yield ": keepalive\n\n"
                continue
            conn.poll()
            while conn.notifies:
                note = conn.notifies.pop(0)
                try:
                    payload = json.loads(note.payload)
                except (ValueError, TypeError):
                    continue
                if _event_visible_to(payload, user_id, role, owner_cache):
                    public = _sanitize_event(payload)
                    yield f"event: task\ndata: {json.dumps(public, ensure_ascii=False)}\n\n"
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/events")
async def task_events(token: str = Query(..., description="JWT access token")):
    payload = _authenticate(token)
    if not settings.database_url.startswith("postgresql"):
        raise HTTPException(status_code=501, detail="SSE requires PostgreSQL")
    return StreamingResponse(
        _task_event_stream(payload["sub"], payload.get("role") or ""),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
