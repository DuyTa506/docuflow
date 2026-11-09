"""Liveness, readiness, and host-capacity probes.

``/health/live`` is process-up. ``/health/ready`` checks PostgreSQL, MinIO and
Temporal so systemd/compose do not send traffic at a half-started API.
``/health/capacity`` is the operator view of GPU leases and admission slots.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from api.dependencies import require_role
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def health_live():
    return {"status": "ok"}


def _check_postgres() -> dict:
    from data.database import get_db_manager

    try:
        with get_db_manager().session() as db:
            db.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def _check_minio() -> dict:
    try:
        from services.object_storage import get_object_storage

        storage = get_object_storage()
        storage.ensure_bucket()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


async def _check_temporal() -> dict:
    try:
        from temporalio.service import RPCError, RPCStatusCode

        from services.pipeline.temporal_client import get_temporal_client

        client = await get_temporal_client()
        handle = client.get_workflow_handle("__docuflow_health_probe__")
        try:
            await handle.describe()
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                return {"ok": False, "error": str(exc)[:200]}
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


@router.get("/health/ready")
async def health_ready():
    checks = {
        "postgres": _check_postgres(),
        "minio": _check_minio(),
        "temporal": await _check_temporal(),
    }
    # Anyone who can reach the port may call this, so it says which dependency is
    # down but never why — the reason (hosts, ports, driver errors) is logged instead.
    for name, item in checks.items():
        if not item.get("ok"):
            logger.warning("Readiness: %s is down: %s", name, item.get("error"))
    public = {name: {"ok": bool(item.get("ok"))} for name, item in checks.items()}
    if not all(item["ok"] for item in public.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": public})
    return {"status": "ok", "checks": public}


@router.get("/health/capacity", dependencies=[Depends(require_role("ADMIN"))])
async def health_capacity():
    from data.database import get_db_manager
    from services.gpu_lease import gpu_snapshot, lease_status
    from services.pipeline.admission import admission_snapshot

    with get_db_manager().session() as db:
        admission = admission_snapshot(db)
    return {
        "status": "ok",
        "admission": admission,
        "gpu": gpu_snapshot(),
        "leases": lease_status(),
        "host": {
            "api_port": settings.api_port,
            "ocr_use_temporal": settings.ocr_use_temporal,
            "max_upload_bytes": settings.max_upload_bytes,
        },
    }
