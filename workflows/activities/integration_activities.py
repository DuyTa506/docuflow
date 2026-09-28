"""Activities of IntegrationJobWorkflow: read the job's stages, deliver results.

Thin wrappers — the logic lives in services/integration_service.py and
services/integration_delivery.py, where it is tested without Temporal.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Optional

from temporalio import activity


@dataclass
class IntegrationJobInput:
    job_id: str
    poll_seconds: int = 20
    # What this workflow last mirrored to MSSQL; carried across continue-as-new.
    last_state: Optional[dict] = None


@dataclass
class DeliverInput:
    job_id: str
    final: bool


def _job(db, job_id: str):
    from data.db_models import IntegrationJob

    job = db.get(IntegrationJob, job_id)
    if job is None:
        from temporalio.exceptions import ApplicationError

        raise ApplicationError(f"Integration job {job_id} not found", non_retryable=True)
    return job


def _refresh(job_id: str) -> dict:
    from data.database import get_db_manager
    from services.integration_service import FINAL, refresh_job

    with get_db_manager().session() as db:
        job = _job(db, job_id)
        refresh_job(db, job)
        stages = dict(job.stages or {})
        return {"final": all(v in FINAL for v in stages.values()), "status": job.status, "stages": stages}


@activity.defn(name="refresh_integration_job")
async def refresh_integration_job_activity(job_id: str) -> dict:
    return await asyncio.to_thread(_refresh, job_id)


@functools.lru_cache(maxsize=1)
def _sink(url: str, schema: str):
    """One engine (and connection pool) per worker process, not per attempt."""
    from services.integration_sink import MssqlSink

    return MssqlSink(url, schema=schema or None) if url else None


def _deliver_mssql(inp: DeliverInput, attempt: int) -> str:
    from config.settings import settings
    from data.database import get_db_manager
    from services.integration_delivery import deliver_mssql

    sink = _sink(settings.integration_mssql_url, settings.integration_mssql_schema)
    with get_db_manager().session() as db:
        return deliver_mssql(
            db,
            _job(db, inp.job_id),
            settings.integration_public_base_url,
            final=inp.final,
            sink=sink,
            attempt=attempt,
        )


@activity.defn(name="deliver_integration_mssql")
async def deliver_mssql_activity(inp: DeliverInput) -> str:
    return await asyncio.to_thread(_deliver_mssql, inp, activity.info().attempt)


def _deliver_callback(job_id: str, attempt: int) -> str:
    import httpx

    from config.settings import settings
    from data.database import get_db_manager
    from services.integration_delivery import deliver_callback

    with get_db_manager().session() as db:
        return deliver_callback(
            db,
            _job(db, job_id),
            settings.integration_public_base_url,
            post=httpx.post,
            attempt=attempt,
        )


@activity.defn(name="deliver_integration_callback")
async def deliver_callback_activity(job_id: str) -> str:
    return await asyncio.to_thread(_deliver_callback, job_id, activity.info().attempt)
