"""Deliver an integration job's outcome to the library system.

Two channels, both optional and both retried by the workflow:

* **MSSQL** — the job row on every stage change, the full result once final
  (``services/integration_sink.py``).
* **Callback** — one signed POST of the job view when the job is final. The
  library then pulls ``links.result`` (a thin event keeps the POST small and
  the pull API the single source of the data).

Each attempt is recorded in ``job.delivery``; a failure is raised so Temporal
retries it, and never touches the job's processing status.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy.orm import Session

from data.db_models import IntegrationJob
from services import integration_service as svc
from utils.integration_security import sign_callback
from utils.url_safety import UnsafeUrl, validate_callback_url

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_S = 15


class DeliveryError(RuntimeError):
    pass


def _record(db: Session, job: IntegrationJob, channel: str, status: str, attempt: int,
            error: Optional[str] = None) -> None:
    # MSSQL and callback deliver side by side in separate sessions: re-read the
    # row under a lock (Postgres) so one channel never writes back a stale copy
    # of the other's record.
    db.commit()
    job = (
        db.query(IntegrationJob)
        .filter(IntegrationJob.id == job.id)
        .with_for_update()
        .populate_existing()
        .one()
    )
    delivery = dict(job.delivery or {})
    delivery[channel] = {
        "status": status,
        "attempts": attempt,
        "at": datetime.utcnow().isoformat(timespec="seconds"),
        "error": error,
    }
    job.delivery = delivery
    db.commit()


def deliver_mssql(db: Session, job: IntegrationJob, base_url: str, *, final: bool, sink,
                  attempt: int = 1) -> str:
    if sink is None:
        return "SKIPPED"
    try:
        view = svc.job_view(job, base_url)
        if final:
            sink.write(view, result=svc.build_result(db, job, base_url))
        else:
            sink.write(view)
    except Exception as exc:
        _record(db, job, "mssql", "FAILED", attempt, f"{type(exc).__name__}: {exc}")
        raise
    _record(db, job, "mssql", "DELIVERED", attempt)
    return "DELIVERED"


def deliver_callback(db: Session, job: IntegrationJob, base_url: str, *,
                     post: Callable, attempt: int = 1) -> str:
    if not job.callback_url:
        return "SKIPPED"
    try:
        validate_callback_url(job.callback_url)
    except UnsafeUrl as exc:
        _record(db, job, "callback", "FAILED", attempt, f"refused: {exc}")
        raise DeliveryError(f"callback refused: {exc}") from exc
    event = f"job.{job.status.lower()}"
    body = json.dumps(
        {"event": event, "job": svc.job_view(job, base_url)}, ensure_ascii=False
    ).encode("utf-8")
    ts = int(time.time())
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-DocuFlow-Event": event,
        "X-DocuFlow-Delivery": uuid.uuid4().hex,
        "X-DocuFlow-Timestamp": str(ts),
        "X-DocuFlow-Signature": sign_callback(job.client.webhook_secret, ts, body),
    }
    try:
        resp = post(job.callback_url, content=body, headers=headers, timeout=CALLBACK_TIMEOUT_S)
    except Exception as exc:
        _record(db, job, "callback", "FAILED", attempt, f"{type(exc).__name__}: {exc}")
        raise DeliveryError(f"callback to {job.callback_url} failed: {exc}") from exc
    if not 200 <= resp.status_code < 300:
        error = f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
        _record(db, job, "callback", "FAILED", attempt, error)
        raise DeliveryError(f"callback to {job.callback_url} rejected: {error}")
    _record(db, job, "callback", "DELIVERED", attempt)
    return "DELIVERED"
