"""Submit-layer admission: bound in-flight GPU/LLM work before Temporal starts.

Temporal fairness spreads *dispatch* across users. It does not stop us from
starting a third digest on a host whose profile says two. Counting OPEN Task
rows (except ``queued`` waiters) is the single-host stand-in for a scheduler
queue. HTTP submit always accepts the click: if this host is full the row
stays PENDING with ``progress_meta.queued`` until a slot frees.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from config.capacity import (
    HEAVY_STAGE_TYPES,
    SLOT_DIGEST,
    SLOT_EXTRACT,
    SLOT_STAGE,
    SLOT_TRANSLATE,
    CapacityProfile,
    capacity_profile,
)
from data.db_models import Document, Task

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("PENDING", "RUNNING")

_SLOT_LABELS = {
    SLOT_DIGEST: "tổng thuật",
    SLOT_EXTRACT: "OCR",
    SLOT_TRANSLATE: "dịch",
    SLOT_STAGE: "tổng thuật",
}

_SLOT_TYPES = {
    SLOT_DIGEST: frozenset({SLOT_DIGEST, *HEAVY_STAGE_TYPES}),
    SLOT_EXTRACT: frozenset({SLOT_EXTRACT}),
    SLOT_TRANSLATE: frozenset({SLOT_TRANSLATE}),
    SLOT_STAGE: frozenset({*HEAVY_STAGE_TYPES}),
}


QUEUED_META_KEY = "queued"


def is_queued(task) -> bool:
    """Waiting for a GPU/LLM slot — does not occupy capacity."""
    meta = task.progress_meta if isinstance(getattr(task, "progress_meta", None), dict) else {}
    return bool(meta.get(QUEUED_META_KEY))


def mark_queued(task, extra: Optional[dict] = None, message: str = "Đang chờ máy rảnh…") -> None:
    meta = dict(task.progress_meta) if isinstance(task.progress_meta, dict) else {}
    meta[QUEUED_META_KEY] = True
    if extra:
        meta.update({k: v for k, v in extra.items() if v is not None})
    task.progress_meta = meta
    task.status = "PENDING"
    task.message = message


def mark_dispatched(task) -> None:
    from datetime import datetime

    meta = dict(task.progress_meta) if isinstance(task.progress_meta, dict) else {}
    if QUEUED_META_KEY not in meta:
        return
    meta.pop(QUEUED_META_KEY, None)
    task.progress_meta = meta or None
    # Bump so reconcile grace is measured from dispatch, not insert — otherwise
    # a long-queued row can race-fail as "no workflow" right after unqueue.
    task.updated_at = datetime.utcnow()


class AdmissionRejected(Exception):
    """Host or per-user capacity is full. Submit paths queue; do not HTTP 429."""

    def __init__(
        self,
        message: str,
        *,
        slot: str,
        current: int,
        limit: int,
        retry_after: int = 30,
    ):
        super().__init__(message)
        self.slot = slot
        self.current = current
        self.limit = limit
        self.retry_after = retry_after

    def as_detail(self) -> dict:
        return {
            "error": "admission_rejected",
            "message": str(self),
            "slot": self.slot,
            "current": self.current,
            "limit": self.limit,
            "retry_after": self.retry_after,
        }


def _limit_for(slot: str, cap: CapacityProfile) -> int:
    if slot == SLOT_DIGEST or slot == SLOT_STAGE:
        return cap.max_digest_pipelines
    if slot == SLOT_EXTRACT:
        return cap.max_extractions
    if slot == SLOT_TRANSLATE:
        return cap.max_translations
    return cap.max_digest_pipelines


def count_open(db: Session, slot: str, *, excluding_task_id: Optional[str] = None) -> int:
    types = _SLOT_TYPES.get(slot, frozenset({slot}))
    q = db.query(Task).filter(Task.task_type.in_(types), Task.status.in_(OPEN_STATUSES))
    if excluding_task_id:
        q = q.filter(Task.id != excluding_task_id)
    return sum(1 for t in q.all() if not is_queued(t))


def count_user_open(db: Session, user_id: str, *, excluding_task_id: Optional[str] = None) -> int:
    if not user_id:
        return 0
    types = set().union(*_SLOT_TYPES.values())
    q = (
        db.query(Task)
        .join(Document, Task.document_id == Document.id)
        .filter(
            Document.user_id == user_id,
            Task.task_type.in_(types),
            Task.status.in_(OPEN_STATUSES),
        )
    )
    if excluding_task_id:
        q = q.filter(Task.id != excluding_task_id)
    return sum(1 for t in q.all() if not is_queued(t))


def assert_can_admit(
    db: Session,
    slot: str,
    *,
    user_id: Optional[str] = None,
    excluding_task_id: Optional[str] = None,
    cap: Optional[CapacityProfile] = None,
) -> None:
    """Raise AdmissionRejected when the host or the user is at capacity.

    ``excluding_task_id`` ignores a row the caller just inserted for this submit
    so the new job does not count against itself.
    """
    cap = cap or capacity_profile()
    limit = _limit_for(slot, cap)
    current = count_open(db, slot, excluding_task_id=excluding_task_id)
    if current >= limit:
        label = _SLOT_LABELS.get(slot, slot)
        raise AdmissionRejected(
            f"Máy đang đầy cho {label} ({current}/{limit}). Thử lại sau.",
            slot=slot,
            current=current,
            limit=limit,
            retry_after=30,
        )
    if user_id:
        user_current = count_user_open(db, user_id, excluding_task_id=excluding_task_id)
        if user_current >= cap.max_jobs_per_user:
            raise AdmissionRejected(
                f"Bạn đang chạy tối đa {cap.max_jobs_per_user} tác vụ "
                f"({user_current}/{cap.max_jobs_per_user}). Thử lại sau.",
                slot=slot,
                current=user_current,
                limit=cap.max_jobs_per_user,
                retry_after=20,
            )


def admission_snapshot(db: Session) -> dict:
    cap = capacity_profile()
    return {
        "profile": {
            "max_digest_pipelines": cap.max_digest_pipelines,
            "max_extractions": cap.max_extractions,
            "max_translations": cap.max_translations,
            "max_jobs_per_user": cap.max_jobs_per_user,
            "digest_group_a_parallelism": cap.digest_group_a_parallelism,
            "digest_group_b_parallel": cap.digest_group_b_parallel,
        },
        "open": {
            SLOT_DIGEST: count_open(db, SLOT_DIGEST),
            SLOT_EXTRACT: count_open(db, SLOT_EXTRACT),
            SLOT_TRANSLATE: count_open(db, SLOT_TRANSLATE),
        },
    }


def http_exception(exc: AdmissionRejected):
    from fastapi import HTTPException

    return HTTPException(
        status_code=429,
        detail=exc.as_detail(),
        headers={"Retry-After": str(exc.retry_after)},
    )
