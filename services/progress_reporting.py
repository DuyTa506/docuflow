"""Versioned structured progress with ContextVar stage forwarding."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from data.database import get_db_manager
from services.eta.types import PROGRESS_VERSION, sanitize_progress_meta
from services.task_manager import TaskManager

ProgressSink = Callable[[int, str, dict], None]

_sink: ContextVar[Optional[ProgressSink]] = ContextVar("progress_sink", default=None)
_defaults: ContextVar[dict] = ContextVar("progress_defaults", default={})
_task_id: ContextVar[Optional[str]] = ContextVar("progress_task_id", default=None)


@dataclass(frozen=True)
class ProgressSnapshot:
    pipeline: str
    phase: str = "active"
    mode: Optional[str] = None
    stage: Optional[str] = None
    unit_kind: Optional[str] = None
    units_done: Optional[int] = None
    units_total: Optional[int] = None
    attempt: int = 1
    target_language: Optional[str] = None
    feature_bucket: Optional[str] = None
    checkpoint_units: Optional[int] = None

    def as_dict(self) -> dict:
        return sanitize_progress_meta({"version": PROGRESS_VERSION, **self.__dict__}) or {}


@contextmanager
def progress_context(
    *,
    task_id: Optional[str] = None,
    sink: Optional[ProgressSink] = None,
    defaults: Optional[dict[str, Any]] = None,
) -> Iterator[None]:
    """Install defaults/sink across awaited calls, restoring them afterward."""

    merged = {**_defaults.get(), **(defaults or {})}
    defaults_token = _defaults.set(merged)
    task_token = _task_id.set(task_id if task_id is not None else _task_id.get())
    sink_token = _sink.set(sink if sink is not None else _sink.get())
    try:
        yield
    finally:
        _sink.reset(sink_token)
        _task_id.reset(task_token)
        _defaults.reset(defaults_token)


def emit_progress(
    task_id: Optional[str],
    progress: int,
    message: str,
    **structured: Any,
) -> None:
    """Report one structured snapshot without interpreting ``message``."""

    effective_task_id = task_id or _task_id.get()
    values = {**_defaults.get(), **structured}
    meta = sanitize_progress_meta({"version": PROGRESS_VERSION, **values})
    sink = _sink.get()
    if sink is not None and meta is not None:
        sink(progress, message, meta)
        return
    if not effective_task_id:
        return
    with get_db_manager().session() as db:
        TaskManager.update_progress(db, effective_task_id, progress, message, meta)


def emit_current_units(
    progress: int,
    message: str,
    *,
    units_done: int,
    units_total: int,
    unit_kind: Optional[str] = None,
) -> None:
    """Emit units from a generic worker only when a progress context is active."""

    if not _defaults.get() and _sink.get() is None and _task_id.get() is None:
        return
    emit_progress(
        None,
        progress,
        message,
        units_done=units_done,
        units_total=units_total,
        **({"unit_kind": unit_kind} if unit_kind else {}),
    )
