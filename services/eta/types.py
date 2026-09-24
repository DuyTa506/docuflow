"""Typed, versioned structures shared by ETA strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

PROGRESS_VERSION = 1
ETA_STATES = {
    "unknown",
    "active",
    "waiting_upstream",
    "exporting",
    "stalled",
    "terminal",
}
ACTIVE_PHASES = {"active", "exporting", "finalizing"}


def utc_iso(value: datetime) -> str:
    value = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return value.isoformat().replace("+00:00", "Z")


def _number(value: Any, *, integer: bool = False) -> Optional[float | int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def sanitize_progress_meta(value: Any) -> Optional[dict]:
    """Return the public allow-listed progress contract or ``None``."""

    if not isinstance(value, dict):
        return None
    version = _number(value.get("version"), integer=True)
    pipeline = str(value.get("pipeline") or "").strip().lower()
    phase = str(value.get("phase") or "active").strip().lower()
    if version != PROGRESS_VERSION or pipeline not in {"extract", "translate", "digest"}:
        return None

    clean: dict[str, Any] = {
        "version": PROGRESS_VERSION,
        "pipeline": pipeline,
        "phase": phase,
        "mode": str(value.get("mode") or "").strip() or None,
        "stage": str(value.get("stage") or "").strip() or None,
        "unit_kind": str(value.get("unit_kind") or "").strip() or None,
        "units_done": _number(value.get("units_done"), integer=True),
        "units_total": _number(value.get("units_total"), integer=True),
        "attempt": _number(value.get("attempt"), integer=True) or 1,
        "target_language": str(value.get("target_language") or "").strip() or None,
        "translation_id": str(value.get("translation_id") or "").strip() or None,
        "feature_bucket": str(value.get("feature_bucket") or "").strip() or None,
        "checkpoint_units": _number(value.get("checkpoint_units"), integer=True),
    }
    stages = value.get("stages")
    if isinstance(stages, dict):
        clean["stages"] = {
            str(name)[:80]: sanitize_stage_progress(stage)
            for name, stage in stages.items()
            if isinstance(stage, dict)
        }
    return {key: item for key, item in clean.items() if item is not None}


def sanitize_stage_progress(value: dict) -> dict:
    clean = {
        "phase": str(value.get("phase") or "active").strip().lower(),
        "unit_kind": str(value.get("unit_kind") or "").strip() or None,
        "units_done": _number(value.get("units_done"), integer=True),
        "units_total": _number(value.get("units_total"), integer=True),
        "attempt": _number(value.get("attempt"), integer=True) or 1,
        "progress": min(100, int(_number(value.get("progress"), integer=True) or 0)),
    }
    return {key: item for key, item in clean.items() if item is not None}


def terminal_eta(now: datetime) -> dict:
    return {
        "state": "terminal",
        "low_seconds": None,
        "high_seconds": None,
        "confidence": 1.0,
        "estimated_finish_at": None,
        "calculated_at": utc_iso(now),
    }


def sanitize_eta(value: Any) -> Optional[dict]:
    if not isinstance(value, dict):
        return None
    state = str(value.get("state") or "unknown")
    if state not in ETA_STATES:
        state = "unknown"
    low = _number(value.get("low_seconds"), integer=True)
    high = _number(value.get("high_seconds"), integer=True)
    if low is not None and high is not None:
        high = max(low, high)
    confidence = _number(value.get("confidence"))
    return {
        "state": state,
        "low_seconds": low,
        "high_seconds": high,
        "confidence": max(0.0, min(1.0, float(confidence or 0))),
        "estimated_finish_at": str(value.get("estimated_finish_at") or "") or None,
        "calculated_at": str(value.get("calculated_at") or "") or None,
    }


@dataclass(frozen=True)
class Estimate:
    state: str
    low_seconds: Optional[float] = None
    high_seconds: Optional[float] = None
    confidence: float = 0.0
    profile_key: Optional[str] = None

    def public(self, now: datetime, *, shadow: bool = False) -> dict:
        low = None if shadow or self.low_seconds is None else max(0, int(round(self.low_seconds)))
        high = (
            None
            if shadow or self.high_seconds is None
            else max(low or 0, int(round(self.high_seconds)))
        )
        finish = None
        if high is not None:
            from datetime import timedelta

            finish = utc_iso(now + timedelta(seconds=(low + high) / 2))
        return {
            "state": self.state if self.state in ETA_STATES else "unknown",
            "low_seconds": low,
            "high_seconds": high,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "estimated_finish_at": finish,
            "calculated_at": utc_iso(now),
        }
