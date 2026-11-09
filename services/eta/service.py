"""Atomic ETA state evolution used by every task writer."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import settings
from services.eta import digest, ocr, translation
from services.eta.profiles import get_profile, profile_key, record_observation
from services.eta.smoothing import clamp_step, ema, exceeds_hysteresis
from services.eta.types import Estimate, terminal_eta, utc_iso

logger = logging.getLogger(__name__)


def _public_profile_enabled(key: str) -> bool:
    configured = {
        item.strip() for item in settings.eta_public_profile_keys.split(",") if item.strip()
    }
    return "*" in configured or key in configured


def _parse_time(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _dimensions(meta: dict) -> tuple[str, str, str]:
    pipeline = meta.get("pipeline")
    if pipeline == "extract":
        return ocr.dimensions(meta)
    if pipeline == "translate":
        return translation.dimensions(meta)
    return digest.dimensions(meta)


def _segment_key(meta: dict) -> str:
    return "|".join(
        [
            str(meta.get("pipeline") or ""),
            str(meta.get("mode") or ""),
            str(meta.get("stage") or ""),
            str(meta.get("phase") or ""),
            str(meta.get("attempt") or 1),
        ]
    )


def _observe_state(
    db: Session,
    task,
    state: dict,
    *,
    now: datetime,
    success: bool,
) -> None:
    started = _parse_time(state.get("segment_started_at"))
    if not started:
        return
    _log_prediction_outcomes(task.id, state, now=now, success=success)
    dimensions = state.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        return
    start_units = float(state.get("segment_start_units") or 0)
    last_units = float(state.get("last_units") or 0)
    units = max(0.0, last_units - start_units)
    # Calibrated short/export stages are duration profiles (one stage unit).
    if units <= 0:
        units = 1.0
    record_observation(
        db,
        task_id=task.id,
        segment_key=str(state.get("segment_key") or "unknown"),
        pipeline=dimensions[0],
        mode_stage=dimensions[1],
        feature_bucket=dimensions[2],
        units=units,
        duration_seconds=max(0.0, (now - started).total_seconds()),
        success=success,
        now=now,
    )


def _log_prediction_outcomes(task_id: str, state: dict, *, now: datetime, success: bool) -> None:
    predictions = list(state.get("shadow_history") or [])
    if not predictions and state.get("shadow_prediction"):
        predictions = [state["shadow_prediction"]]
    for prediction in predictions:
        calculated_at = _parse_time(prediction.get("calculated_at"))
        if not calculated_at:
            continue
        actual_remaining = max(0.0, (now - calculated_at).total_seconds())
        low = prediction.get("low_seconds")
        high = prediction.get("high_seconds")
        inside = bool(
            low is not None and high is not None and float(low) <= actual_remaining <= float(high)
        )
        midpoint = (float(low) + float(high)) / 2 if low is not None and high is not None else None
        error = (
            abs(midpoint - actual_remaining) / actual_remaining
            if midpoint is not None and actual_remaining > 0
            else None
        )
        logger.info(
            "eta_outcome task_id=%s profile_key=%s predicted_low=%s predicted_high=%s "
            "actual_remaining=%.1f inside_range=%s absolute_percentage_error=%s success=%s",
            task_id,
            state.get("profile_key"),
            low,
            high,
            actual_remaining,
            inside,
            round(error, 4) if error is not None else None,
            success,
        )


def _profile_estimate(db: Session, dims: tuple[str, str, str], remaining: float):
    profile = get_profile(db, *dims)
    if not profile:
        return None
    return (
        max(0.0, remaining) * profile.rate_p50,
        max(0.0, remaining) * profile.rate_p90,
        profile,
    )


def _digest_tail(db: Session, meta: dict) -> Optional[tuple[float, float]]:
    stages = meta.get("stages") or {}
    current = str(meta.get("stage") or "")

    def lookup(stage: str):
        stage_meta = dict(stages.get(stage) or {})
        # FINALIZE observations are stored under FINALIZE:exporting; forcing
        # phase=active here looked up a key that never exists and wiped ETA.
        phase = "exporting" if stage == "FINALIZE" else "active"
        stage_meta.update(
            {
                "version": 1,
                "pipeline": "digest",
                "stage": stage,
                "phase": phase,
            }
        )
        dims = digest.dimensions(stage_meta)
        profile = get_profile(db, *dims)
        if not profile:
            return None
        # Short stage profiles are seconds per one completed stage.
        return profile.rate_p50, profile.rate_p90

    return digest.remaining_profile_seconds(stages, lookup, current_stage=current)


def update_eta(
    db: Session,
    task,
    meta: dict,
    *,
    now: Optional[datetime] = None,
) -> tuple[dict, dict]:
    """Advance private estimator state and return ``(public_eta, state)``."""

    now = now or datetime.utcnow()
    phase = str(meta.get("phase") or "active")
    existing = dict(task.eta_estimator_state or {})
    digest_root: Optional[dict] = None
    digest_stage = str(meta.get("stage") or "UNKNOWN")
    if meta.get("pipeline") == "digest" and meta.get("mode") == "digest_pipeline":
        digest_root = existing
        stage_states = digest_root.get("stage_states") or {}
        existing = dict(stage_states.get(digest_stage) or {})

    def packaged(current: dict) -> dict:
        if digest_root is None:
            return current
        stage_states = dict(digest_root.get("stage_states") or {})
        stage_states[digest_stage] = current
        return {
            **digest_root,
            "version": 1,
            "active_stage": digest_stage,
            "stage_states": stage_states,
        }

    if not settings.eta_enabled:
        return Estimate("unknown").public(now), packaged(existing)
    if phase == "waiting_upstream":
        return Estimate("waiting_upstream").public(now), packaged(existing)
    if phase in {"classifying", "routing", "queued", "pending"}:
        return Estimate("unknown").public(now), packaged(existing)

    segment_key = _segment_key(meta)
    dims = _dimensions(meta)
    units_done = float(meta.get("units_done") or 0)
    units_total = float(meta.get("units_total") or 0)
    changed = bool(existing.get("segment_key") and existing.get("segment_key") != segment_key)
    if changed:
        previous_attempt = str(existing.get("attempt") or 1)
        current_attempt = str(meta.get("attempt") or 1)
        _observe_state(
            db,
            task,
            existing,
            now=now,
            success=previous_attempt == current_attempt,
        )
        logger.info(
            "eta_retry_or_segment_reset task_id=%s old=%s new=%s",
            task.id,
            existing.get("segment_key"),
            segment_key,
        )
        existing = {}

    state = {
        **existing,
        "version": 1,
        "segment_key": segment_key,
        "attempt": int(meta.get("attempt") or 1),
        "dimensions": list(dims),
        "profile_key": profile_key(*dims),
    }
    if not state.get("segment_started_at"):
        state["segment_started_at"] = utc_iso(now)
        state["segment_start_units"] = units_done
        state["last_units"] = units_done
        state["last_sample_at"] = utc_iso(now)
        state["last_movement_at"] = utc_iso(now)
        state["sample_count"] = 0

    last_units = float(state.get("last_units") or 0)
    last_sample_at = _parse_time(state.get("last_sample_at")) or now
    if units_done > last_units:
        elapsed = max(0.001, (now - last_sample_at).total_seconds())
        rate_sample = elapsed / (units_done - last_units)
        state["ema_seconds_per_unit"] = ema(
            state.get("ema_seconds_per_unit"),
            rate_sample,
            settings.eta_ema_alpha,
        )
        state["sample_count"] = int(state.get("sample_count") or 0) + 1
        state["last_units"] = units_done
        state["last_sample_at"] = utc_iso(now)
        state["last_movement_at"] = utc_iso(now)

    remaining = max(0.0, units_total - units_done) if units_total > 0 else 1.0
    prior = _profile_estimate(db, dims, remaining)
    last_movement = _parse_time(state.get("last_movement_at")) or now
    if prior:
        profile = prior[2]
        stall_after = max(
            float(settings.eta_stall_min_seconds),
            profile.rate_p90 * float(settings.eta_stall_p90_multiplier),
        )
        if remaining > 0 and (now - last_movement).total_seconds() > stall_after:
            state["transition"] = "stalled"
            logger.warning(
                "eta_stalled task_id=%s profile_key=%s idle_seconds=%.1f threshold=%.1f",
                task.id,
                state["profile_key"],
                (now - last_movement).total_seconds(),
                stall_after,
            )
            return Estimate("stalled", confidence=0.0, profile_key=state["profile_key"]).public(
                now
            ), packaged(state)

    samples = int(state.get("sample_count") or 0)
    live_rate = state.get("ema_seconds_per_unit")
    low: Optional[float] = None
    high: Optional[float] = None
    confidence = 0.0
    if prior:
        low, high = prior[0], prior[1]
        confidence = min(0.85, 0.45 + min(0.40, prior[2].sample_count / 100))
    if live_rate is not None and samples >= max(1, settings.eta_live_sample_threshold):
        live_low = remaining * float(live_rate) * 0.85
        live_high = remaining * float(live_rate) * 1.25
        if low is None:
            low, high = live_low, live_high
        else:
            live_weight = min(0.75, samples / 8)
            low = low * (1 - live_weight) + live_low * live_weight
            high = high * (1 - live_weight) + live_high * live_weight
        confidence = max(confidence, min(0.9, 0.5 + samples * 0.05))

    if (
        meta.get("pipeline") == "digest"
        and meta.get("mode") == "digest_pipeline"
        and low is not None
    ):
        # Incomplete downstream calibration must not erase the current-stage
        # estimate — publish what we know and omit the missing tail.
        tail = _digest_tail(db, meta)
        if tail is not None:
            low += tail[0]
            high = (high or low) + tail[1]

    eta_state = "exporting" if phase in {"exporting", "finalizing"} else "active"
    if low is None or high is None:
        public = Estimate(
            eta_state, confidence=confidence, profile_key=state["profile_key"]
        ).public(now)
        return public, packaged(state)

    previous_low = state.get("published_low_seconds")
    previous_high = state.get("published_high_seconds")
    if remaining <= 0:
        # Last unit done: step clamping would keep advertising work that no
        # longer exists.
        candidate_low, candidate_high = low, high
    else:
        candidate_low = clamp_step(previous_low, low, settings.eta_max_step_ratio)
        candidate_high = clamp_step(previous_high, high, settings.eta_max_step_ratio)
    # The absolute hysteresis floor (60s) is bigger than one clamped step
    # (25%) for any range under ~4 minutes, which froze the published values
    # until the task ended. A countdown must always be free to shrink, so gate
    # upward revisions only.
    shrinking = previous_low is None or (
        candidate_low < float(previous_low) or candidate_high < float(previous_high or 0)
    )
    should_publish = (
        shrinking
        or exceeds_hysteresis(
            previous_low,
            candidate_low,
            ratio=settings.eta_hysteresis_ratio,
            seconds=settings.eta_hysteresis_seconds,
        )
        or exceeds_hysteresis(
            previous_high,
            candidate_high,
            ratio=settings.eta_hysteresis_ratio,
            seconds=settings.eta_hysteresis_seconds,
        )
    )
    if should_publish:
        state["published_low_seconds"] = candidate_low
        state["published_high_seconds"] = max(candidate_low, candidate_high)
    else:
        candidate_low = float(previous_low)
        candidate_high = float(previous_high)
    state["shadow_prediction"] = {
        "low_seconds": candidate_low,
        "high_seconds": max(candidate_low, candidate_high),
        "calculated_at": utc_iso(now),
    }
    if should_publish:
        history = list(state.get("shadow_history") or [])
        history.append(state["shadow_prediction"])
        state["shadow_history"] = history[-50:]
    estimate = Estimate(
        eta_state,
        candidate_low,
        max(candidate_low, candidate_high),
        confidence,
        state["profile_key"],
    )
    logger.info(
        "eta_prediction task_id=%s profile_key=%s low=%.1f high=%.1f confidence=%.3f shadow=%s",
        task.id,
        state["profile_key"],
        candidate_low,
        candidate_high,
        confidence,
        settings.eta_shadow_mode or not _public_profile_enabled(state["profile_key"]),
    )
    return estimate.public(
        now,
        shadow=settings.eta_shadow_mode or not _public_profile_enabled(state["profile_key"]),
    ), packaged(state)


def finish_eta(
    db: Session,
    task,
    *,
    success: bool,
    now: Optional[datetime] = None,
) -> tuple[dict, dict]:
    now = now or datetime.utcnow()
    state = dict(task.eta_estimator_state or {})
    stage_states = state.get("stage_states")
    if isinstance(stage_states, dict):
        for stage_state in stage_states.values():
            if isinstance(stage_state, dict):
                _observe_state(db, task, stage_state, now=now, success=success)
    else:
        _observe_state(db, task, state, now=now, success=success)
    return terminal_eta(now), state
