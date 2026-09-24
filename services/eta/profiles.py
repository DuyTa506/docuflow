"""ETA observation persistence and bounded profile calibration."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import TaskEtaObservation, TaskEtaProfile
from services.eta.smoothing import percentile, winsorize


def profile_key(pipeline: str, mode_stage: str, feature_bucket: str) -> str:
    return f"{pipeline}|{mode_stage}|{feature_bucket}"


def get_profile(
    db: Session,
    pipeline: str,
    mode_stage: str,
    feature_bucket: str,
    *,
    require_calibrated: bool = True,
) -> Optional[TaskEtaProfile]:
    def fetch(bucket: str) -> Optional[TaskEtaProfile]:
        return (
            db.query(TaskEtaProfile)
            .filter(
                TaskEtaProfile.pipeline == pipeline,
                TaskEtaProfile.mode_stage == mode_stage,
                TaskEtaProfile.feature_bucket == bucket,
            )
            .first()
        )

    row = fetch(feature_bucket)
    if require_calibrated and (
        row is None or row.sample_count < max(1, settings.eta_profile_min_samples)
    ):
        # Read path only: borrow an adjacent size bucket so ETA can publish
        # while the exact key is still collecting samples. Write/refresh paths
        # keep require_calibrated=False and must stay on the exact key.
        for fallback in ("small", "medium", "large", "xlarge"):
            if fallback == feature_bucket:
                continue
            alt = fetch(fallback)
            if alt is not None and alt.sample_count >= max(1, settings.eta_profile_min_samples):
                return alt
        return None
    return row


def record_observation(
    db: Session,
    *,
    task_id: str,
    segment_key: str,
    pipeline: str,
    mode_stage: str,
    feature_bucket: str,
    units: float,
    duration_seconds: float,
    success: bool,
    now: Optional[datetime] = None,
) -> bool:
    """Insert one idempotent segment observation and refresh its profile."""

    units = max(0.0, float(units))
    duration_seconds = max(0.0, float(duration_seconds))
    if units <= 0 or duration_seconds <= 0:
        return False
    existing = (
        db.query(TaskEtaObservation.id)
        .filter(
            TaskEtaObservation.task_id == task_id,
            TaskEtaObservation.segment_key == segment_key,
        )
        .first()
    )
    if existing:
        return False
    try:
        with db.begin_nested():
            db.add(
                TaskEtaObservation(
                    task_id=task_id,
                    segment_key=segment_key,
                    pipeline=pipeline,
                    mode_stage=mode_stage,
                    feature_bucket=feature_bucket,
                    units=units,
                    active_duration_seconds=duration_seconds,
                    success=success,
                    created_at=now or datetime.utcnow(),
                )
            )
            db.flush()
    except IntegrityError:
        return False
    refresh_profile(db, pipeline, mode_stage, feature_bucket, now=now)
    return True


def refresh_profile(
    db: Session,
    pipeline: str,
    mode_stage: str,
    feature_bucket: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[TaskEtaProfile]:
    observations = (
        db.query(TaskEtaObservation)
        .filter(
            TaskEtaObservation.pipeline == pipeline,
            TaskEtaObservation.mode_stage == mode_stage,
            TaskEtaObservation.feature_bucket == feature_bucket,
            TaskEtaObservation.success.is_(True),
        )
        .order_by(TaskEtaObservation.created_at.desc())
        .limit(max(1, settings.eta_profile_max_observations))
        .all()
    )
    rates = [
        row.active_duration_seconds / row.units
        for row in observations
        if row.units and row.active_duration_seconds > 0
    ]
    if not rates:
        return None
    bounded = winsorize(rates)
    p50 = max(0.001, percentile(bounded, 0.50))
    p90 = max(p50, percentile(bounded, 0.90))
    profile = get_profile(
        db,
        pipeline,
        mode_stage,
        feature_bucket,
        require_calibrated=False,
    )
    if profile is None:
        profile = TaskEtaProfile(
            pipeline=pipeline,
            mode_stage=mode_stage,
            feature_bucket=feature_bucket,
            rate_p50=p50,
            rate_p90=p90,
            sample_count=len(rates),
            refreshed_at=now or datetime.utcnow(),
        )
        db.add(profile)
    else:
        profile.rate_p50 = p50
        profile.rate_p90 = p90
        profile.sample_count = len(rates)
        profile.refreshed_at = now or datetime.utcnow()
    db.flush()
    return profile
