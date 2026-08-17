"""Digest DAG profile dimensions and parallel remaining-time semantics."""

from __future__ import annotations

from typing import Callable, Optional

from services.eta.ocr import size_bucket

GROUP_A = ("BIBLIOGRAPHIC", "KEYWORDS", "RESEARCH_DIRECTIONS", "USAGE_SCOPE")
GROUP_B = ("HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT")


def dimensions(meta: dict) -> tuple[str, str, str]:
    stage = str(meta.get("stage") or "UNKNOWN")
    phase = str(meta.get("phase") or "active")
    mode_stage = f"{stage}:{phase}" if phase in {"exporting", "finalizing"} else stage
    return (
        "digest",
        mode_stage,
        str(meta.get("feature_bucket") or size_bucket(meta.get("units_total"))),
    )


def remaining_profile_seconds(
    stages: dict,
    lookup_duration: Callable[[str], Optional[tuple[float, float]]],
    *,
    current_stage: str | None,
) -> Optional[tuple[float, float]]:
    """Return remaining calibrated DAG duration.

    Sequential boundaries are summed. Durations within each actual Temporal
    parallel group use ``max`` rather than the weighted-progress sum.
    """

    def pending(stage: str) -> bool:
        return int((stages.get(stage) or {}).get("progress") or 0) < 100 and stage != current_stage

    def one(stage: str) -> Optional[tuple[float, float]]:
        if not pending(stage):
            return (0.0, 0.0)
        found = lookup_duration(stage)
        # Missing calibrated stage: treat as unknown-zero rather than failing
        # the whole DAG estimate (current-stage ETA still useful).
        return (0.0, 0.0) if found is None else found

    parts: list[tuple[float, float]] = []
    parts.append(one("BUILD_TREE") or (0.0, 0.0))

    for group in (GROUP_A, GROUP_B):
        values = [one(stage) or (0.0, 0.0) for stage in group]
        parts.append(
            (
                max(value[0] for value in values),
                max(value[1] for value in values),
            )
        )

    parts.append(one("FINALIZE") or (0.0, 0.0))
    return sum(value[0] for value in parts), sum(value[1] for value in parts)
