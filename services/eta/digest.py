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
        return lookup_duration(stage) if pending(stage) else (0.0, 0.0)

    parts: list[tuple[float, float]] = []
    build = one("BUILD_TREE")
    if build is None:
        return None
    parts.append(build)

    for group in (GROUP_A, GROUP_B):
        values = [one(stage) for stage in group]
        if any(value is None for value in values):
            return None
        parts.append(
            (
                max((value or (0.0, 0.0))[0] for value in values),
                max((value or (0.0, 0.0))[1] for value in values),
            )
        )

    finalize = one("FINALIZE")
    if finalize is None:
        return None
    parts.append(finalize)
    return sum(value[0] for value in parts), sum(value[1] for value in parts)
