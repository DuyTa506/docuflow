"""OCR/extraction-specific profile dimensions."""

from __future__ import annotations


def size_bucket(units_total: int | None) -> str:
    total = max(0, int(units_total or 0))
    if total <= 10:
        return "small"
    if total <= 100:
        return "medium"
    if total <= 500:
        return "large"
    return "xlarge"


def dimensions(meta: dict) -> tuple[str, str, str]:
    mode = str(meta.get("mode") or "unknown")
    phase = str(meta.get("phase") or "active")
    mode_stage = f"{mode}:{phase}" if phase in {"exporting", "finalizing"} else mode
    return (
        "extract",
        mode_stage,
        str(meta.get("feature_bucket") or size_bucket(meta.get("units_total"))),
    )
