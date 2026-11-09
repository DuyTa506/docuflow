"""Translation-specific mode/language/work-size profile dimensions."""

from __future__ import annotations

from services.eta.ocr import size_bucket


def dimensions(meta: dict) -> tuple[str, str, str]:
    mode = str(meta.get("mode") or "unknown")
    phase = str(meta.get("phase") or "active")
    mode_stage = f"{mode}:{phase}" if phase in {"exporting", "finalizing"} else mode
    language = str(meta.get("target_language") or "unknown").lower()
    bucket = str(meta.get("feature_bucket") or size_bucket(meta.get("units_total")))
    return "translate", mode_stage, f"lang={language};size={bucket}"
