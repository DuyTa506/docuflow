"""Core package - Domain models and constants."""

from .constants import (
    DEFAULT_OCR_PARAMS,
    DEFAULT_SPATIAL_WEIGHTS,
    GROUNDING_PATTERN,
    HIERARCHY_THRESHOLDS,
    LABEL_HIERARCHY_WEIGHTS,
    OCR_PROMPTS,
)
from .models import BoundingBox, LayoutElement, ServicePageResult

__all__ = [
    "ServicePageResult",
    "LayoutElement",
    "BoundingBox",
    "LABEL_HIERARCHY_WEIGHTS",
    "DEFAULT_SPATIAL_WEIGHTS",
    "HIERARCHY_THRESHOLDS",
    "OCR_PROMPTS",
    "DEFAULT_OCR_PARAMS",
    "GROUNDING_PATTERN",
]
