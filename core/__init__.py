"""Core package - Domain models and constants."""

from .models import ServicePageResult, LayoutElement, BoundingBox
from .constants import (
    LABEL_HIERARCHY_WEIGHTS,
    DEFAULT_SPATIAL_WEIGHTS,
    HIERARCHY_THRESHOLDS,
    OCR_PROMPTS,
    DEFAULT_OCR_PARAMS,
    GROUNDING_PATTERN
)

__all__ = [
    'ServicePageResult',
    'LayoutElement',
    'BoundingBox',
    'LABEL_HIERARCHY_WEIGHTS',
    'DEFAULT_SPATIAL_WEIGHTS',
    'HIERARCHY_THRESHOLDS',
    'OCR_PROMPTS',
    'DEFAULT_OCR_PARAMS',
    'GROUNDING_PATTERN'
]
