"""
SpatialConfig — single source of truth for all spatial algorithm constants.

Import the singleton:
    from config.spatial_config import spatial_config

Or use the class directly for custom instances:
    from config.spatial_config import SpatialConfig
    cfg = SpatialConfig(label_weight=0.5)
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SpatialConfig:
    """
    All magic numbers for the spatial analysis pipeline in one place.

    Hierarchy scoring weights
    ─────────────────────────
    Must sum to 1.0.  Defaults match DEFAULT_SPATIAL_WEIGHTS in core/constants.py.
    """

    # ── Hierarchy scoring weights ────────────────────────────────────
    label_weight: float = 0.40  # OCR label type (strongest signal)
    whitespace_weight: float = 0.25  # White-space isolation
    size_weight: float = 0.15  # Element size
    vertical_weight: float = 0.10  # Vertical position on page
    indent_weight: float = 0.10  # Left-margin indentation

    # ── Fixed hierarchy level thresholds ────────────────────────────
    # Used when adaptive thresholds are disabled or as fallback.
    hierarchy_thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.80,  # Document title / chapter
            1: 0.60,  # Major section
            2: 0.40,  # Subsection
            3: 0.25,  # Subsubsection
            4: 0.15,  # Paragraph
            5: 0.0,  # Supporting elements (captions, footers)
        }
    )

    # ── Noise / area filters (spatial/filters.py) ────────────────────
    min_element_area_ratio: float = 0.001  # 0.1% of page — filter smaller
    max_element_area_ratio: float = 0.50  # 50% of page — filter larger
    footer_height_ratio: float = 0.15  # bottom 15% of page → footer zone
    header_height_ratio: float = 0.10  # top 10% of page → header zone
    min_repeat_pages: int = 3  # must appear on ≥N pages to be header/footer

    # ── Reading order (spatial/reading_order.py) ─────────────────────
    same_row_threshold: float = 0.30  # vertical overlap → same row
    same_column_threshold: float = 0.30  # horizontal overlap → same column

    # ── Thinning (spatial/thinning.py) ───────────────────────────────
    gap_threshold_percentile: int = 70  # percentile of gaps → merge threshold
    barrier_labels: frozenset = field(
        default_factory=lambda: frozenset(
            {
                "title",
                "sub_title",
                "subtitle",
                "heading",
                "equation",
                "formula",
                "image",
                "figure",
                "table",
                "tablecaption",
                "tablefootnote",
                "imagecaption",
                "caption",
                "section_heading",
                "title_block",
                "abstract",
            }
        )
    )

    # ── Zone classifier confidence scores (spatial/zone_classifier.py)
    label_confidence: float = 0.80  # label-based classification
    position_confidence: float = 0.85  # position-based (page_number, footer…)
    pattern_confidence: float = 0.75  # text-pattern-based

    def as_weights_dict(self) -> Dict[str, float]:
        """Return scoring weights in the format expected by spatial/hierarchy.py."""
        return {
            "label": self.label_weight,
            "whitespace": self.whitespace_weight,
            "size": self.size_weight,
            "vertical": self.vertical_weight,
            "indent": self.indent_weight,
        }


# ── Module-level singleton ───────────────────────────────────────────

spatial_config = SpatialConfig()
