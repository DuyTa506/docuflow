"""Spatial analysis package - Hierarchy detection using spatial metadata."""

from .filters import (
    analyze_cross_page_repetitions,
    apply_all_filters,
    filter_noise_elements,
    filter_repeated_elements,
)
from .grouping import (
    detect_columns_projection,
    group_into_lines,
    group_lines_to_blocks,
    process_page_layout,
)
from .hierarchy import (
    calculate_adaptive_thresholds,
    classify_elements_with_metadata,
    get_page_dimensions_from_elements,
    indentation_score,
    label_hierarchy_weight,
    predict_hierarchy_level,
    size_importance_score,
    spatial_proximity_score,
    vertical_hierarchy_score,
    whitespace_isolation_score,
)
from .reading_order import (
    build_reading_order_graph,
    get_reading_order,
    get_reading_order_by_page,
)
from .spatial_tree_builder import build_spatial_tree  # NEW - Recommended API
from .spatial_tree_builder import (
    extract_markdown_level,
    predict_hierarchy_spatial,
    validate_with_markdown_syntax,
)
from .thinning import hierarchical_thinning  # NEW
from .thinning import (
    can_merge_text_blocks,
    merge_nodes_content,
)
from .tree_builder import calculate_bbox_iou  # Still useful for other purposes
from .tree_builder import (  # Kept for backward compatibility / utility
    TreeNode,
    parse_markdown_headers,
)
from .zone_classifier import (
    ZoneType,
    classify_zone_heuristic,
    classify_zones_batch,
    get_zone_priority,
)

__all__ = [
    # Hierarchy functions
    "vertical_hierarchy_score",
    "size_importance_score",
    "label_hierarchy_weight",
    "indentation_score",
    "spatial_proximity_score",
    "whitespace_isolation_score",
    "calculate_adaptive_thresholds",
    "predict_hierarchy_level",
    "classify_elements_with_metadata",
    "get_page_dimensions_from_elements",
    # Tree building - NEW SPATIAL-FIRST API
    "build_spatial_tree",  # Recommended
    "predict_hierarchy_spatial",
    "validate_with_markdown_syntax",
    "extract_markdown_level",
    "TreeNode",
    "parse_markdown_headers",
    "calculate_bbox_iou",
    # Filters
    "analyze_cross_page_repetitions",
    "filter_repeated_elements",
    "filter_noise_elements",
    "apply_all_filters",
    # Zone classification
    "ZoneType",
    "classify_zone_heuristic",
    "classify_zones_batch",
    "get_zone_priority",
    # Reading order
    "get_reading_order",
    "get_reading_order_by_page",
    "build_reading_order_graph",
    # Grouping
    "detect_columns_projection",
    "group_into_lines",
    "group_lines_to_blocks",
    "process_page_layout",
]
