"""Spatial analysis package - Hierarchy detection using spatial metadata."""

from .hierarchy import (
    vertical_hierarchy_score,
    size_importance_score,
    label_hierarchy_weight,
    indentation_score,
    predict_hierarchy_level,
    classify_elements_with_metadata
)

from .tree_builder import (
    build_enhanced_tree,
    TreeNode
)

__all__ = [
    # Hierarchy functions
    'vertical_hierarchy_score',
    'size_importance_score',
    'label_hierarchy_weight',
    'indentation_score',
    'predict_hierarchy_level',
    'classify_elements_with_metadata',
    
    # Tree building
    'build_enhanced_tree',
    'TreeNode'
]
