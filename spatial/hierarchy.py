"""
Spatial Hierarchy Analyzer

Uses bounding box coordinates and grounding labels to predict document hierarchy.
Provides heuristic scoring for element importance and relationships.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.constants import LABEL_HIERARCHY_WEIGHTS


# Use imported weights from core.constants
# LABEL_HIERARCHY_WEIGHTS is now imported from core.constants


def vertical_hierarchy_score(element: Dict, page_height: int) -> float:
    """
    Score based on vertical position (0-1, higher = more important).
    Elements at top of page score higher.
    
    Args:
        element: Layout element with bbox_y1
        page_height: Total page height
    
    Returns:
        Score from 0 to 1 (1.0 = top of page)
    """
    if page_height == 0:
        return 0.5
    
    y_position = element.get('bbox_y1', element.get('y1', 0))
    normalized_y = y_position / page_height
    
    # Top of page = high score
    return 1.0 - normalized_y


def size_importance_score(element: Dict, page_width: int, page_height: int) -> float:
    """
    Score based on bounding box size.
    Larger elements tend to be more important (titles, headings).
    
    Args:
        element: Layout element with bbox coordinates
        page_width: Total page width
        page_height: Total page height
    
    Returns:
        Score from 0 to 1
    """
    if page_width == 0 or page_height == 0:
        return 0.3
    
    # Get bbox coordinates
    x1 = element.get('bbox_x1', element.get('x1', 0))
    y1 = element.get('bbox_y1', element.get('y1', 0))
    x2 = element.get('bbox_x2', element.get('x2', page_width))
    y2 = element.get('bbox_y2', element.get('y2', page_height))
    
    width = x2 - x1
    height = y2 - y1
    
    # Calculate relative size
    width_ratio = width / page_width
    height_ratio = height / page_height
    
    # Combine width and height (width is more important for titles)
    size_score = (width_ratio * 0.7 + height_ratio * 0.3)
    
    # Scale up (titles are typically 30-100% of page width)
    return min(1.0, size_score * 2.0)


def label_hierarchy_weight(label: str) -> float:
    """
    Get hierarchy weight from grounding label.
    
    Args:
        label: Grounding label (e.g., 'title', 'text', 'table')
    
    Returns:
        Weight from 0 to 1 (1.0 = highest importance)
    """
    label_lower = label.lower().strip()
    return LABEL_HIERARCHY_WEIGHTS.get(label_lower, 0.3)


def indentation_score(element: Dict, page_width: int) -> float:
    """
    Score based on left margin indentation.
    Left-aligned elements tend to be higher in hierarchy.
    
    Args:
        element: Layout element with bbox_x1
        page_width: Total page width
    
    Returns:
        Score from 0 to 1 (1.0 = left-aligned)
    """
    if page_width == 0:
        return 0.5
    
    x1 = element.get('bbox_x1', element.get('x1', 0))
    
    # Max indent we consider (30% of page width)
    max_indent = page_width * 0.3
    
    if x1 > max_indent:
        return 0.0  # Too indented
    
    # Linear scale: left edge = 1.0, max_indent = 0.0
    return 1.0 - (x1 / max_indent)


def spatial_proximity_score(elem1: Dict, elem2: Dict, threshold: int = 100) -> float:
    """
    Score how likely two elements belong together.
    Based on vertical distance between elements.
    
    Args:
        elem1: First element
        elem2: Second element
        threshold: Max distance to consider (pixels)
    
    Returns:
        Score from 0 to 1 (1.0 = very close)
    """
    y1_bottom = elem1.get('bbox_y2', elem1.get('y2', 0))
    y2_top = elem2.get('bbox_y1', elem2.get('y1', 0))
    
    vertical_distance = abs(y2_top - y1_bottom)
    
    if vertical_distance > threshold:
        return 0.0
    
    return 1.0 - (vertical_distance / threshold)


def whitespace_isolation_score(
    element: Dict,
    prev_element: Optional[Dict],
    next_element: Optional[Dict],
    median_line_height: float = 20.0
) -> float:
    """
    Score based on white-space isolation (before and after element).
    
    Headings typically have more white-space before/after them
    compared to regular body text.
    
    Args:
        element: Current layout element
        prev_element: Previous element in reading order (or None)
        next_element: Next element in reading order (or None)
        median_line_height: Median line height for normalization
    
    Returns:
        Score from 0 to 1 (1.0 = highly isolated with whitespace)
    """
    y1 = element.get('bbox_y1', element.get('y1', 0))
    y2 = element.get('bbox_y2', element.get('y2', 0))
    
    # Calculate spacing before
    space_before = 0.0
    if prev_element:
        prev_y2 = prev_element.get('bbox_y2', prev_element.get('y2', 0))
        space_before = max(0, y1 - prev_y2)
    else:
        # First element - give moderate score
        space_before = median_line_height * 1.5
    
    # Calculate spacing after
    space_after = 0.0
    if next_element:
        next_y1 = next_element.get('bbox_y1', next_element.get('y1', 0))
        space_after = max(0, next_y1 - y2)
    else:
        # Last element - give moderate score
        space_after = median_line_height * 1.0
    
    # Normalize by median line height
    if median_line_height <= 0:
        median_line_height = 20.0
    
    before_ratio = space_before / median_line_height
    after_ratio = space_after / median_line_height
    
    # Combined score (weight before more than after)
    # Headings typically have MORE space before (section break)
    combined_ratio = (before_ratio * 0.6 + after_ratio * 0.4)
    
    # Normalize to 0-1 range
    # 2x median spacing = 1.0 score
    score = min(1.0, combined_ratio / 2.0)
    
    return score


def calculate_adaptive_thresholds(
    elements: List[Dict],
    weights: Optional[Dict[str, float]] = None
) -> Dict[int, float]:
    """
    Calculate adaptive thresholds based on document score distribution.
    
    Instead of using fixed thresholds, this uses percentile-based
    thresholds that adapt to the specific document.
    
    Args:
        elements: List of elements with computed scores
        weights: Scoring weights to use
    
    Returns:
        Dict mapping hierarchy level (0-5) to score threshold
    """
    if not elements:
        # Return default thresholds
        return {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.0}
    
    # Collect all spatial scores
    scores = []
    for elem in elements:
        score = elem.get('spatial_score', 0.5)
        scores.append(score)
    
    if not scores:
        return {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.0}
    
    # Use percentiles for adaptive thresholds
    try:
        return {
            0: float(np.percentile(scores, 95)),  # Top 5% → title
            1: float(np.percentile(scores, 80)),  # Top 20% → major section
            2: float(np.percentile(scores, 60)),  # Top 40% → subsection
            3: float(np.percentile(scores, 40)),  # Middle → subsubsection
            4: float(np.percentile(scores, 20)),  # Lower → paragraph
            5: 0.0                                 # Bottom → supporting
        }
    except Exception:
        # Fallback to defaults
        return {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.0}


def predict_hierarchy_level(
    element: Dict,
    page_width: int,
    page_height: int,
    weights: Optional[Dict[str, float]] = None,
    prev_element: Optional[Dict] = None,
    next_element: Optional[Dict] = None,
    median_line_height: Optional[float] = None,
    thresholds: Optional[Dict[int, float]] = None
) -> int:
    """
    Predict hierarchy level (0=highest, 5=lowest) using spatial metadata.
    
    UPDATED: Now includes whitespace isolation scoring.
    
    Args:
        element: Layout element with bbox and label
        page_width: Page width for normalization
        page_height: Page height for normalization
        weights: Optional custom weights for combining scores
        prev_element: Previous element (for whitespace calculation)
        next_element: Next element (for whitespace calculation)
        median_line_height: Median line height (for whitespace normalization)
        thresholds: Optional custom thresholds (for adaptive calibration)
    
    Returns:
        int: Hierarchy level (0-5)
            0 = Document title / chapter
            1 = Major section
            2 = Subsection
            3 = Subsubsection
            4 = Paragraph
            5 = Caption/footer
    """
    # Import updated weights from constants
    from core.constants import DEFAULT_SPATIAL_WEIGHTS
    
    # Use weights from constants if not provided
    if weights is None:
        weights = DEFAULT_SPATIAL_WEIGHTS.copy()
    
    # Calculate individual scores
    vertical_score = vertical_hierarchy_score(element, page_height)
    size_score = size_importance_score(element, page_width, page_height)
    label_weight = label_hierarchy_weight(element.get('label', 'text'))
    indent_score = indentation_score(element, page_width)
    
    # Calculate whitespace score (NEW)
    if median_line_height is None:
        # Estimate from element height
        y1 = element.get('bbox_y1', element.get('y1', 0))
        y2 = element.get('bbox_y2', element.get('y2', 0))
        median_line_height = max(20.0, y2 - y1)
    
    whitespace_score = whitespace_isolation_score(
        element, prev_element, next_element, median_line_height
    )
    
    # Weighted combination (UPDATED formula)
    combined_score = (
        label_weight * weights.get('label', 0.40) +
        whitespace_score * weights.get('whitespace', 0.25) +
        size_score * weights.get('size', 0.15) +
        vertical_score * weights.get('vertical', 0.10) +
        indent_score * weights.get('indent', 0.10)
    )
    
    # Use custom thresholds if provided, otherwise use defaults
    if thresholds is None:
        thresholds = {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.0}
    
    # Convert to hierarchy level (0-5)
    if combined_score > thresholds.get(0, 0.8):
        return 0  # Top level (document title)
    elif combined_score > thresholds.get(1, 0.6):
        return 1  # Major section
    elif combined_score > thresholds.get(2, 0.4):
        return 2  # Subsection
    elif combined_score > thresholds.get(3, 0.25):
        return 3  # Subsubsection
    elif combined_score > thresholds.get(4, 0.15):
        return 4  # Paragraph
    else:
        return 5  # Supporting elements (caption, footer)


def classify_elements_with_metadata(
    layout_elements: List[Dict],
    page_dims: Dict[str, int],
    weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Classify each element's hierarchy using spatial metadata.
    
    Args:
        layout_elements: List of layout elements with bbox and labels
        page_dims: Page dimensions {'width': int, 'height': int}
        weights: Optional custom weights
    
    Returns:
        List of elements with added 'predicted_level' and 'spatial_score'
    """
    classified = []
    
    page_width = page_dims.get('width', 800)
    page_height = page_dims.get('height', 1000)
    
    for elem in layout_elements:
        hierarchy_level = predict_hierarchy_level(
            elem, page_width, page_height, weights
        )
        
        # Calculate combined score for reference
        vertical_score = vertical_hierarchy_score(elem, page_height)
        size_score = size_importance_score(elem, page_width, page_height)
        label_weight = label_hierarchy_weight(elem.get('label', 'text'))
        indent_score = indentation_score(elem, page_width)
        
        w = weights or {'vertical': 0.2, 'size': 0.3, 'label': 0.4, 'indent': 0.1}
        spatial_score = (
            vertical_score * w['vertical'] +
            size_score * w['size'] +
            label_weight * w['label'] +
            indent_score * w['indent']
        )
        
        classified.append({
            **elem,
            'predicted_level': hierarchy_level,
            'spatial_score': spatial_score,
            'component_scores': {
                'vertical': vertical_score,
                'size': size_score,
                'label': label_weight,
                'indent': indent_score
            }
        })
    
    return classified


def cluster_by_spatial_proximity(
    elements: List[Dict],
    proximity_threshold: int = 100
) -> List[List[Dict]]:
    """
    Group elements into clusters based on spatial proximity.
    
    Args:
        elements: List of elements sorted by vertical position
        proximity_threshold: Max vertical distance to group together
    
    Returns:
        List of element clusters
    """
    if not elements:
        return []
    
    clusters = []
    current_cluster = [elements[0]]
    
    for i in range(1, len(elements)):
        prev_elem = elements[i-1]
        curr_elem = elements[i]
        
        proximity = spatial_proximity_score(prev_elem, curr_elem, proximity_threshold)
        
        if proximity > 0.3:  # Close enough to be in same cluster
            current_cluster.append(curr_elem)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [curr_elem]
    
    # Add last cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def get_page_dimensions_from_elements(elements: List[Dict]) -> Dict[str, int]:
    """
    Estimate page dimensions from bounding boxes.
    
    Args:
        elements: List of layout elements
    
    Returns:
        Dict with 'width' and 'height'
    """
    if not elements:
        return {'width': 800, 'height': 1000}
    
    max_x = max(elem.get('bbox_x2', elem.get('x2', 0)) for elem in elements)
    max_y = max(elem.get('bbox_y2', elem.get('y2', 0)) for elem in elements)
    
    # Add some padding
    return {
        'width': int(max_x * 1.1),
        'height': int(max_y * 1.1)
    }
