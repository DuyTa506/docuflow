"""
Spatial Filters Module

Provides preprocessing filters to clean OCR output before tree building:
- Repetition filter: Remove headers/footers that repeat across pages
- Noise filter: Remove small artifacts and margin elements
- Zone detection: Identify page zones (top/bottom margins)
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class RepetitionInfo:
    """Information about a repeated element across pages."""
    text: str
    page_numbers: List[int] = field(default_factory=list)
    avg_y_position: float = 0.0
    avg_x_position: float = 0.0
    zone: str = "unknown"  # 'header', 'footer', or 'unknown'
    count: int = 0


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for repetition matching.
    Removes page numbers, whitespace, and common variations.
    
    Args:
        text: Raw text content
    
    Returns:
        Normalized text for comparison
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    normalized = text.strip().lower()
    
    # Remove page number patterns
    normalized = re.sub(r'\b(page\s*)?\d+\b', '', normalized)
    normalized = re.sub(r'^\d+\s*$', '', normalized)
    
    # Remove common header/footer patterns that may vary
    normalized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', normalized)  # Dates
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse whitespace
    
    return normalized.strip()


def analyze_cross_page_repetitions(
    elements: List[Dict],
    min_pages: int = 3,
    position_tolerance: float = 0.05,
    text_min_length: int = 3
) -> Dict[str, RepetitionInfo]:
    """
    Detect text that repeats across multiple pages (headers/footers).
    
    Uses (normalized_text, relative_y_position) as matching key to identify
    elements that appear in the same position across pages.
    
    Args:
        elements: List of layout elements with bbox, text, page_number
        min_pages: Minimum pages an element must appear on to be considered repeated
        position_tolerance: Tolerance for y-position matching (as fraction of page height)
        text_min_length: Minimum text length to consider (filter out noise)
    
    Returns:
        Dict mapping normalized_text to RepetitionInfo
    """
    # Group elements by normalized text
    text_occurrences: Dict[str, List[Dict]] = defaultdict(list)
    
    for elem in elements:
        text = elem.get('text_content', elem.get('text', ''))
        
        # Skip short or empty text
        if len(text.strip()) < text_min_length:
            continue
        
        normalized = normalize_text_for_matching(text)
        if normalized:
            text_occurrences[normalized].append(elem)
    
    # Analyze repetitions
    repetitions: Dict[str, RepetitionInfo] = {}
    
    for normalized_text, occurrences in text_occurrences.items():
        # Get unique pages
        pages = set()
        y_positions = []
        x_positions = []
        
        for elem in occurrences:
            page_num = elem.get('page_number', elem.get('page', 1))
            pages.add(page_num)
            
            # Get y position (top of bbox)
            y1 = elem.get('bbox_y1', elem.get('y1', 0))
            x1 = elem.get('bbox_x1', elem.get('x1', 0))
            y_positions.append(y1)
            x_positions.append(x1)
        
        # Check if repeats on enough pages
        if len(pages) >= min_pages:
            avg_y = sum(y_positions) / len(y_positions) if y_positions else 0
            avg_x = sum(x_positions) / len(x_positions) if x_positions else 0
            
            # Determine zone based on average y position
            # Assuming page height normalized or actual pixel values
            zone = "unknown"
            if y_positions:
                # Get page height from first occurrence
                first_elem = occurrences[0]
                page_height = first_elem.get('page_height', 1000)
                relative_y = avg_y / page_height if page_height > 0 else 0.5
                
                if relative_y < 0.15:
                    zone = "header"
                elif relative_y > 0.85:
                    zone = "footer"
            
            repetitions[normalized_text] = RepetitionInfo(
                text=occurrences[0].get('text_content', occurrences[0].get('text', '')),
                page_numbers=sorted(list(pages)),
                avg_y_position=avg_y,
                avg_x_position=avg_x,
                zone=zone,
                count=len(pages)
            )
    
    return repetitions


def filter_repeated_elements(
    elements: List[Dict],
    repetitions: Optional[Dict[str, RepetitionInfo]] = None,
    min_pages: int = 3,
    filter_zones: Optional[Set[str]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove elements that are identified as headers/footers.
    
    Args:
        elements: List of layout elements
        repetitions: Pre-computed repetitions (or None to compute)
        min_pages: Minimum pages for repetition detection (if computing)
        filter_zones: Zones to filter out (default: {'header', 'footer'})
    
    Returns:
        Tuple of (filtered_elements, removed_elements)
    """
    if filter_zones is None:
        filter_zones = {'header', 'footer'}
    
    # Compute repetitions if not provided
    if repetitions is None:
        repetitions = analyze_cross_page_repetitions(elements, min_pages=min_pages)
    
    # Build set of normalized texts to filter
    texts_to_filter: Set[str] = set()
    for normalized_text, info in repetitions.items():
        if info.zone in filter_zones:
            texts_to_filter.add(normalized_text)
    
    # Filter elements
    filtered = []
    removed = []
    
    for elem in elements:
        text = elem.get('text_content', elem.get('text', ''))
        normalized = normalize_text_for_matching(text)
        
        if normalized in texts_to_filter:
            elem['filter_reason'] = 'repeated_header_footer'
            removed.append(elem)
        else:
            filtered.append(elem)
    
    return filtered, removed


def filter_noise_elements(
    elements: List[Dict],
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.5,
    page_dims: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove elements that are too small (artifacts) or too large (backgrounds).
    
    Args:
        elements: List of layout elements with bbox
        min_area_ratio: Minimum area as fraction of page (filter smaller)
        max_area_ratio: Maximum area as fraction of page (filter larger)
        page_dims: Page dimensions {'width': int, 'height': int}
    
    Returns:
        Tuple of (filtered_elements, removed_elements)
    """
    filtered = []
    removed = []
    
    # Estimate page dims if not provided
    if page_dims is None:
        page_dims = estimate_page_dims(elements)
    
    page_area = page_dims.get('width', 800) * page_dims.get('height', 1000)
    
    for elem in elements:
        # Calculate element area
        x1 = elem.get('bbox_x1', elem.get('x1', 0))
        y1 = elem.get('bbox_y1', elem.get('y1', 0))
        x2 = elem.get('bbox_x2', elem.get('x2', 0))
        y2 = elem.get('bbox_y2', elem.get('y2', 0))
        
        elem_area = abs(x2 - x1) * abs(y2 - y1)
        area_ratio = elem_area / page_area if page_area > 0 else 0
        
        if area_ratio < min_area_ratio:
            elem['filter_reason'] = 'too_small'
            removed.append(elem)
        elif area_ratio > max_area_ratio:
            elem['filter_reason'] = 'too_large'
            removed.append(elem)
        else:
            filtered.append(elem)
    
    return filtered, removed


def filter_margin_elements(
    elements: List[Dict],
    margin_ratio: float = 0.05,
    page_dims: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove elements in extreme margins (likely page numbers, artifacts).
    
    Args:
        elements: List of layout elements
        margin_ratio: Margin size as fraction of page dimension
        page_dims: Page dimensions
    
    Returns:
        Tuple of (filtered_elements, removed_elements)
    """
    filtered = []
    removed = []
    
    if page_dims is None:
        page_dims = estimate_page_dims(elements)
    
    page_width = page_dims.get('width', 800)
    page_height = page_dims.get('height', 1000)
    
    left_margin = page_width * margin_ratio
    right_margin = page_width * (1 - margin_ratio)
    top_margin = page_height * margin_ratio
    bottom_margin = page_height * (1 - margin_ratio)
    
    for elem in elements:
        x1 = elem.get('bbox_x1', elem.get('x1', 0))
        y1 = elem.get('bbox_y1', elem.get('y1', 0))
        x2 = elem.get('bbox_x2', elem.get('x2', 0))
        y2 = elem.get('bbox_y2', elem.get('y2', 0))
        
        # Check if element is entirely in margin
        in_left_margin = x2 < left_margin
        in_right_margin = x1 > right_margin
        in_top_margin = y2 < top_margin
        in_bottom_margin = y1 > bottom_margin
        
        # Only filter if in corner or extreme side margins
        # Keep top/bottom for now (handled by repetition filter)
        if in_left_margin or in_right_margin:
            # Check if it's a small element (page number, margin note)
            elem_width = x2 - x1
            elem_height = y2 - y1
            
            if elem_width < page_width * 0.15 and elem_height < page_height * 0.1:
                elem['filter_reason'] = 'margin_element'
                removed.append(elem)
                continue
        
        filtered.append(elem)
    
    return filtered, removed


def estimate_page_dims(elements: List[Dict]) -> Dict[str, int]:
    """
    Estimate page dimensions from element bounding boxes.
    
    Args:
        elements: List of layout elements
    
    Returns:
        Dict with 'width' and 'height'
    """
    if not elements:
        return {'width': 800, 'height': 1000}
    
    max_x = 0
    max_y = 0
    
    for elem in elements:
        x2 = elem.get('bbox_x2', elem.get('x2', 0))
        y2 = elem.get('bbox_y2', elem.get('y2', 0))
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    
    # Add small margin
    return {
        'width': int(max_x * 1.05) if max_x > 0 else 800,
        'height': int(max_y * 1.05) if max_y > 0 else 1000
    }


def apply_all_filters(
    elements: List[Dict],
    filter_repeated: bool = True,
    filter_noise: bool = True,
    filter_margins: bool = False,
    min_repeat_pages: int = 3,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.5
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Apply all preprocessing filters in sequence.
    
    Args:
        elements: List of layout elements
        filter_repeated: Whether to filter repeated headers/footers
        filter_noise: Whether to filter noise elements
        filter_margins: Whether to filter margin elements
        min_repeat_pages: Minimum pages for repetition detection
        min_area_ratio: Minimum area ratio for noise filter
        max_area_ratio: Maximum area ratio for noise filter
    
    Returns:
        Tuple of (filtered_elements, removed_by_filter)
        where removed_by_filter is a dict mapping filter name to removed elements
    """
    current = elements
    removed_by_filter: Dict[str, List[Dict]] = {}
    
    # Filter 1: Repeated elements (headers/footers)
    if filter_repeated:
        current, removed = filter_repeated_elements(
            current, 
            min_pages=min_repeat_pages
        )
        removed_by_filter['repeated'] = removed
    
    # Filter 2: Noise elements
    if filter_noise:
        current, removed = filter_noise_elements(
            current,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio
        )
        removed_by_filter['noise'] = removed
    
    # Filter 3: Margin elements
    if filter_margins:
        current, removed = filter_margin_elements(current)
        removed_by_filter['margin'] = removed
    
    return current, removed_by_filter
