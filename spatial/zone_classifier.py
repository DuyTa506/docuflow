"""
Zone Classifier Module

Classifies layout elements into functional zones using heuristic rules.
Zones include: title_block, main_text, figure, table, caption, equation,
header, footer, footnote, etc.

This is a heuristic-only implementation. Can be extended with LLM fallback.
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ZoneType(Enum):
    """Enumeration of zone types for document layout."""
    TITLE_BLOCK = "title_block"
    AUTHOR_BLOCK = "author_block"
    ABSTRACT = "abstract"
    SECTION_HEADING = "section_heading"
    MAIN_TEXT = "main_text"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    EQUATION = "equation"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


# Zone priority for reading order
ZONE_PRIORITY = {
    ZoneType.TITLE_BLOCK: 0,
    ZoneType.AUTHOR_BLOCK: 1,
    ZoneType.ABSTRACT: 2,
    ZoneType.SECTION_HEADING: 3,
    ZoneType.MAIN_TEXT: 4,
    ZoneType.FIGURE: 5,
    ZoneType.TABLE: 5,
    ZoneType.CAPTION: 6,
    ZoneType.EQUATION: 4,
    ZoneType.FOOTNOTE: 8,
    ZoneType.HEADER: 9,
    ZoneType.FOOTER: 10,
    ZoneType.PAGE_NUMBER: 10,
    ZoneType.SIDEBAR: 7,
    ZoneType.UNKNOWN: 5,
}


# Caption patterns (regex) - Updated to handle DeepSeek HTML-style output
# Example: <center>Figure 6. Picture of DJI mini2. </center>
CAPTION_PATTERNS = [
    # Standard patterns
    r'^(Figure|Fig\.?)\s*\d+',
    r'^(Table|Tab\.?)\s*\d+',
    r'^(Hình)\s*\d+',  # Vietnamese
    r'^(Bảng)\s*\d+',   # Vietnamese
    r'^(Image|Img\.?)\s*\d+',
    r'^(Chart|Graph)\s*\d+',
    r'^\[\d+\]',  # Reference style
    
    # HTML-wrapped patterns (DeepSeek output)
    r'^<center>\s*(Figure|Fig\.?)\s*\d+',
    r'^<center>\s*(Table|Tab\.?)\s*\d+',
    r'^<center>\s*(Hình)\s*\d+',
    r'^<center>\s*(Bảng)\s*\d+',
    r'^<center>\s*(Image|Img\.?)\s*\d+',
]


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text for matching.
    
    Handles DeepSeek output like: <center>Figure 6. Description</center>
    
    Args:
        text: Text that may contain HTML tags
    
    Returns:
        Text with HTML tags stripped
    """
    if not text:
        return ""
    
    # Remove common HTML tags
    cleaned = re.sub(r'</?center>', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?b>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?i>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?strong>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?em>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<br\s*/?>', ' ', cleaned, flags=re.IGNORECASE)
    
    # Generic tag removal as fallback
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    return cleaned.strip()

# Section numbering patterns
SECTION_PATTERNS = [
    r'^\d+\.(\d+\.)*\s+\S',    # 1. Introduction, 1.2.3 Methods
    r'^[A-Z]+\.(\d+\.)*\s+\S', # A.1 Appendix
    r'^(Chapter|Section|Part)\s+\d+',
    r'^(Chương|Phần|Mục)\s+\d+',  # Vietnamese
]

# Page number patterns
PAGE_NUMBER_PATTERNS = [
    r'^\d{1,4}$',           # Standalone numbers
    r'^-\s*\d+\s*-$',       # - 5 -
    r'^page\s*\d+',         # Page 5
    r'^trang\s*\d+',        # Vietnamese: Trang 5
]


@dataclass
class ZoneClassification:
    """Result of zone classification for an element."""
    zone: ZoneType
    confidence: float  # 0.0 to 1.0
    method: str  # 'heuristic', 'llm', or 'fallback'
    features: Dict = None  # Features used for classification


def classify_by_label(
    element: Dict,
    label_to_zone: Optional[Dict[str, ZoneType]] = None
) -> Optional[ZoneClassification]:
    """
    Classify zone based on OCR grounding label.
    
    Args:
        element: Layout element with 'label' field
        label_to_zone: Optional custom mapping
    
    Returns:
        ZoneClassification or None if label not definitive
    """
    if label_to_zone is None:
        label_to_zone = {
            'title': ZoneType.TITLE_BLOCK,
            'sub_title': ZoneType.SECTION_HEADING,
            'subtitle': ZoneType.SECTION_HEADING,
            'heading': ZoneType.SECTION_HEADING,
            'header': ZoneType.HEADER,
            'figure': ZoneType.FIGURE,
            'table': ZoneType.TABLE,
            'equation': ZoneType.EQUATION,
            'formula': ZoneType.EQUATION,
            'caption': ZoneType.CAPTION,
            'footnote': ZoneType.FOOTNOTE,
            'footer': ZoneType.FOOTER,
            'page_number': ZoneType.PAGE_NUMBER,
        }
    
    label = element.get('label', '').lower().strip()
    
    if label in label_to_zone:
        return ZoneClassification(
            zone=label_to_zone[label],
            confidence=0.8,  # Label-based has good but not perfect confidence
            method='heuristic_label'
        )
    
    return None


def classify_by_position(
    element: Dict,
    page_dims: Dict[str, int]
) -> Optional[ZoneClassification]:
    """
    Classify zone based on bbox position on page.
    
    Args:
        element: Layout element with bbox
        page_dims: Page dimensions {'width': int, 'height': int}
    
    Returns:
        ZoneClassification or None if position not definitive
    """
    page_height = page_dims.get('height', 1000)
    page_width = page_dims.get('width', 800)
    
    y1 = element.get('bbox_y1', element.get('y1', 0))
    y2 = element.get('bbox_y2', element.get('y2', 0))
    x1 = element.get('bbox_x1', element.get('x1', 0))
    x2 = element.get('bbox_x2', element.get('x2', 0))
    
    # Relative positions
    rel_y1 = y1 / page_height if page_height > 0 else 0
    rel_y2 = y2 / page_height if page_height > 0 else 0
    rel_x1 = x1 / page_width if page_width > 0 else 0
    rel_x2 = x2 / page_width if page_width > 0 else 0
    
    # Element dimensions
    elem_width = x2 - x1
    elem_height = y2 - y1
    rel_width = elem_width / page_width if page_width > 0 else 0
    rel_height = elem_height / page_height if page_height > 0 else 0
    
    # Page number: bottom, centered, very small
    if (rel_y1 > 0.92 and 
        rel_height < 0.03 and 
        0.4 < (rel_x1 + rel_x2) / 2 < 0.6):
        return ZoneClassification(
            zone=ZoneType.PAGE_NUMBER,
            confidence=0.85,
            method='heuristic_position',
            features={'rel_y1': rel_y1, 'rel_height': rel_height}
        )
    
    # Footer zone: bottom of page
    if rel_y1 > 0.9 and rel_height < 0.08:
        return ZoneClassification(
            zone=ZoneType.FOOTER,
            confidence=0.7,
            method='heuristic_position',
            features={'rel_y1': rel_y1}
        )
    
    # Header zone: top of page
    if rel_y2 < 0.1 and rel_height < 0.08:
        return ZoneClassification(
            zone=ZoneType.HEADER,
            confidence=0.7,
            method='heuristic_position',
            features={'rel_y2': rel_y2}
        )
    
    # Footnote: bottom, smaller text (based on height)
    if rel_y1 > 0.85 and rel_height < 0.12:
        return ZoneClassification(
            zone=ZoneType.FOOTNOTE,
            confidence=0.6,
            method='heuristic_position',
            features={'rel_y1': rel_y1}
        )
    
    # Title block: very top, large width
    if rel_y1 < 0.15 and rel_width > 0.5:
        # Could be title, but wait for text pattern check
        pass
    
    return None


def classify_by_text_pattern(
    element: Dict
) -> Optional[ZoneClassification]:
    """
    Classify zone based on text content patterns.
    
    Args:
        element: Layout element with text content
    
    Returns:
        ZoneClassification or None if pattern not matched
    """
    raw_text = element.get('text_content', element.get('text', '')).strip()
    
    if not raw_text:
        return None
    
    # Strip HTML tags for matching (handles DeepSeek output)
    text = strip_html_tags(raw_text)
    
    # Caption patterns (Figure 1, Table 2, etc.)
    # Try both raw (with HTML) and stripped text
    for pattern in CAPTION_PATTERNS:
        if re.match(pattern, raw_text, re.IGNORECASE) or \
           re.match(pattern, text, re.IGNORECASE):
            return ZoneClassification(
                zone=ZoneType.CAPTION,
                confidence=0.9,
                method='heuristic_pattern',
                features={'pattern': pattern, 'html_stripped': raw_text != text}
            )
    
    # Page number patterns
    for pattern in PAGE_NUMBER_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return ZoneClassification(
                zone=ZoneType.PAGE_NUMBER,
                confidence=0.85,
                method='heuristic_pattern',
                features={'pattern': pattern}
            )
    
    # Section heading patterns (only if short text)
    if len(text) < 200:  # Headings usually short
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return ZoneClassification(
                    zone=ZoneType.SECTION_HEADING,
                    confidence=0.8,
                    method='heuristic_pattern',
                    features={'pattern': pattern}
                )
    
    # Abstract keyword
    if text.lower().startswith('abstract'):
        return ZoneClassification(
            zone=ZoneType.ABSTRACT,
            confidence=0.85,
            method='heuristic_pattern',
            features={'keyword': 'abstract'}
        )
    
    return None


def classify_by_geometry(
    element: Dict,
    page_dims: Dict[str, int]
) -> Optional[ZoneClassification]:
    """
    Classify zone based on element geometry (aspect ratio, size).
    
    Useful for detecting figures, tables, equations.
    
    Args:
        element: Layout element with bbox
        page_dims: Page dimensions
    
    Returns:
        ZoneClassification or None if geometry not definitive
    """
    x1 = element.get('bbox_x1', element.get('x1', 0))
    y1 = element.get('bbox_y1', element.get('y1', 0))
    x2 = element.get('bbox_x2', element.get('x2', page_dims.get('width', 0)))
    y2 = element.get('bbox_y2', element.get('y2', page_dims.get('height', 0)))
    
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0 or height <= 0:
        return None
    
    aspect_ratio = width / height
    page_width = page_dims.get('width', 800)
    page_height = page_dims.get('height', 1000)
    
    # Calculate relative size
    rel_width = width / page_width if page_width > 0 else 0
    rel_height = height / page_height if page_height > 0 else 0
    
    # Figure: usually square-ish, medium to large size
    label = element.get('label', '').lower()
    if label == 'figure':
        return ZoneClassification(
            zone=ZoneType.FIGURE,
            confidence=0.9,
            method='heuristic_label',
            features={'aspect_ratio': aspect_ratio}
        )
    
    # Equation: typically wide and short, centered
    if (0.7 < aspect_ratio and 
        rel_height < 0.1 and 
        rel_width > 0.3 and 
        label in ['equation', 'formula']):
        return ZoneClassification(
            zone=ZoneType.EQUATION,
            confidence=0.8,
            method='heuristic_geometry',
            features={'aspect_ratio': aspect_ratio, 'rel_height': rel_height}
        )
    
    return None


def classify_zone_heuristic(
    element: Dict,
    page_dims: Dict[str, int],
    cross_page_stats: Optional[Dict] = None
) -> ZoneClassification:
    """
    Main heuristic classification function.
    Applies multiple heuristic rules in priority order.
    
    Args:
        element: Layout element with bbox, label, text
        page_dims: Page dimensions
        cross_page_stats: Optional stats about repeated elements
    
    Returns:
        ZoneClassification (always returns something, may be UNKNOWN)
    """
    # Priority 1: Check if marked as repeated (header/footer)
    if cross_page_stats:
        from .filters import normalize_text_for_matching
        text = element.get('text_content', element.get('text', ''))
        normalized = normalize_text_for_matching(text)
        
        if normalized in cross_page_stats:
            info = cross_page_stats[normalized]
            if info.zone == 'header':
                return ZoneClassification(
                    zone=ZoneType.HEADER,
                    confidence=0.95,
                    method='heuristic_repetition'
                )
            elif info.zone == 'footer':
                return ZoneClassification(
                    zone=ZoneType.FOOTER,
                    confidence=0.95,
                    method='heuristic_repetition'
                )
    
    # Priority 2: Label-based (OCR grounding labels)
    label_result = classify_by_label(element)
    if label_result and label_result.confidence >= 0.8:
        return label_result
    
    # Priority 3: Text pattern matching
    pattern_result = classify_by_text_pattern(element)
    if pattern_result and pattern_result.confidence >= 0.8:
        return pattern_result
    
    # Priority 4: Position-based
    position_result = classify_by_position(element, page_dims)
    if position_result and position_result.confidence >= 0.7:
        return position_result
    
    # Priority 5: Geometry-based
    geometry_result = classify_by_geometry(element, page_dims)
    if geometry_result and geometry_result.confidence >= 0.7:
        return geometry_result
    
    # Priority 6: Lower confidence results
    if label_result:
        return label_result
    if pattern_result:
        return pattern_result
    if position_result:
        return position_result
    if geometry_result:
        return geometry_result
    
    # Fallback: classify as main_text (most common)
    return ZoneClassification(
        zone=ZoneType.MAIN_TEXT,
        confidence=0.5,
        method='fallback'
    )


def classify_zones_batch(
    elements: List[Dict],
    page_dims: Dict[str, int],
    cross_page_stats: Optional[Dict] = None
) -> List[Dict]:
    """
    Classify zones for a batch of elements.
    
    Args:
        elements: List of layout elements
        page_dims: Page dimensions
        cross_page_stats: Optional repetition stats
    
    Returns:
        List of elements with 'zone' and 'zone_confidence' added
    """
    results = []
    
    for elem in elements:
        classification = classify_zone_heuristic(elem, page_dims, cross_page_stats)
        
        elem_with_zone = {
            **elem,
            'zone': classification.zone.value,
            'zone_type': classification.zone,
            'zone_confidence': classification.confidence,
            'zone_method': classification.method
        }
        
        results.append(elem_with_zone)
    
    return results


def get_zone_priority(zone: ZoneType) -> int:
    """Get reading order priority for a zone type."""
    return ZONE_PRIORITY.get(zone, 5)


def group_elements_by_zone(
    elements: List[Dict]
) -> Dict[ZoneType, List[Dict]]:
    """
    Group elements by their zone classification.
    
    Args:
        elements: List of elements with 'zone_type' field
    
    Returns:
        Dict mapping ZoneType to list of elements
    """
    groups: Dict[ZoneType, List[Dict]] = {}
    
    for elem in elements:
        zone = elem.get('zone_type', ZoneType.UNKNOWN)
        if isinstance(zone, str):
            try:
                zone = ZoneType(zone)
            except ValueError:
                zone = ZoneType.UNKNOWN
        
        if zone not in groups:
            groups[zone] = []
        groups[zone].append(elem)
    
    return groups
