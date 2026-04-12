"""
Grouping Module

Groups layout elements into logical units:
- Line grouping: cluster elements on same line
- Block grouping: cluster lines into paragraphs/blocks
- Column detection: identify multi-column layouts
"""
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class Column:
    """Represents a detected column in the layout."""
    x1: float
    x2: float
    width: float
    index: int  # 0-indexed column number


@dataclass
class Block:
    """Represents a grouped block of text."""
    id: str
    elements: List[Dict] = field(default_factory=list)
    bbox: Dict = field(default_factory=dict)
    column_index: int = 0
    block_type: str = "text"  # text, figure, table, etc.


def estimate_median_line_height(elements: List[Dict]) -> float:
    """
    Estimate median line height from element bboxes.
    
    Args:
        elements: List of layout elements
    
    Returns:
        Median line height in pixels
    """
    heights = []
    
    for elem in elements:
        y1 = elem.get('bbox_y1', elem.get('y1', 0))
        y2 = elem.get('bbox_y2', elem.get('y2', 0))
        height = y2 - y1
        
        if height > 0:
            heights.append(height)
    
    if not heights:
        return 20.0  # Default line height
    
    return statistics.median(heights)


def detect_columns_projection(
    elements: List[Dict],
    page_width: int,
    min_gap_ratio: float = 0.05,
    bin_width: int = 5
) -> List[Column]:
    """
    Detect columns using X-axis projection profile.
    
    Projects all elements onto X-axis and finds valleys (gaps)
    that indicate column boundaries.
    
    Args:
        elements: List of layout elements
        page_width: Total page width
        min_gap_ratio: Minimum gap width as ratio of page width
        bin_width: Width of histogram bins
    
    Returns:
        List of detected columns
    """
    if not elements or page_width <= 0:
        return [Column(x1=0, x2=page_width, width=page_width, index=0)]
    
    # Create projection histogram
    num_bins = page_width // bin_width + 1
    histogram = [0] * num_bins
    
    for elem in elements:
        x1 = int(elem.get('bbox_x1', elem.get('x1', 0)))
        x2 = int(elem.get('bbox_x2', elem.get('x2', page_width)))
        
        # Fill histogram
        start_bin = max(0, x1 // bin_width)
        end_bin = min(num_bins - 1, x2 // bin_width)
        
        for b in range(start_bin, end_bin + 1):
            histogram[b] += 1
    
    # Find valleys (gaps)
    min_gap_width = int(page_width * min_gap_ratio)
    min_gap_bins = min_gap_width // bin_width
    
    valleys = []
    in_valley = False
    valley_start = 0
    
    for i, count in enumerate(histogram):
        if count == 0:
            if not in_valley:
                in_valley = True
                valley_start = i
        else:
            if in_valley:
                valley_end = i
                valley_width_bins = valley_end - valley_start
                
                if valley_width_bins >= min_gap_bins:
                    # Significant valley found
                    valley_center = (valley_start + valley_end) // 2 * bin_width
                    valleys.append(valley_center)
                
                in_valley = False
    
    # Create columns from valleys
    if not valleys:
        # Single column
        return [Column(x1=0, x2=page_width, width=page_width, index=0)]
    
    columns = []
    prev_x = 0
    
    for i, valley_x in enumerate(valleys):
        columns.append(Column(
            x1=prev_x,
            x2=valley_x,
            width=valley_x - prev_x,
            index=len(columns)
        ))
        prev_x = valley_x
    
    # Add final column
    columns.append(Column(
        x1=prev_x,
        x2=page_width,
        width=page_width - prev_x,
        index=len(columns)
    ))
    
    # Filter out very narrow columns (artifacts)
    min_col_width = page_width * 0.15
    columns = [c for c in columns if c.width >= min_col_width]
    
    # Re-index
    for i, col in enumerate(columns):
        col.index = i
    
    return columns if columns else [Column(x1=0, x2=page_width, width=page_width, index=0)]


def assign_column_membership(
    elements: List[Dict],
    columns: List[Column]
) -> List[Dict]:
    """
    Assign each element to a column based on its center position.
    
    Args:
        elements: List of layout elements
        columns: Detected columns
    
    Returns:
        Elements with 'column_index' added
    """
    result = []
    
    for elem in elements:
        x1 = elem.get('bbox_x1', elem.get('x1', 0))
        x2 = elem.get('bbox_x2', elem.get('x2', 0))
        center_x = (x1 + x2) / 2
        
        # Find containing column
        col_index = 0
        for col in columns:
            if col.x1 <= center_x <= col.x2:
                col_index = col.index
                break
        
        elem_with_col = {**elem, 'column_index': col_index}
        result.append(elem_with_col)
    
    return result


def group_into_lines(
    elements: List[Dict],
    vertical_tolerance: Optional[float] = None
) -> List[List[Dict]]:
    """
    Group elements into lines based on vertical proximity.
    
    Elements are on the same line if their vertical positions overlap
    significantly.
    
    Args:
        elements: List of layout elements
        vertical_tolerance: Max vertical gap to consider same line
                           (default: 0.3 * median line height)
    
    Returns:
        List of lines, each line is a list of elements
    """
    if not elements:
        return []
    
    # Estimate tolerance
    if vertical_tolerance is None:
        median_height = estimate_median_line_height(elements)
        vertical_tolerance = median_height * 0.3
    
    # Sort by y-position
    sorted_elements = sorted(
        elements,
        key=lambda e: e.get('bbox_y1', e.get('y1', 0))
    )
    
    lines = []
    current_line = [sorted_elements[0]]
    current_line_bottom = sorted_elements[0].get(
        'bbox_y2', sorted_elements[0].get('y2', 0)
    )
    
    for elem in sorted_elements[1:]:
        y1 = elem.get('bbox_y1', elem.get('y1', 0))
        y2 = elem.get('bbox_y2', elem.get('y2', 0))
        
        # Check if element overlaps with current line
        if y1 <= current_line_bottom + vertical_tolerance:
            current_line.append(elem)
            current_line_bottom = max(current_line_bottom, y2)
        else:
            # Start new line
            if current_line:
                # Sort line by x-position
                current_line.sort(
                    key=lambda e: e.get('bbox_x1', e.get('x1', 0))
                )
                lines.append(current_line)
            
            current_line = [elem]
            current_line_bottom = y2
    
    # Don't forget last line
    if current_line:
        current_line.sort(key=lambda e: e.get('bbox_x1', e.get('x1', 0)))
        lines.append(current_line)
    
    return lines


def group_lines_to_blocks(
    lines: List[List[Dict]],
    gap_threshold_ratio: float = 1.5,
    median_line_height: Optional[float] = None
) -> List[Block]:
    """
    Group lines into blocks based on vertical spacing.
    
    Lines with large vertical gaps between them are split into
    different blocks.
    
    Args:
        lines: List of lines (from group_into_lines)
        gap_threshold_ratio: Gap > ratio * median_height → new block
        median_line_height: Optional pre-computed median height
    
    Returns:
        List of Block objects
    """
    if not lines:
        return []
    
    # Estimate median line height
    if median_line_height is None:
        all_elements = [elem for line in lines for elem in line]
        median_line_height = estimate_median_line_height(all_elements)
    
    gap_threshold = median_line_height * gap_threshold_ratio
    
    # Calculate gaps between consecutive lines
    blocks = []
    current_block_lines = [lines[0]]
    
    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        curr_line = lines[i]
        
        # Get bottom of previous line
        prev_bottom = max(
            elem.get('bbox_y2', elem.get('y2', 0)) for elem in prev_line
        )
        
        # Get top of current line
        curr_top = min(
            elem.get('bbox_y1', elem.get('y1', 0)) for elem in curr_line
        )
        
        gap = curr_top - prev_bottom
        
        if gap > gap_threshold:
            # New block
            blocks.append(create_block_from_lines(
                current_block_lines, 
                len(blocks)
            ))
            current_block_lines = [curr_line]
        else:
            current_block_lines.append(curr_line)
    
    # Add final block
    if current_block_lines:
        blocks.append(create_block_from_lines(
            current_block_lines, 
            len(blocks)
        ))
    
    return blocks


def create_block_from_lines(
    lines: List[List[Dict]], 
    block_index: int
) -> Block:
    """Create a Block object from a list of lines."""
    all_elements = [elem for line in lines for elem in line]
    
    if not all_elements:
        return Block(id=f"block_{block_index}", elements=[])
    
    # Calculate bounding box
    x1 = min(e.get('bbox_x1', e.get('x1', 0)) for e in all_elements)
    y1 = min(e.get('bbox_y1', e.get('y1', 0)) for e in all_elements)
    x2 = max(e.get('bbox_x2', e.get('x2', 0)) for e in all_elements)
    y2 = max(e.get('bbox_y2', e.get('y2', 0)) for e in all_elements)
    
    # Get column index (from first element)
    col_index = all_elements[0].get('column_index', 0)
    
    # Determine block type from element zones/labels
    zones = [e.get('zone', 'main_text') for e in all_elements]
    labels = [e.get('label', 'text') for e in all_elements]
    
    block_type = "text"
    if 'figure' in zones or 'figure' in labels:
        block_type = "figure"
    elif 'table' in zones or 'table' in labels:
        block_type = "table"
    elif 'section_heading' in zones:
        block_type = "heading"
    
    return Block(
        id=f"block_{block_index}",
        elements=all_elements,
        bbox={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        column_index=col_index,
        block_type=block_type
    )


def link_captions_to_figures(
    elements: List[Dict],
    max_distance_ratio: float = 0.15
) -> List[Dict]:
    """
    Link caption elements to their corresponding figures/tables.
    
    Captions are usually directly below or above figures.
    
    Args:
        elements: List of layout elements with zone classification
        max_distance_ratio: Max distance as ratio of figure height
    
    Returns:
        Elements with 'linked_to' field for captions
    """
    figures = [e for e in elements if e.get('zone') in ['figure', 'table']]
    captions = [e for e in elements if e.get('zone') == 'caption']
    
    if not figures or not captions:
        return elements
    
    # For each caption, find nearest figure
    for caption in captions:
        cap_y1 = caption.get('bbox_y1', caption.get('y1', 0))
        cap_y2 = caption.get('bbox_y2', caption.get('y2', 0))
        cap_center_x = (caption.get('bbox_x1', 0) + caption.get('bbox_x2', 0)) / 2
        
        best_figure = None
        best_distance = float('inf')
        
        for fig in figures:
            fig_y1 = fig.get('bbox_y1', fig.get('y1', 0))
            fig_y2 = fig.get('bbox_y2', fig.get('y2', 0))
            fig_height = fig_y2 - fig_y1
            fig_center_x = (fig.get('bbox_x1', 0) + fig.get('bbox_x2', 0)) / 2
            
            # Check horizontal alignment
            x_distance = abs(cap_center_x - fig_center_x)
            fig_width = fig.get('bbox_x2', 0) - fig.get('bbox_x1', 0)
            
            if x_distance > fig_width * 0.5:
                continue  # Not aligned
            
            # Check if caption is below or above figure
            if cap_y1 >= fig_y1:  # Caption below
                distance = cap_y1 - fig_y2
            else:  # Caption above
                distance = fig_y1 - cap_y2
            
            max_distance = fig_height * max_distance_ratio
            
            if 0 <= distance <= max_distance and distance < best_distance:
                best_distance = distance
                best_figure = fig
        
        if best_figure:
            caption['linked_to'] = best_figure.get('id')
            caption['linked_zone'] = best_figure.get('zone')
    
    return elements


def group_elements_by_page_and_column(
    elements: List[Dict],
    page_dims: Dict[int, Dict[str, int]]
) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Group elements by page number and column index.
    
    Args:
        elements: List of layout elements
        page_dims: Dict mapping page_number to {'width': int, 'height': int}
    
    Returns:
        Dict mapping (page_number, column_index) to list of elements
    """
    # First detect columns per page
    pages: Dict[int, List[Dict]] = defaultdict(list)
    for elem in elements:
        page = elem.get('page_number', elem.get('page', 1))
        pages[page].append(elem)
    
    result: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    
    for page_num, page_elements in pages.items():
        dims = page_dims.get(page_num, {'width': 800, 'height': 1000})
        
        # Detect columns for this page
        columns = detect_columns_projection(
            page_elements, 
            dims.get('width', 800)
        )
        
        # Assign elements to columns
        elements_with_cols = assign_column_membership(page_elements, columns)
        
        # Group by column
        for elem in elements_with_cols:
            col_idx = elem.get('column_index', 0)
            result[(page_num, col_idx)].append(elem)
    
    return result


def process_page_layout(
    elements: List[Dict],
    page_dims: Dict[str, int],
    detect_multi_column: bool = True,
    group_blocks: bool = True,
    link_captions: bool = True
) -> Dict:
    """
    Full layout processing pipeline for a single page.
    
    Args:
        elements: Layout elements for one page
        page_dims: Page dimensions {'width': int, 'height': int}
        detect_multi_column: Whether to detect columns
        group_blocks: Whether to group lines into blocks
        link_captions: Whether to link captions to figures
    
    Returns:
        Dict with processing results:
        - columns: Detected columns
        - blocks: Grouped blocks
        - elements: Processed elements
    """
    result = {
        'columns': [],
        'blocks': [],
        'elements': elements
    }
    
    if not elements:
        return result
    
    # Step 1: Detect columns
    columns = [Column(x1=0, x2=page_dims.get('width', 800), 
                      width=page_dims.get('width', 800), index=0)]
    
    if detect_multi_column:
        columns = detect_columns_projection(
            elements, 
            page_dims.get('width', 800)
        )
    
    result['columns'] = columns
    
    # Step 2: Assign column membership
    elements = assign_column_membership(elements, columns)
    
    # Step 3: Link captions
    if link_captions:
        elements = link_captions_to_figures(elements)
    
    result['elements'] = elements
    
    # Step 4: Group into blocks
    if group_blocks:
        all_blocks = []
        
        # Process each column separately
        for col in columns:
            col_elements = [
                e for e in elements 
                if e.get('column_index', 0) == col.index
            ]
            
            if col_elements:
                lines = group_into_lines(col_elements)
                blocks = group_lines_to_blocks(lines)
                
                # Update column index in blocks
                for block in blocks:
                    block.column_index = col.index
                
                all_blocks.extend(blocks)
        
        result['blocks'] = all_blocks
    
    return result
