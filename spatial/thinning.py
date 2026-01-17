"""
Hierarchical Thinning Module

Implements spatial-aware thinning for document trees with 2-tier approach:

Tier A (Intra-page merge):
- Merge consecutive text blocks → paragraphs (scanline algorithm)
- Merge captions with figures/tables
- Process each page independently

Tier B (Section hierarchy):
- Build section tree with title/subtitle
- Assign content blocks to sections

Strategy:
1. Barriers (equation, figure, table, title) = NEVER merge
2. Text blocks merge only if: same page, small gap, good overlap, same column
3. Preserve reading order and spatial proximity
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class MergeCandidate:
    """Candidate for merging nodes."""
    node1_id: int
    node2_id: int
    merge_reason: str
    confidence: float


def calculate_vertical_gap(node1: Dict, node2: Dict) -> float:
    """
    Calculate vertical gap between two nodes.
    
    Args:
        node1: First node (above)
        node2: Second node (below)
    
    Returns:
        Vertical distance in pixels
    """
    y2_node1 = node1.get('bbox_y2', node1.get('y2', 0))
    y1_node2 = node2.get('bbox_y1', node2.get('y1', 0))
    
    return y1_node2 - y2_node1


def calculate_horizontal_overlap(node1: Dict, node2: Dict) -> float:
    """
    Calculate horizontal overlap ratio between two nodes.
    
    Args:
        node1: First node
        node2: Second node
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    x1_n1 = node1.get('bbox_x1', node1.get('x1', 0))
    x2_n1 = node1.get('bbox_x2', node1.get('x2', 0))
    x1_n2 = node2.get('bbox_x1', node2.get('x1', 0))
    x2_n2 = node2.get('bbox_x2', node2.get('x2', 0))
    
    # Intersection
    overlap_start = max(x1_n1, x1_n2)
    overlap_end = min(x2_n1, x2_n2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_width = overlap_end - overlap_start
    min_width = min(x2_n1 - x1_n1, x2_n2 - x1_n2)
    
    return overlap_width / min_width if min_width > 0 else 0.0


def estimate_median_line_height(nodes: List[Dict]) -> float:
    """
    Estimate median line height from node heights.
    
    Args:
        nodes: List of layout nodes
    
    Returns:
        Median line height in pixels
    """
    heights = []
    for node in nodes:
        y1 = node.get('bbox_y1', node.get('y1', 0))
        y2 = node.get('bbox_y2', node.get('y2', 0))
        height = y2 - y1
        if height > 0:
            heights.append(height)
    
    return statistics.median(heights) if heights else 40.0


def estimate_gap_threshold_dynamic(text_nodes: List[Dict], median_line_height: float) -> float:
    """
    Estimate gap threshold dynamically from data (percentile 70 of actual gaps).
    
    Args:
        text_nodes: Text nodes only
        median_line_height: Fallback if no gaps found
    
    Returns:
        Gap threshold in pixels
    """
    gaps = []
    
    # Sort by page then y position
    sorted_nodes = sorted(
        text_nodes,
        key=lambda n: (n.get('page_number', 1), n.get('bbox_y1', n.get('y1', 0)))
    )
    
    for i in range(len(sorted_nodes) - 1):
        node1 = sorted_nodes[i]
        node2 = sorted_nodes[i + 1]
        
        # Only consider gaps within same page
        if node1.get('page_number', 1) != node2.get('page_number', 1):
            continue
        
        gap = calculate_vertical_gap(node1, node2)
        if gap >= 0:
            gaps.append(gap)
    
    if gaps:
        # Use percentile 70 as suggested in idea
        gaps_sorted = sorted(gaps)
        idx = int(len(gaps_sorted) * 0.7)
        return gaps_sorted[min(idx, len(gaps_sorted) - 1)]
    else:
        # Fallback to median line height * 1.5
        return median_line_height * 1.5


def can_merge_text_blocks(
    node1: Dict,
    node2: Dict,
    gap_threshold: float,
    overlap_threshold: float = 0.5,
    same_left_threshold: float = 10.0,
    indent_threshold: float = 30.0
) -> Tuple[bool, str]:
    """
    Determine if two text blocks can be merged into a paragraph.
    
    Rules:
    1. Both must be 'text' label (NOT title, equation, etc.)
    2. Same page
    3. Vertical gap < threshold (same paragraph flow)
    4. Horizontal overlap sufficient OR same left edge
    5. Not indented (list items)
    
    Args:
        node1: First node
        node2: Second node
        gap_threshold: Maximum vertical gap to allow merge
        overlap_threshold: Minimum horizontal overlap ratio
        same_left_threshold: Threshold for "same left edge" (pixels)
        indent_threshold: Indentation that suggests list item (pixels)
    
    Returns:
        (can_merge: bool, reason: str)
    """
    # Rule 1: Both must be 'text'
    label1 = node1.get('label', '').lower()
    label2 = node2.get('label', '').lower()
    
    if label1 != 'text' or label2 != 'text':
        return False, f"Not both text: {label1}, {label2}"
    
    # Rule 2: Same page
    page1 = node1.get('page_number', 1)
    page2 = node2.get('page_number', 1)
    
    if page1 != page2:
        return False, "Different pages"
    
    # Rule 3: Vertical gap check
    gap = calculate_vertical_gap(node1, node2)
    
    if gap < 0:
        return False, f"Negative gap (overlap): {gap:.1f}"
    
    if gap > gap_threshold:
        return False, f"Gap too large: {gap:.1f} > {gap_threshold:.1f}"
    
    # Rule 4: Horizontal overlap OR same left edge
    overlap = calculate_horizontal_overlap(node1, node2)
    
    x1_node1 = node1.get('bbox_x1', node1.get('x1', 0))
    x1_node2 = node2.get('bbox_x1', node2.get('x1', 0))
    same_left = abs(x1_node1 - x1_node2) <= same_left_threshold
    
    if overlap < overlap_threshold and not same_left:
        return False, f"Insufficient overlap: {overlap:.2f} < {overlap_threshold} and not same_left"
    
    # Rule 5: Check for indent (list items)
    indent = x1_node2 - x1_node1
    
    if indent >= indent_threshold:
        return False, f"Indented (looks like list item): indent={indent:.1f}px"
    
    return True, f"Mergeable (gap={gap:.1f}, overlap={overlap:.2f})"


def merge_nodes_content(nodes: List[Dict]) -> Dict:
    """
    Merge multiple nodes into a single paragraph node.
    
    Args:
        nodes: List of nodes to merge
    
    Returns:
        Merged node dict
    """
    if not nodes:
        return {}
    
    # Union bbox
    x1_min = min(n.get('bbox_x1', n.get('x1', 0)) for n in nodes)
    y1_min = min(n.get('bbox_y1', n.get('y1', 0)) for n in nodes)
    x2_max = max(n.get('bbox_x2', n.get('x2', 0)) for n in nodes)
    y2_max = max(n.get('bbox_y2', n.get('y2', 0)) for n in nodes)
    
    # Concatenate text
    texts = [n.get('text_content', '') for n in nodes if n.get('text_content')]
    text_fulls = [n.get('text_full', '') for n in nodes if n.get('text_full')]
    
    merged = {
        'label': 'paragraph',  # New label for merged
        'bbox_x1': x1_min,
        'bbox_y1': y1_min,
        'bbox_x2': x2_max,
        'bbox_y2': y2_max,
        'x1': x1_min,  # Backward compat
        'y1': y1_min,
        'x2': x2_max,
        'y2': y2_max,
        'text_content': ' '.join(texts),
        'text_full': '\n'.join(text_fulls),  # Join with newline for better readability
        'page_number': nodes[0].get('page_number', 1),
        'merged_from': len(nodes),
        'original_labels': [n.get('label') for n in nodes]
    }
    
    return merged


def merge_text_blocks_in_page(
    page_nodes: List[Dict],
    gap_threshold: float,
    barrier_labels: Set[str]
) -> List[Dict]:
    """
    Merge consecutive text blocks within a single page using scanline algorithm.
    
    CRITICAL FIX: Use correct scanline merge - track current_group and compare
    with LAST node in group, not just previous node in array.
    
    Args:
        page_nodes: Nodes from single page (already sorted by reading order)
        gap_threshold: Maximum vertical gap for merging
        barrier_labels: Labels that act as barriers (standalone)
    
    Returns:
        Merged blocks (paragraphs + barriers)
    """
    if not page_nodes:
        return []
    
    merge_groups = []
    current_group = None
    
    for i, node in enumerate(page_nodes):
        label = node.get('label', '').lower()
        
        # Barrier nodes: standalone
        if label in barrier_labels:
            # Finalize current text group if exists
            if current_group:
                merge_groups.append(current_group)
                current_group = None
            
            # Add barrier as standalone
            merge_groups.append([i])
            continue
        
        # Text nodes: check merge with current group
        if label == 'text':
            if current_group is None:
                # Start new group
                current_group = [i]
            else:
                # Check if can merge with LAST node in current_group
                last_idx = current_group[-1]
                last_node = page_nodes[last_idx]
                
                can_merge, reason = can_merge_text_blocks(
                    last_node,
                    node,
                    gap_threshold
                )
                
                if can_merge:
                    # Add to current group
                    current_group.append(i)
                else:
                    # Finalize current group, start new
                    merge_groups.append(current_group)
                    current_group = [i]
        else:
            # Other labels: standalone
            if current_group:
                merge_groups.append(current_group)
                current_group = None
            merge_groups.append([i])
    
    # Don't forget last group
    if current_group:
        merge_groups.append(current_group)
    
    # Create merged nodes
    merged_result = []
    for group in merge_groups:
        if len(group) == 1:
            # Single node, keep as-is
            merged_result.append(page_nodes[group[0]])
        else:
            # Multiple nodes, merge into paragraph
            group_nodes = [page_nodes[idx] for idx in group]
            merged_node = merge_nodes_content(group_nodes)
            merged_result.append(merged_node)
    
    return merged_result


def hierarchical_thinning(
    nodes: List[Dict],
    preserve_barriers: bool = True,
    merge_text_to_paragraphs: bool = True,
    gap_threshold_multiplier: float = 2.0,
    use_dynamic_gap: bool = True,
    min_paragraph_tokens: int = 50
) -> List[Dict]:
    """
    Apply hierarchical thinning to node list with 2-tier approach.
    
    Tier A: Intra-page merge
    - Group nodes by page
    - Merge text blocks within each page
    - Respect barriers (equation, figure, table, title)
    
    Tier B: (Future) Section hierarchy building
    - Not implemented yet in this version
    
    Args:
        nodes: List of layout nodes (should be in reading order)
        preserve_barriers: Keep equation/figure/table standalone
        merge_text_to_paragraphs: Merge text blocks
        gap_threshold_multiplier: Gap threshold multiplier (if not using dynamic)
        use_dynamic_gap: Use dynamic gap threshold from data
        min_paragraph_tokens: Minimum tokens for paragraph (unused for now)
    
    Returns:
        Thinned list of nodes
    """
    if not nodes:
        return []
    
    # Barrier labels that prevent merging
    barrier_labels = {
        'title', 'subtitle', 'heading', 'sub_title',  # Section boundaries
        'equation', 'formula',  # Math
        'image', 'figure',  # Images
        'table', 'tablecaption', 'tablefootnote',  # Tables
        'imagecaption', 'caption'  # Captions
    }
    
    if not merge_text_to_paragraphs:
        # No merging, return as-is
        return nodes
    
    # Calculate median line height
    median_line_height = estimate_median_line_height(nodes)
    
    # Calculate gap threshold
    if use_dynamic_gap:
        # Dynamic threshold from data (idea approach)
        text_nodes = [n for n in nodes if n.get('label', '').lower() == 'text']
        gap_threshold = estimate_gap_threshold_dynamic(text_nodes, median_line_height)
    else:
        # Fixed multiplier
        gap_threshold = median_line_height * gap_threshold_multiplier
    
    # Tier A: Merge per-page
    # Group nodes by page
    pages = {}
    for node in nodes:
        page_num = node.get('page_number', 1)
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(node)
    
    # Process each page independently
    merged_blocks = []
    for page_num in sorted(pages.keys()):
        page_nodes = pages[page_num]
        
        # Merge text blocks within this page
        page_merged = merge_text_blocks_in_page(
            page_nodes,
            gap_threshold,
            barrier_labels if preserve_barriers else set()
        )
        
        merged_blocks.extend(page_merged)
    
    return merged_blocks


def apply_thinning_to_tree(
    tree: Dict,
    **thinning_kwargs
) -> Dict:
    """
    Apply thinning to entire tree structure.
    
    Recursively processes tree nodes and applies thinning at each level.
    
    Args:
        tree: Tree dict from build_spatial_tree
        **thinning_kwargs: Arguments for hierarchical_thinning
    
    Returns:
        Thinned tree
    """
    # Extract nodes from tree
    children = tree.get('children', [])
    
    if not children:
        return tree
    
    # Flatten children to list of dicts
    nodes_list = []
    for child in children:
        # Convert TreeNode-like structure to dict if needed
        if hasattr(child, 'to_dict'):
            node_dict = child.to_dict()
        else:
            node_dict = child
        nodes_list.append(node_dict)
    
    # Apply thinning
    thinned_nodes = hierarchical_thinning(nodes_list, **thinning_kwargs)
    
    # Update tree
    tree['children'] = thinned_nodes
    
    # Update metadata
    if '_pipeline_info' in tree:
        tree['_pipeline_info']['thinning_applied'] = True
        tree['_pipeline_info']['nodes_before_thinning'] = len(nodes_list)
        tree['_pipeline_info']['nodes_after_thinning'] = len(thinned_nodes)
    
    return tree
