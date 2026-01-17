"""
Hierarchical Thinning Module

Implements spatial-aware thinning for document trees with focus on:
- Translation: Paragraph-level chunks với context
- Summarization: Multi-level hierarchy
- Content extraction: Preserve semantic units

Strategy:
1. Title nodes = Section boundaries (NEVER merge across)
2. Consecutive text blocks → Paragraphs (conservative merge)
3. Equations/Figures/Tables = Barriers (standalone, no merge)
4. Preserve reading order and spatial proximity
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


def can_merge_text_blocks(
    node1: Dict,
    node2: Dict,
    median_line_height: float,
    gap_threshold_multiplier: float = 2.0,
    overlap_threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Determine if two text blocks can be merged into a paragraph.
    
    Rules:
    1. Both must be 'text' label (NOT title, equation, etc.)
    2. Same page
    3. Vertical gap < threshold (same paragraph flow)
    4. Horizontal overlap sufficient (same column)
    
    Args:
        node1: First node
        node2: Second node
        median_line_height: Median line height for gap threshold
        gap_threshold_multiplier: Multiplier for gap threshold
        overlap_threshold: Minimum horizontal overlap ratio
    
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
    gap_threshold = median_line_height * gap_threshold_multiplier
    
    if gap < 0:
        return False, f"Negative gap (overlap): {gap}"
    
    if gap > gap_threshold:
        return False, f"Gap too large: {gap:.1f} > {gap_threshold:.1f}"
    
    # Rule 4: Horizontal overlap check
    overlap = calculate_horizontal_overlap(node1, node2)
    
    if overlap < overlap_threshold:
        return False, f"Insufficient overlap: {overlap:.2f} < {overlap_threshold}"
    
    return True, f"Mergeable (gap={gap:.1f}, overlap={overlap:.2f})"


def has_barrier_between(
    node1_idx: int,
    node2_idx: int,
    nodes: List[Dict],
    barrier_labels: Set[str]
) -> bool:
    """
    Check if there's a barrier node between two nodes in reading order.
    
    Barriers: equation, figure, table, title
    
    Args:
        node1_idx: Index of first node
        node2_idx: Index of second node
        nodes: Ordered list of nodes
        barrier_labels: Set of barrier labels
    
    Returns:
        True if barrier exists between
    """
    if node2_idx <= node1_idx + 1:
        return False  # Adjacent, no room for barrier
    
    # Check nodes between
    for i in range(node1_idx + 1, node2_idx):
        label = nodes[i].get('label', '').lower()
        if label in barrier_labels:
            return True
    
    return False


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
        'text_full': '\n\n'.join(text_fulls),
        'page_number': nodes[0].get('page_number', 1),
        'merged_from': len(nodes),
        'original_labels': [n.get('label') for n in nodes]
    }
    
    return merged


def hierarchical_thinning(
    nodes: List[Dict],
    preserve_barriers: bool = True,
    merge_text_to_paragraphs: bool = True,
    gap_threshold_multiplier: float = 2.0,
    min_paragraph_tokens: int = 50
) -> List[Dict]:
    """
    Apply hierarchical thinning to node list.
    
    Strategy:
    1. Preserve title/equation/figure/table as section boundaries
    2. Merge consecutive text blocks into paragraphs
    3. Maintain reading order
    
    Args:
        nodes: List of layout nodes (should be in reading order)
        preserve_barriers: Keep equation/figure/table standalone
        merge_text_to_paragraphs: Merge text blocks
        gap_threshold_multiplier: Gap threshold for merging
        min_paragraph_tokens: Minimum tokens for paragraph (unused for now)
    
    Returns:
        Thinned list of nodes
    """
    if not nodes:
        return []
    
    # Barrier labels that prevent merging
    barrier_labels = {'title', 'equation', 'formula', 'figure', 'table', 'caption'}
    
    # Calculate median line height
    median_line_height = estimate_median_line_height(nodes)
    
    # Find merge candidates
    merge_groups = []  # List of [indices] to merge
    current_group = [0]
    
    for i in range(1, len(nodes)):
        prev_node = nodes[i-1]
        curr_node = nodes[i]
        
        # Check if can merge with previous
        if merge_text_to_paragraphs:
            can_merge, reason = can_merge_text_blocks(
                prev_node,
                curr_node,
                median_line_height,
                gap_threshold_multiplier
            )
            
            # Also check for barriers
            if can_merge and preserve_barriers:
                if has_barrier_between(i-1, i, nodes, barrier_labels):
                    can_merge = False
                    reason = "Barrier between nodes"
            
            if can_merge:
                # Add to current group
                current_group.append(i)
            else:
                # Finalize current group
                if len(current_group) > 0:
                    merge_groups.append(current_group)
                current_group = [i]
        else:
            # No merging, each node is its own group
            merge_groups.append(current_group)
            current_group = [i]
    
    # Don't forget last group
    if current_group:
        merge_groups.append(current_group)
    
    # Create thinned nodes
    thinned_nodes = []
    
    for group in merge_groups:
        if len(group) == 1:
            # Single node, keep as-is
            thinned_nodes.append(nodes[group[0]])
        else:
            # Multiple nodes, merge
            group_nodes = [nodes[idx] for idx in group]
            merged_node = merge_nodes_content(group_nodes)
            thinned_nodes.append(merged_node)
    
    return thinned_nodes


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
