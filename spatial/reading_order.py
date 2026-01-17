"""
Reading Order Module

Determines reading order of document elements using topological sorting.
Builds a DAG from spatial relationships and applies topological sort
to produce a natural reading sequence.
"""
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class SpatialRelation(Enum):
    """Spatial relationship between two elements."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    SAME_ROW = "same_row"
    SAME_COLUMN = "same_column"
    OVERLAPPING = "overlapping"
    NO_RELATION = "no_relation"


@dataclass
class Edge:
    """Directed edge in reading order graph."""
    source_id: str
    target_id: str
    weight: float = 1.0
    relation: str = ""


def get_bbox_center(element: Dict) -> Tuple[float, float]:
    """Get center point of element bbox."""
    x1 = element.get('bbox_x1', element.get('x1', 0))
    y1 = element.get('bbox_y1', element.get('y1', 0))
    x2 = element.get('bbox_x2', element.get('x2', 0))
    y2 = element.get('bbox_y2', element.get('y2', 0))
    
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_horizontal_overlap(elem_a: Dict, elem_b: Dict) -> float:
    """
    Calculate horizontal overlap ratio between two elements.
    Returns value between 0 (no overlap) and 1 (complete overlap).
    """
    a_x1 = elem_a.get('bbox_x1', elem_a.get('x1', 0))
    a_x2 = elem_a.get('bbox_x2', elem_a.get('x2', 0))
    b_x1 = elem_b.get('bbox_x1', elem_b.get('x1', 0))
    b_x2 = elem_b.get('bbox_x2', elem_b.get('x2', 0))
    
    overlap_start = max(a_x1, b_x1)
    overlap_end = min(a_x2, b_x2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_width = overlap_end - overlap_start
    min_width = min(a_x2 - a_x1, b_x2 - b_x1)
    
    if min_width <= 0:
        return 0.0
    
    return overlap_width / min_width


def calculate_vertical_overlap(elem_a: Dict, elem_b: Dict) -> float:
    """
    Calculate vertical overlap ratio between two elements.
    Returns value between 0 (no overlap) and 1 (complete overlap).
    """
    a_y1 = elem_a.get('bbox_y1', elem_a.get('y1', 0))
    a_y2 = elem_a.get('bbox_y2', elem_a.get('y2', 0))
    b_y1 = elem_b.get('bbox_y1', elem_b.get('y1', 0))
    b_y2 = elem_b.get('bbox_y2', elem_b.get('y2', 0))
    
    overlap_start = max(a_y1, b_y1)
    overlap_end = min(a_y2, b_y2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_height = overlap_end - overlap_start
    min_height = min(a_y2 - a_y1, b_y2 - b_y1)
    
    if min_height <= 0:
        return 0.0
    
    return overlap_height / min_height


def should_read_before(
    elem_a: Dict,
    elem_b: Dict,
    same_column_threshold: float = 0.3,
    same_row_threshold: float = 0.3
) -> Optional[bool]:
    """
    Determine if element A should be read before element B.
    
    Rules (in priority order):
    1. Zone priority: title_block < main_text < footnote
    2. Same column: top-to-bottom (by y-position)
    3. Same row but different column: left-to-right (by x-position)
    4. Different zones: by zone priority
    
    Args:
        elem_a: First element
        elem_b: Second element
        same_column_threshold: Horizontal overlap threshold to consider same column
        same_row_threshold: Vertical overlap threshold to consider same row
    
    Returns:
        True if A before B, False if B before A, None if no clear ordering
    """
    # Get positions
    a_center_x, a_center_y = get_bbox_center(elem_a)
    b_center_x, b_center_y = get_bbox_center(elem_b)
    
    # Check overlaps
    h_overlap = calculate_horizontal_overlap(elem_a, elem_b)
    v_overlap = calculate_vertical_overlap(elem_a, elem_b)
    
    # Zone priority (if available)
    a_zone_priority = elem_a.get('zone_priority', 5)
    b_zone_priority = elem_b.get('zone_priority', 5)
    
    if a_zone_priority != b_zone_priority:
        return a_zone_priority < b_zone_priority
    
    # Same column (significant horizontal overlap): top-to-bottom
    if h_overlap > same_column_threshold:
        if abs(a_center_y - b_center_y) > 5:  # Not at same y-level
            return a_center_y < b_center_y
    
    # Same row (significant vertical overlap): left-to-right
    if v_overlap > same_row_threshold:
        if abs(a_center_x - b_center_x) > 5:  # Not at same x-level
            return a_center_x < b_center_x
    
    # Different positions with no overlap
    # Prioritize top-to-bottom, then left-to-right
    a_y1 = elem_a.get('bbox_y1', elem_a.get('y1', 0))
    b_y1 = elem_b.get('bbox_y1', elem_b.get('y1', 0))
    a_y2 = elem_a.get('bbox_y2', elem_a.get('y2', 0))
    
    # If A ends before B starts (vertically): A before B
    if a_y2 < b_y1:
        return True
    
    # If in roughly same vertical band: left-to-right
    if abs(a_y1 - b_y1) < (elem_a.get('bbox_y2', 0) - a_y1) * 0.5:
        return a_center_x < b_center_x
    
    # Top-to-bottom as fallback
    return a_center_y < b_center_y


def build_reading_order_graph(
    elements: List[Dict],
    include_zone_priority: bool = True
) -> Dict[str, List[Edge]]:
    """
    Build a directed graph for reading order.
    
    Creates edges A→B when A should be read before B.
    
    Args:
        elements: List of layout elements with bbox and optional zone
        include_zone_priority: Whether to consider zone priorities
    
    Returns:
        Adjacency list representation: {node_id: [outgoing edges]}
    """
    if not elements:
        return {}
    
    # Assign IDs if not present
    for i, elem in enumerate(elements):
        if 'id' not in elem:
            elem['id'] = f"elem_{i}"
    
    # Add zone priority to elements
    if include_zone_priority:
        from core.constants import ZONE_PRIORITY
        for elem in elements:
            zone = elem.get('zone', 'unknown')
            elem['zone_priority'] = ZONE_PRIORITY.get(zone, 5)
    
    # Build graph
    graph: Dict[str, List[Edge]] = {elem['id']: [] for elem in elements}
    
    # Compare all pairs
    for i, elem_a in enumerate(elements):
        for j, elem_b in enumerate(elements):
            if i >= j:
                continue
            
            result = should_read_before(elem_a, elem_b)
            
            if result is True:
                graph[elem_a['id']].append(Edge(
                    source_id=elem_a['id'],
                    target_id=elem_b['id'],
                    weight=1.0,
                    relation="before"
                ))
            elif result is False:
                graph[elem_b['id']].append(Edge(
                    source_id=elem_b['id'],
                    target_id=elem_a['id'],
                    weight=1.0,
                    relation="before"
                ))
    
    return graph


def detect_cycles(graph: Dict[str, List[Edge]]) -> List[List[str]]:
    """
    Detect cycles in the reading order graph.
    
    Args:
        graph: Adjacency list representation
    
    Returns:
        List of cycles (each cycle is a list of node IDs)
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    parent = {node: None for node in graph}
    cycles = []
    
    def dfs(node: str, path: List[str]):
        color[node] = GRAY
        path.append(node)
        
        for edge in graph.get(node, []):
            next_node = edge.target_id
            
            if color.get(next_node, WHITE) == GRAY:
                # Found cycle
                cycle_start = path.index(next_node)
                cycles.append(path[cycle_start:].copy())
            elif color.get(next_node, WHITE) == WHITE:
                parent[next_node] = node
                dfs(next_node, path)
        
        path.pop()
        color[node] = BLACK
    
    for node in graph:
        if color[node] == WHITE:
            dfs(node, [])
    
    return cycles


def break_cycles(
    graph: Dict[str, List[Edge]],
    elements: List[Dict]
) -> Dict[str, List[Edge]]:
    """
    Break cycles in the graph by removing edges.
    
    Uses geometric fallback: in a cycle, remove edge that goes
    from lower element to higher element (wrong direction).
    
    Args:
        graph: Adjacency list with cycles
        elements: Original elements for geometric info
    
    Returns:
        Graph with cycles removed
    """
    id_to_elem = {elem['id']: elem for elem in elements}
    cycles = detect_cycles(graph)
    
    while cycles:
        cycle = cycles[0]
        
        # Find edge to remove (the one going "upward")
        worst_edge = None
        worst_score = float('-inf')
        
        for i in range(len(cycle)):
            source_id = cycle[i]
            target_id = cycle[(i + 1) % len(cycle)]
            
            source = id_to_elem.get(source_id, {})
            target = id_to_elem.get(target_id, {})
            
            source_y = source.get('bbox_y1', source.get('y1', 0))
            target_y = target.get('bbox_y1', target.get('y1', 0))
            
            # If source is below target, this edge is suspicious
            score = source_y - target_y  # Positive if going upward
            
            if score > worst_score:
                worst_score = score
                worst_edge = (source_id, target_id)
        
        # Remove worst edge
        if worst_edge:
            source_id, target_id = worst_edge
            graph[source_id] = [
                e for e in graph[source_id] 
                if e.target_id != target_id
            ]
        
        # Recheck for cycles
        cycles = detect_cycles(graph)
    
    return graph


def topological_sort(
    graph: Dict[str, List[Edge]],
    elements: List[Dict]
) -> List[str]:
    """
    Perform topological sort on the reading order graph.
    
    Uses Kahn's algorithm with geometric tie-breaking.
    
    Args:
        graph: DAG in adjacency list format
        elements: Original elements for tie-breaking
    
    Returns:
        List of element IDs in reading order
    """
    id_to_elem = {elem['id']: elem for elem in elements}
    
    # Calculate in-degree
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for edge in graph[node]:
            in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1
    
    # Initialize queue with nodes having in-degree 0
    # Sort by (y, x) for deterministic ordering
    def get_sort_key(node_id: str) -> Tuple[float, float]:
        elem = id_to_elem.get(node_id, {})
        y = elem.get('bbox_y1', elem.get('y1', 0))
        x = elem.get('bbox_x1', elem.get('x1', 0))
        return (y, x)
    
    queue = sorted(
        [node for node in in_degree if in_degree[node] == 0],
        key=get_sort_key
    )
    
    result = []
    
    while queue:
        # Take node with smallest (y, x)
        node = queue.pop(0)
        result.append(node)
        
        # Update in-degrees
        for edge in graph.get(node, []):
            target = edge.target_id
            in_degree[target] -= 1
            
            if in_degree[target] == 0:
                # Insert in sorted position
                queue.append(target)
                queue.sort(key=get_sort_key)
    
    # Check for remaining nodes (not all nodes were reached)
    if len(result) < len(graph):
        # Add remaining nodes in geometric order
        remaining = [n for n in graph if n not in result]
        remaining.sort(key=get_sort_key)
        result.extend(remaining)
    
    return result


def get_reading_order(
    elements: List[Dict],
    include_zone_priority: bool = True,
    break_cycles_enabled: bool = True
) -> List[Dict]:
    """
    Get elements in reading order using topological sort.
    
    Main entry point for reading order calculation.
    
    Args:
        elements: List of layout elements
        include_zone_priority: Whether to use zone priorities
        break_cycles_enabled: Whether to break cycles automatically
    
    Returns:
        List of elements sorted in reading order
    """
    if not elements:
        return []
    
    if len(elements) == 1:
        return elements
    
    # Ensure all elements have IDs
    for i, elem in enumerate(elements):
        if 'id' not in elem:
            elem['id'] = f"elem_{i}"
    
    # Build graph
    graph = build_reading_order_graph(elements, include_zone_priority)
    
    # Break cycles if any
    if break_cycles_enabled:
        cycles = detect_cycles(graph)
        if cycles:
            graph = break_cycles(graph, elements)
    
    # Topological sort
    sorted_ids = topological_sort(graph, elements)
    
    # Map back to elements
    id_to_elem = {elem['id']: elem for elem in elements}
    
    return [id_to_elem[node_id] for node_id in sorted_ids if node_id in id_to_elem]


def get_reading_order_by_page(
    elements: List[Dict],
    include_zone_priority: bool = True
) -> Dict[int, List[Dict]]:
    """
    Get reading order grouped by page.
    
    Args:
        elements: List of layout elements with page_number
        include_zone_priority: Whether to use zone priorities
    
    Returns:
        Dict mapping page_number to ordered elements
    """
    # Group by page
    pages: Dict[int, List[Dict]] = defaultdict(list)
    for elem in elements:
        page = elem.get('page_number', elem.get('page', 1))
        pages[page].append(elem)
    
    # Sort each page
    result = {}
    for page_num in sorted(pages.keys()):
        result[page_num] = get_reading_order(
            pages[page_num], 
            include_zone_priority
        )
    
    return result


def flatten_reading_order(
    pages_order: Dict[int, List[Dict]]
) -> List[Dict]:
    """
    Flatten page-based reading order into single list.
    
    Args:
        pages_order: Dict from get_reading_order_by_page
    
    Returns:
        Flat list of elements in document reading order
    """
    result = []
    for page_num in sorted(pages_order.keys()):
        result.extend(pages_order[page_num])
    return result
