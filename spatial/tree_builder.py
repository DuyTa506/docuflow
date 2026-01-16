"""
Enhanced Tree Builder

Combines markdown structure with spatial metadata (bounding boxes, labels)
to build more accurate document hierarchies.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from spatial.hierarchy import (
    classify_elements_with_metadata,
    get_page_dimensions_from_elements,
    cluster_by_spatial_proximity,
)
from core.constants import LABEL_HIERARCHY_WEIGHTS


@dataclass
class TreeNode:
    """Represents a node in the document tree."""
    node_id: str
    title: str
    level: int
    page_number: int
    content: str = ""
    children: List['TreeNode'] = None
    bbox: Optional[Dict] = None  # Bounding box metadata
    label: Optional[str] = None  # Grounding label
    spatial_score: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for storage."""
        return {
            'node_id': self.node_id,
            'title': self.title,
            'level': self.level,
            'page_number': self.page_number,
            'content': self.content,
            'children': [child.to_dict() for child in self.children],
            'bbox': self.bbox,
            'label': self.label,
            'spatial_score': self.spatial_score
        }


def parse_markdown_headers(markdown: str) -> List[Dict]:
    """
    Extract markdown headers with their levels and positions.
    
    Args:
        markdown: Markdown text
    
    Returns:
        List of dicts with header info
    """
    headers = []
    lines = markdown.split('\n')
    
    char_position = 0
    for line_num, line in enumerate(lines, 1):
        # Match markdown headers (# Header)
        match = re.match(r'^(#+)\s+(.+)$', line)
        if match:
            level = len(match.group(1)) - 1  # # = level 0, ## = level 1, etc.
            title = match.group(2).strip()
            
            headers.append({
                'title': title,
                'level': level,
                'line_number': line_num,
                'char_position': char_position,
                'markdown_source': True
            })
        
        char_position += len(line) + 1  # +1 for newline
    
    return headers


def find_spatial_match(
    markdown_header: Dict,
    spatial_elements: List[Dict],
    fuzzy_match: bool = True
) -> Optional[Dict]:
    """
    Find spatial element that corresponds to a markdown header.
    
    Args:
        markdown_header: Header dict from parse_markdown_headers
        spatial_elements: Classified spatial elements
        fuzzy_match: Allow fuzzy text matching
    
    Returns:
        Matching spatial element or None
    """
    header_text = markdown_header['title'].lower().strip()
    
    for elem in spatial_elements:
        elem_text = elem.get('text_content', elem.get('text', '')).lower().strip()
        
        # Exact match
        if elem_text == header_text:
            return elem
        
        # Fuzzy match
        if fuzzy_match and header_text in elem_text or elem_text in header_text:
            # Check if label suggests it's a header
            if elem.get('label', '').lower() in ['title', 'sub_title', 'subtitle', 'heading', 'header']:
                return elem
    
    return None


def discover_implicit_sections(
    layout_elements: List[Dict],
    markdown_headers: List[Dict]
) -> List[Dict]:
    """
    Find sections that have visual structure but lack markdown headers.
    
    Args:
        layout_elements: All layout elements with spatial classification
        markdown_headers: Parsed markdown headers
    
    Returns:
        List of implicit section dicts
    """
    markdown_texts = {h['title'].lower().strip() for h in markdown_headers}
    implicit_sections = []
    
    for elem in layout_elements:
        # Only consider title/heading elements
        if elem.get('label', '').lower() not in ['title', 'sub_title', 'subtitle', 'heading', 'header']:
            continue
        
        elem_text = elem.get('text_content', elem.get('text', '')).strip()
        
        # Skip if already in markdown
        if elem_text.lower() in markdown_texts:
            continue
        
        # This is an implicit section!
        implicit_sections.append({
            'title': elem_text,
            'level': elem.get('predicted_level', 2),
            'page_number': elem.get('page_number', 1),
            'bbox': {
                'x1': elem.get('bbox_x1', elem.get('x1')),
                'y1': elem.get('bbox_y1', elem.get('y1')),
                'x2': elem.get('bbox_x2', elem.get('x2')),
                'y2': elem.get('bbox_y2', elem.get('y2'))
            },
            'label': elem.get('label'),
            'spatial_score': elem.get('spatial_score', 0.0),
            'markdown_source': False,
            'implicit': True
        })
    
    return implicit_sections


def fuse_markdown_and_spatial(
    markdown_headers: List[Dict],
    spatial_elements: List[Dict]
) -> List[Dict]:
    """
    Combine markdown headers with spatial metadata.
    Adjust hierarchy levels based on spatial cues.
    
    Args:
        markdown_headers: Headers from markdown
        spatial_elements: Elements with spatial classification
    
    Returns:
        Fused list of section definitions
    """
    fused_sections = []
    
    for md_header in markdown_headers:
        # Find corresponding spatial element
        spatial_match = find_spatial_match(md_header, spatial_elements)
        
        section = {**md_header}  # Copy markdown info
        
        if spatial_match:
            # Add spatial metadata
            section['bbox'] = {
                'x1': spatial_match.get('bbox_x1', spatial_match.get('x1')),
                'y1': spatial_match.get('bbox_y1', spatial_match.get('y1')),
                'x2': spatial_match.get('bbox_x2', spatial_match.get('x2')),
                'y2': spatial_match.get('bbox_y2', spatial_match.get('y2'))
            }
            section['label'] = spatial_match.get('label')
            section['spatial_score'] = spatial_match.get('spatial_score', 0.0)
            section['page_number'] = spatial_match.get('page_number', 1)
            
            # Validate hierarchy level
            md_level = section['level']
            spatial_level = spatial_match.get('predicted_level', md_level)
            
            # If levels disagree significantly, blend them
            if abs(md_level - spatial_level) > 1:
                # Give more weight to markdown (60/40) but consider spatial
                adjusted_level = int(md_level * 0.6 + spatial_level * 0.4)
                section['level'] = adjusted_level
                section['metadata_adjusted'] = True
        
        fused_sections.append(section)
    
    return fused_sections


def build_tree_from_sections(sections: List[Dict]) -> TreeNode:
    """
    Build hierarchical tree from section definitions.
    
    Args:
        sections: List of section dicts with levels
    
    Returns:
        Root TreeNode
    """
    if not sections:
        return TreeNode(node_id="root", title="Document", level=-1, page_number=1)
    
    # Create root
    root = TreeNode(
        node_id="root",
        title=sections[0].get('title', 'Document') if sections[0].get('level',  0) == 0 else "Document",
        level=-1,
        page_number=1
    )
    
    # Stack to track parent nodes at each level
    stack = [root]
    node_counter = 0
    
    for section in sections:
        level = section.get('level', 0)
        
        # Create node
        node = TreeNode(
            node_id=f"node_{node_counter}",
            title=section.get('title', f'Section {node_counter}'),
            level=level,
            page_number=section.get('page_number', 1),
            content=section.get('content', ''),
            bbox=section.get('bbox'),
            label=section.get('label'),
            spatial_score=section.get('spatial_score', 0.0)
        )
        node_counter += 1
        
        # Find parent (first node in stack with level < current level)
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        
        # Add as child of parent
        parent = stack[-1]
        parent.children.append(node)
        
        # Add to stack
        stack.append(node)
    
    return root


def build_enhanced_tree(
    markdown: str,
    layout_elements: List[Dict],
    use_spatial: bool = True,
    discover_implicit: bool = True,
    spatial_weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Build document tree using both markdown and spatial metadata.
    
    Args:
        markdown: Markdown content
        layout_elements: Layout elements with bounding boxes and labels
        use_spatial: Whether to use spatial metadata
        discover_implicit: Whether to discover implicit sections
        spatial_weights: Optional custom weights for spatial scoring
    
    Returns:
        Tree structure as dict
    """
    # Parse markdown headers
    markdown_headers = parse_markdown_headers(markdown)
    
    if not use_spatial:
        # Build tree from markdown only
        root = build_tree_from_sections(markdown_headers)
        return root.to_dict()
    
    # Get page dimensions from elements
    page_dims = get_page_dimensions_from_elements(layout_elements)
    
    # Classify elements spatially
    spatial_elements = classify_elements_with_metadata(
        layout_elements,
        page_dims,
        weights=spatial_weights
    )
    
    # Fuse markdown and spatial
    fused_sections = fuse_markdown_and_spatial(markdown_headers, spatial_elements)
    
    # Discover implicit sections if enabled
    if discover_implicit:
        implicit_sections = discover_implicit_sections(spatial_elements, markdown_headers)
        # Merge and sort by page + vertical position
        all_sections = fused_sections + implicit_sections
        all_sections.sort(key=lambda s: (
            s.get('page_number', 1),
            s.get('bbox', {}).get('y1', 0) if s.get('bbox') else 0
        ))
    else:
        all_sections = fused_sections
    
    # Build tree
    root = build_tree_from_sections(all_sections)
    
    return root.to_dict()


def add_content_to_tree(tree: Dict, markdown: str, layout_elements: List[Dict]) -> Dict:
    """
    Add content text to tree nodes based on their position in document.
    
    Args:
        tree: Tree structure
        markdown: Full markdown content
        layout_elements: Layout elements
    
    Returns:
        Tree with content added to nodes
    """
    # TODO: Implement content assignment
    # For now, return tree as-is
    return tree
