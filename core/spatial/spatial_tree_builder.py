"""
Spatial-First Tree Builder

NEW: This module provides the spatial-first pipeline where layout_elements
are enriched with full text content from raw OCR output.

Markdown syntax is used only for optional validation, not as primary source.
"""
import re
from typing import List, Dict, Optional
from core.spatial.tree_builder import TreeNode

# This file will be added to tree_builder.py
# Adding here as a placeholder for the new functions

_HEADINGISH_LABELS = frozenset({"title", "sub_title", "subtitle", "heading", "section_heading"})

# "1 Introduction" -> "1"; "3.1 Problem Formulation" -> "3.1"; "A.1 Schema Selector" -> "A.1".
# Deliberately requires a bare digit run OR letter+dot+digit -- a bare single letter
# ("A Detailed Algorithms") is excluded to avoid matching ordinary titles that just
# happen to start with a capital letter and a space.
_NUMBERED_SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*|[A-Z]\.\d+(?:\.\d+)*)\s+\S')


def extract_markdown_level(text: str) -> Optional[int]:
    """
    Extract hierarchy level from markdown syntax if present.

    Args:
        text: Text that may start with markdown headers

    Returns:
        Level (0-5) or None if no markdown syntax
    """
    import re

    # Match markdown headers: # Title, ## Subtitle, etc.
    match = re.match(r'^(#{1,6})\s+', text.strip())

    if match:
        hashes = len(match.group(1))
        return hashes - 1  # # = level 0, ## = level 1, etc.

    return None


def extract_numbered_section_level(text: str, *, max_len: int = 150) -> Optional[int]:
    """Extract hierarchy depth from an explicit numbered-section prefix
    (e.g. "1 Introduction" -> 1, "3.1 Problem Formulation" -> 2,
    "A.1 Schema Selector" -> 2). Returns None if no such prefix is present,
    or the text is too long to plausibly be a heading.

    This is a far more reliable signal than bbox geometry for numbered
    academic papers/textbooks: confirmed live on a real 2-column paper
    that a subsection's raw spatial score can exceed its own parent
    section's score by a razor-thin, essentially noise-level margin
    (0.0004), which adaptive percentile thresholds then amplify into a
    structurally wrong tree (the subsection ends up shallower than its
    own parent). Section numbering in the text itself doesn't have this
    ambiguity.
    """
    stripped = text.strip()
    if len(stripped) > max_len:
        return None
    match = _NUMBERED_SECTION_RE.match(stripped)
    if not match:
        return None
    return match.group(1).count('.') + 1


def validate_with_markdown_syntax(elements: List[Dict]) -> List[Dict]:
    """
    Optional: Cross-check spatial hierarchy with markdown syntax or an
    explicit numbered-section prefix.

    Priority: numbered-section prefix (most reliable for heading-labeled
    elements) > markdown "#" syntax > spatial-only.

    Args:
        elements: Elements with 'spatial_level' already predicted

    Returns:
        Elements with 'final_level' set
    """
    for elem in elements:
        text = elem.get('text_content', '')
        spatial_level = elem.get('spatial_level', 3)
        label = (elem.get('label') or '').lower()

        numbered_level = (
            extract_numbered_section_level(text) if label in _HEADINGISH_LABELS else None
        )
        if numbered_level is not None:
            elem['final_level'] = numbered_level
            elem['level_source'] = 'numbered_section'
            continue

        # Check for markdown syntax
        md_level = extract_markdown_level(text)

        if md_level is not None:
            # Blend if disagreement
            if abs(md_level - spatial_level) > 1:
                blended = int(md_level * 0.5 + spatial_level * 0.5)
                elem['final_level'] = blended
                elem['level_source'] = 'blended'
                elem['markdown_level'] = md_level
            else:
                elem['final_level'] = spatial_level
                elem['level_source'] = 'spatial_validated'
                elem['markdown_level'] = md_level
        else:
            elem['final_level'] = spatial_level
            elem['level_source'] = 'spatial_only'

    return elements


def predict_hierarchy_spatial(
    elements: List[Dict],
    page_dims: Dict[str, int],
    spatial_weights: Optional[Dict] = None
) -> List[Dict]:
    """
    Predict hierarchy purely from spatial features.
    No markdown dependency.
    
    Args:
        elements: Layout elements with bbox and text
        page_dims: Page dimensions
        spatial_weights: Optional weights
    
    Returns:
        Elements with 'spatial_level' added
    """
    from core.spatial.hierarchy import predict_hierarchy_level
    from core.spatial.grouping import estimate_median_line_height
    
    if not elements:
        return elements
    
    median_height = estimate_median_line_height(elements)
    page_width = page_dims.get('width', 800)
    page_height = page_dims.get('height', 1000)
    
    for i, elem in enumerate(elements):
        prev = elements[i-1] if i > 0 else None
        next_elem = elements[i+1] if i < len(elements) - 1 else None
        
        level = predict_hierarchy_level(
            elem,
            page_width,
            page_height,
            weights=spatial_weights,
            prev_element=prev,
            next_element=next_elem,
            median_line_height=median_height
        )
        
        elem['spatial_level'] = level
        elem['predicted_level'] = level  # Backward compat
    
    return elements


def build_tree_from_elements(elements: List[Dict]) -> TreeNode:
    """
    Build tree directly from spatial elements.

    Args:
        elements: Ordered elements with 'final_level'

    Returns:
        Tree root
    """
    if not elements:
        return TreeNode(
            node_id="root",
            title="Document",
            level=-1,
            page_number=1
        )

    # ── Preface fix ────────────────────────────────────────────────
    # If leading elements are body text (no heading level), collect them
    # into a synthetic "Preface" node so they are preserved in the tree.
    preface_parts = []
    first_heading_idx = 0
    for idx, elem in enumerate(elements):
        lvl = elem.get('final_level', elem.get('spatial_level', 5))
        if lvl <= 4:  # genuine heading (levels 1-4)
            first_heading_idx = idx
            break
        else:
            preface_parts.append(elem)
            first_heading_idx = idx + 1  # will stay past end if all body

    if preface_parts and first_heading_idx > 0:
        # Build a synthetic element for the preface
        preface_text = "\n\n".join(
            e.get('text_full') or e.get('text_content', '') for e in preface_parts
        ).strip()
        if preface_text:
            synthetic = {
                'text_content': 'Preface',
                'text_full': preface_text,
                'final_level': 1,
                'spatial_level': 1,
                'level_source': 'preface_synthetic',
                'label': 'title',
                'page_number': preface_parts[0].get('page_number', 1),
                'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0,
                'is_synthetic': True,
            }
            elements = [synthetic] + elements[first_heading_idx:]
    # ──────────────────────────────────────────────────────────────

    # Create root
    root = TreeNode(
        node_id="root",
        title="Document",
        level=-1,
        page_number=1
    )
    
    stack = [root]
    node_counter = 0
    
    for elem in elements:
        level = elem.get('final_level', elem.get('spatial_level', 3))
        title = elem.get('text_content', f'Section {node_counter}')
        
        # Create node
        node = TreeNode(
            node_id=f"node_{node_counter}",
            title=title,
            level=level,
            page_number=elem.get('page_number', 1),
            content=elem.get('text_full', ''),
            bbox={
                'x1': elem.get('bbox_x1', elem.get('x1', 0)),
                'y1': elem.get('bbox_y1', elem.get('y1', 0)),
                'x2': elem.get('bbox_x2', elem.get('x2', 0)),
                'y2': elem.get('bbox_y2', elem.get('y2', 0))
            },
            label=elem.get('label'),
            spatial_score=elem.get('spatial_score', 0.0)
        )
        node_counter += 1
        
        # Find parent
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        
        parent = stack[-1]
        parent.children.append(node)
        stack.append(node)
    
    return root


def _is_docx_elements(elements: List[Dict]) -> bool:
    """
    Detect whether all elements come from the DOCX extractor (all bboxes zero).
    This is the signal to skip spatial phases and use heading_level directly.
    """
    if not elements:
        return False
    return not any(
        elem.get('bbox_x1', 0) or elem.get('bbox_y1', 0) or
        elem.get('bbox_x2', 0) or elem.get('bbox_y2', 0)
        for elem in elements
    )


def _build_tree_docx_fastpath(elements: List[Dict]) -> Dict:
    """
    Fast-path tree build for DOCX elements: use heading_level directly
    as the final level.  Skips all spatial scoring phases 1-6.
    """
    # Map DOCX label to numeric level; body text becomes level 5 (leaf)
    _LABEL_LEVEL = {
        "title": 1,
        "sub_title": 2,
        "heading": 3,
        "text": 5,
        "table": 5,
        "figure": 5,
    }

    for elem in elements:
        heading_level = elem.get('heading_level')
        if heading_level is not None:
            elem['final_level'] = heading_level
        else:
            label = elem.get('label', 'text')
            elem['final_level'] = _LABEL_LEVEL.get(label, 5)
        elem['level_source'] = 'docx_native'

    root = build_tree_from_elements(elements)
    result = root.to_dict()
    result['_pipeline_info'] = {
        'version': 'docx_fastpath_v1',
        'filters_applied': False,
        'zone_classification': False,
        'reading_order': False,
        'markdown_validation': False,
        'adaptive_thresholds': False,
        'thinning_applied': False,
        'thinning_stats': {},
        'elements_processed': len(elements),
    }
    return result


def build_spatial_tree(
    layout_elements: List[Dict],
    use_filters: bool = True,
    use_zone_classification: bool = True,
    use_reading_order: bool = True,
    use_markdown_validation: bool = True,
    use_adaptive_thresholds: bool = True,
    use_thinning: bool = False,
    thinning_gap_multiplier: float = 2.0,
    spatial_weights: Optional[Dict] = None
) -> Dict:
    """
    Build tree from spatial elements only (SPATIAL-FIRST).
    
    This is the new recommended API for tree building.
    Layout elements must be enriched with text_content (use extract_layout_coordinates_v2).
    
    Markdown validation is OPTIONAL - used only to cross-check spatial predictions.
    
    Args:
        layout_elements: Elements with bbox + text_content + text_full
        use_filters: Apply preprocessing filters
        use_zone_classification: Classify zones
        use_reading_order: Use topological sort
        use_markdown_validation: Validate with markdown syntax (optional)
        use_adaptive_thresholds: Use adaptive calibration
        use_thinning: Apply hierarchical thinning (paragraph merging)
        thinning_gap_multiplier: Gap threshold multiplier for thinning
        spatial_weights: Custom scoring weights
    
    Returns:
        Document tree with metadata
    """
    from core.spatial.hierarchy import (
        get_page_dimensions_from_elements,
        calculate_adaptive_thresholds,
    )
    
    if not layout_elements:
        root = TreeNode(node_id="root", title="Document", level=-1, page_number=1)
        return root.to_dict()

    # ── DOCX fast-path ──────────────────────────────────────────────
    # When all bboxes are zero (DOCX elements have no spatial data),
    # skip the 7-phase spatial pipeline and use heading_level directly.
    if _is_docx_elements(layout_elements):
        return _build_tree_docx_fastpath(layout_elements)

    # Get page dimensions
    page_dims = get_page_dimensions_from_elements(layout_elements)
    current_elements = layout_elements.copy()
    
    filter_stats = {}
    
    # Phase 1: Preprocessing
    if use_filters:
        try:
            from core.spatial.filters import apply_all_filters
            current_elements, removed = apply_all_filters(
                current_elements,
                filter_repeated=True,
                filter_noise=True,
                filter_margins=False
            )
            filter_stats = {k: len(v) for k, v in removed.items()}
        except ImportError:
            pass
    
    # Phase 2: Zone classification
    if use_zone_classification:
        try:
            from core.spatial.zone_classifier import classify_zones_batch
            from core.spatial.filters import analyze_cross_page_repetitions
            
            cross_page_stats = {}
            try:
                repetitions = analyze_cross_page_repetitions(layout_elements)
                cross_page_stats = {k: v for k, v in repetitions.items()}
            except:
                pass
            
            current_elements = classify_zones_batch(
                current_elements,
                page_dims,
                cross_page_stats
            )
        except ImportError:
            pass
    
    # Phase 3: Reading order (per-page to avoid O(n²) on full documents)
    if use_reading_order:
        try:
            from core.spatial.reading_order import (
                flatten_reading_order,
                get_reading_order_by_page,
            )
            pages_order = get_reading_order_by_page(
                current_elements,
                include_zone_priority=True,
            )
            current_elements = flatten_reading_order(pages_order)
        except ImportError:
            # Fallback: simple y-sort
            current_elements = sorted(
                current_elements,
                key=lambda e: (
                    e.get('page_number', 1),
                    e.get('bbox_y1', e.get('y1', 0))
                )
            )
    
    # Phase 4: Hierarchy prediction (spatial-based)
    current_elements = predict_hierarchy_spatial(
        current_elements,
        page_dims,
        spatial_weights
    )

    # Phase 5: Adaptive thresholds — compute from score distribution and re-classify
    if use_adaptive_thresholds:
        thresholds = calculate_adaptive_thresholds(current_elements)
        # Apply thresholds: re-map spatial_score → spatial_level using calibrated cuts
        for elem in current_elements:
            score = elem.get('spatial_score', 0.5)
            if score > thresholds.get(0, 0.8):
                elem['spatial_level'] = 0
            elif score > thresholds.get(1, 0.6):
                elem['spatial_level'] = 1
            elif score > thresholds.get(2, 0.4):
                elem['spatial_level'] = 2
            elif score > thresholds.get(3, 0.25):
                elem['spatial_level'] = 3
            elif score > thresholds.get(4, 0.15):
                elem['spatial_level'] = 4
            else:
                elem['spatial_level'] = 5
    
    # Phase 6: Optional markdown validation
    if use_markdown_validation:
        current_elements = validate_with_markdown_syntax(current_elements)
    else:
        # Set final_level = spatial_level
        for elem in current_elements:
            elem['final_level'] = elem.get('spatial_level', 3)
            elem['level_source'] = 'spatial_only'
    
    # Phase 6.5: Hierarchical thinning (NEW)
    thinning_stats = {}
    if use_thinning:
        from core.spatial.thinning import hierarchical_thinning
        
        nodes_before = len(current_elements)
        current_elements = hierarchical_thinning(
            current_elements,
            preserve_barriers=True,
            merge_text_to_paragraphs=True,
            gap_threshold_multiplier=thinning_gap_multiplier,
            min_paragraph_tokens=50
        )
        nodes_after = len(current_elements)
        
        thinning_stats = {
            'nodes_before': nodes_before,
            'nodes_after': nodes_after,
            'reduction': nodes_before - nodes_after,
            'reduction_percent': round((nodes_before - nodes_after) / nodes_before * 100, 1) if nodes_before > 0 else 0
        }
    
    # Phase 7: Build tree
    root = build_tree_from_elements(current_elements)
    
    # Add metadata
    result = root.to_dict()
    result['_pipeline_info'] = {
        'version': 'spatial_v1',
        'filters_applied': use_filters,
        'filter_stats': filter_stats,
        'zone_classification': use_zone_classification,
        'reading_order': use_reading_order,
        'markdown_validation': use_markdown_validation,
        'adaptive_thresholds': use_adaptive_thresholds,
        'thinning_applied': use_thinning,  # NEW
        'thinning_stats': thinning_stats if use_thinning else {},  # NEW
        'elements_processed': len(current_elements)
    }
    
    return result
