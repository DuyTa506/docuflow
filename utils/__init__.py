"""Utilities package - Helper functions for image, bbox, and text processing."""

from .image_utils import (
    render_pdf_page_to_base64,
    image_to_base64,
    decode_base64_image,
    get_image_dimensions
)

from .bbox_utils import (
    extract_grounding_references,
    extract_layout_coordinates,
    extract_layout_coordinates_v2,  # NEW
    draw_bounding_boxes,
    normalize_bbox,
    denormalize_bbox,
    extract_header_text,  # NEW
)

from .text_utils import (
    clean_grounding_format,
    extract_text_from_grounding,
    strip_markdown_headers
)

__all__ = [
    # Image utils
    'render_pdf_page_to_base64',
    'image_to_base64',
    'decode_base64_image',
    'get_image_dimensions',
    
    # BBox utils
    'extract_grounding_references',
    'extract_layout_coordinates',
    'draw_bounding_boxes',
    'normalize_bbox',
    'denormalize_bbox',
    
    # Text utils
    'clean_grounding_format',
    'extract_text_from_grounding',
    'strip_markdown_headers'
]
