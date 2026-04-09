"""Utilities package - Helper functions for image, bbox, and text processing."""

from .image_utils import (
    render_pdf_page_to_base64,
    image_to_base64,
    decode_base64_image,
)

from .bbox_utils import (
    extract_grounding_references,
    extract_layout_coordinates_v2,
    draw_bounding_boxes,
    extract_header_text,
)

from .text_utils import (
    clean_grounding_format,
)

__all__ = [
    # Image utils
    'render_pdf_page_to_base64',
    'image_to_base64',
    'decode_base64_image',

    # BBox utils
    'extract_grounding_references',
    'extract_layout_coordinates_v2',
    'draw_bounding_boxes',
    'extract_header_text',

    # Text utils
    'clean_grounding_format',
]
