"""Utilities package - Helper functions for image, bbox, and text processing."""

from .bbox_utils import (
    draw_bounding_boxes,
    extract_grounding_references,
    extract_header_text,
    extract_layout_coordinates_v2,
)
from .image_utils import (
    decode_base64_image,
    image_to_base64,
    render_pdf_page_to_base64,
)
from .text_utils import (
    clean_grounding_format,
)

__all__ = [
    # Image utils
    "render_pdf_page_to_base64",
    "image_to_base64",
    "decode_base64_image",
    # BBox utils
    "extract_grounding_references",
    "extract_layout_coordinates_v2",
    "draw_bounding_boxes",
    "extract_header_text",
    # Text utils
    "clean_grounding_format",
]
