"""
Core domain models for OCR workflow.

These are pure data structures without business logic.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ServicePageResult:
    """Result from processing a single page with OCR."""
    page_num: int
    markdown: str
    input_tokens: int = 0
    output_tokens: int = 0
    image_base64: str = ""
    annotated_image_base64: str = ""
    layout_elements: List[dict] = field(default_factory=list)
    crops_base64: List[str] = field(default_factory=list)


@dataclass
class LayoutElement:
    """A detected layout element with bounding box coordinates."""
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""
    crop_image: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'text': self.text,
            'crop_image': self.crop_image
        }


@dataclass
class UnifiedElement:
    """
    Unified intermediate element produced by every extractor
    (DOCX, PDF-text, OCR).  All converge to this format before being
    fed into build_spatial_tree() via to_layout_element_dict().
    """
    element_type: str           # "heading" | "text" | "table" | "figure" | ...
    text: str                   # Content
    page_number: int
    order: int                  # Reading order within page
    source: str                 # "ocr" | "pdf_text" | "docx"
    level: Optional[int] = None         # Heading level 1-6
    bbox: Optional[Dict] = None         # {x1, y1, x2, y2} – available for OCR + PDF text
    font_size: Optional[float] = None   # PDF text only
    style_name: Optional[str] = None    # DOCX only
    image_bytes_b64: Optional[str] = None  # JPEG base64 of embedded image (pdf_text path only)
    image_width: Optional[int] = None       # pixel width of embedded image
    image_height: Optional[int] = None      # pixel height of embedded image

    def to_layout_element_dict(self) -> dict:
        """Convert to dict format build_spatial_tree() expects."""
        if self.element_type in ("image", "figure"):
            label = "figure"
        elif self.element_type == "table":
            label = "table"
        elif self.element_type == "equation":
            label = "equation"
        elif self.level == 1:
            label = "title"
        elif self.level == 2:
            label = "sub_title"
        elif self.level:
            label = "heading"
        else:
            label = "text"

        bbox = self.bbox or {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        out = {
            'label': label,
            'bbox_x1': bbox['x1'],
            'bbox_y1': bbox['y1'],
            'bbox_x2': bbox['x2'],
            'bbox_y2': bbox['y2'],
            'text_content': self.text[:200],
            'text_full': self.text,
            'page_number': self.page_number,
            'heading_level': self.level,
            'source': self.source,
        }
        # Embedded images (PDF text-layer / DOCX): preserve pixels for spatial export.
        if self.image_bytes_b64:
            out['crop_image'] = self.image_bytes_b64
        return out


@dataclass
class BoundingBox:
    """Bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        """Calculate width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Calculate height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Calculate area."""
        return self.width * self.height
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        }
