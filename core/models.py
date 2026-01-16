"""
Core domain models for OCR workflow.

These are pure data structures without business logic.
"""
from dataclasses import dataclass, field
from typing import List, Optional


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
