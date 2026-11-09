"""
Enhanced Tree Builder

Combines markdown structure with spatial metadata (bounding boxes, labels)
to build more accurate document hierarchies.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TreeNode:
    """Represents a node in the document tree."""

    node_id: str
    title: str
    level: int
    page_number: int
    content: str = ""
    children: List["TreeNode"] = None
    bbox: Optional[Dict] = None  # Bounding box metadata
    label: Optional[str] = None  # Grounding label
    spatial_score: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def to_dict(self) -> Dict:
        """Convert to dictionary format for storage."""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "page_number": self.page_number,
            "content": self.content,
            "children": [child.to_dict() for child in self.children],
            "bbox": self.bbox,
            "label": self.label,
            "spatial_score": self.spatial_score,
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
    lines = markdown.split("\n")

    char_position = 0
    for line_num, line in enumerate(lines, 1):
        # Match markdown headers (# Header)
        match = re.match(r"^(#+)\s+(.+)$", line)
        if match:
            level = len(match.group(1)) - 1  # # = level 0, ## = level 1, etc.
            title = match.group(2).strip()

            headers.append(
                {
                    "title": title,
                    "level": level,
                    "line_number": line_num,
                    "char_position": char_position,
                    "markdown_source": True,
                }
            )

        char_position += len(line) + 1  # +1 for newline

    return headers


def calculate_bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bbox with x1, y1, x2, y2
        bbox2: Second bbox with x1, y1, x2, y2

    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1 = bbox1.get("x1", bbox1.get("bbox_x1", 0))
    y1_1 = bbox1.get("y1", bbox1.get("bbox_y1", 0))
    x2_1 = bbox1.get("x2", bbox1.get("bbox_x2", 0))
    y2_1 = bbox1.get("y2", bbox1.get("bbox_y2", 0))

    x1_2 = bbox2.get("x1", bbox2.get("bbox_x1", 0))
    y1_2 = bbox2.get("y1", bbox2.get("bbox_y1", 0))
    x2_2 = bbox2.get("x2", bbox2.get("bbox_x2", 0))
    y2_2 = bbox2.get("y2", bbox2.get("bbox_y2", 0))

    # Calculate intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union
