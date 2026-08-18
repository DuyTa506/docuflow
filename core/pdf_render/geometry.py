"""Normalized PDF-point geometry for the hybrid layout renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

RENDERER_VERSION = "v2"

# Stored OCR/Docling bboxes live in the page-image pixel space. Text-layer
# pages are rasterized at 72 DPI so 1 px ≈ 1 PDF point; OCR rasters are
# downscaled and must be scaled onto the real page rectangle.
DEFAULT_PAGE_WIDTH = 595.0
DEFAULT_PAGE_HEIGHT = 842.0


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "Rect":
        return cls(float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2)))

    @classmethod
    def from_bbox_dict(cls, bbox: dict | None) -> "Rect":
        bbox = bbox or {}
        return cls.from_xyxy(
            bbox.get("x1", bbox.get("x0", 0)) or 0,
            bbox.get("y1", bbox.get("y0", 0)) or 0,
            bbox.get("x2", bbox.get("x1", 0)) or 0,
            bbox.get("y2", bbox.get("y1", 0)) or 0,
        )

    @classmethod
    def from_xyxy_attrs(cls, obj) -> "Rect":
        return cls.from_xyxy(
            getattr(obj, "bbox_x1", 0) or 0,
            getattr(obj, "bbox_y1", 0) or 0,
            getattr(obj, "bbox_x2", 0) or 0,
            getattr(obj, "bbox_y2", 0) or 0,
        )

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def clamp(self, page_w: float, page_h: float) -> "Rect":
        x0 = max(0.0, min(self.x0, page_w))
        y0 = max(0.0, min(self.y0, page_h))
        x1 = max(x0 + 0.5, min(self.x1, page_w))
        y1 = max(y0 + 0.5, min(self.y1, page_h))
        return Rect(x0, y0, x1, y1)

    def expand(self, pad: float) -> "Rect":
        return Rect(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def intersect(self, other: "Rect") -> "Rect":
        return Rect(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    def intersects(self, other: "Rect", *, eps: float = 0.5) -> bool:
        return (
            self.x0 < other.x1 - eps
            and other.x0 < self.x1 - eps
            and self.y0 < other.y1 - eps
            and other.y0 < self.y1 - eps
        )

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersection_over_self(self, other: "Rect") -> float:
        inter = self.intersect(other)
        if self.area <= 0:
            return 0.0
        return max(0.0, inter.area) / self.area

    def x_overlap_ratio(self, other: "Rect") -> float:
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        if overlap <= 0:
            return 0.0
        base = min(self.width, other.width)
        return overlap / base if base > 0 else 0.0

    def y_overlap_ratio(self, other: "Rect") -> float:
        overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        if overlap <= 0:
            return 0.0
        base = min(self.height, other.height)
        return overlap / base if base > 0 else 0.0

    def to_bbox_dict(self) -> dict:
        return {"x1": self.x0, "y1": self.y0, "x2": self.x1, "y2": self.y1}

    def to_fitz(self):
        import fitz

        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)


def stored_to_page_rect(
    bbox: Rect,
    *,
    image_w: float,
    image_h: float,
    page_w: float,
    page_h: float,
    rotation: int = 0,
) -> Rect:
    """Map a stored layout bbox (image pixels) onto the visual PDF page.

    ``rotation`` is the page ``/Rotate`` value. Stored images from this
    pipeline are already visually upright (PyMuPDF pixmap / Docling 72 DPI
    raster), so we scale into ``page.rect`` after swapping axes for 90/270.
    """
    iw = float(image_w) or page_w or DEFAULT_PAGE_WIDTH
    ih = float(image_h) or page_h or DEFAULT_PAGE_HEIGHT
    rot = int(rotation or 0) % 360
    vis_w, vis_h = page_w, page_h
    if rot in (90, 270):
        vis_w, vis_h = page_h, page_w
    sx = vis_w / iw
    sy = vis_h / ih
    mapped = Rect(bbox.x0 * sx, bbox.y0 * sy, bbox.x1 * sx, bbox.y1 * sy)
    if rot == 90:
        # stored (x,y) is visual; page.rect after MuPDF rotation already matches
        mapped = Rect(mapped.x0, mapped.y0, mapped.x1, mapped.y1)
    elif rot == 180:
        mapped = Rect(vis_w - mapped.x1, vis_h - mapped.y1, vis_w - mapped.x0, vis_h - mapped.y0)
    elif rot == 270:
        mapped = Rect(mapped.x0, mapped.y0, mapped.x1, mapped.y1)
    return mapped.clamp(page_w, page_h)


def page_rect_from_fitz(page) -> tuple[float, float, int]:
    """Return (width, height, rotation) for a PyMuPDF page in visual space."""
    rect = page.rect
    return float(rect.width), float(rect.height), int(page.rotation or 0)


@dataclass
class ColumnBand:
    index: int
    x0: float
    x1: float
    full_width: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)


@dataclass
class Region:
    id: str
    label: str
    role: str
    text: str
    bbox: Rect
    column_index: int = 0
    full_width: bool = False
    passthrough: bool = False
    source: str = "unknown"
    crop_image_key: Optional[str] = None
    crop_image_base64: Optional[str] = None
    sequence_order: int = 0
    page_number: int = 1
    extra: dict = field(default_factory=dict)


@dataclass
class PageMeta:
    page_number: int
    width: float
    height: float
    page_type: str = "text"
    rotation: int = 0
    image_width: Optional[float] = None
    image_height: Optional[float] = None
    image_key: Optional[str] = None
    image_bytes: Optional[bytes] = None

    @property
    def storage_width(self) -> float:
        return float(self.image_width or self.width or DEFAULT_PAGE_WIDTH)

    @property
    def storage_height(self) -> float:
        return float(self.image_height or self.height or DEFAULT_PAGE_HEIGHT)


@dataclass
class PageScene:
    meta: PageMeta
    regions: list[Region] = field(default_factory=list)
    columns: list[ColumnBand] = field(default_factory=list)

    def translatable(self) -> list[Region]:
        return [r for r in self.regions if not r.passthrough and (r.text or "").strip()]

    def reserved(self) -> list[Region]:
        return [r for r in self.regions if r.passthrough]


def union_rects(rects: Sequence[Rect]) -> Optional[Rect]:
    if not rects:
        return None
    acc = rects[0]
    for r in rects[1:]:
        acc = acc.union(r)
    return acc


def any_intersects(rect: Rect, others: Iterable[Rect], *, eps: float = 0.5) -> bool:
    return any(rect.intersects(o, eps=eps) for o in others)
