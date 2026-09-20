"""Remove source glyphs (native PDF) or inpaint OCR rasters before redraw."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Iterable, Optional

from core.pdf_render.geometry import Rect, Region

logger = logging.getLogger(__name__)

_INPAINT_PAD_PX = 2
_INPAINT_PAD_HEIGHT_RATIO = 0.08
_RESERVED_COVER = 0.45


def _span_in_reserved(span: Rect, reserved: list[Rect]) -> bool:
    return any(span.intersection_over_self(r) >= _RESERVED_COVER for r in reserved)


def redact_native_text(page, translatable: Iterable[Rect], reserved: Iterable[Rect]) -> int:
    """Delete source glyphs whose boxes hit translatable regions.

    Images and vector graphics are preserved (``PDF_REDACT_IMAGE_NONE`` /
    ``PDF_REDACT_LINE_ART_NONE``). No white fill is painted — the original
    page background stays intact.
    """
    import fitz

    reserved_list = list(reserved)
    targets = list(translatable)
    if not targets:
        return 0
    count = 0
    try:
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    except Exception:
        blocks = page.get_text("dict").get("blocks", [])
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                bbox = span.get("bbox")
                if not bbox:
                    continue
                span_rect = Rect(*bbox)
                if _span_in_reserved(span_rect, reserved_list):
                    continue
                if any(span_rect.intersects(t) for t in targets):
                    # PyMuPDF accepts objects with x0/y0/x1/y1 as annot
                    # geometry but apply_redactions only honors fitz.Rect.
                    page.add_redact_annot(span_rect.to_fitz())
                    count += 1
    if count:
        kwargs = {}
        if hasattr(fitz, "PDF_REDACT_IMAGE_NONE"):
            kwargs["images"] = fitz.PDF_REDACT_IMAGE_NONE
        if hasattr(fitz, "PDF_REDACT_LINE_ART_NONE"):
            kwargs["graphics"] = fitz.PDF_REDACT_LINE_ART_NONE
        if hasattr(fitz, "PDF_REDACT_TEXT_REMOVE"):
            kwargs["text"] = fitz.PDF_REDACT_TEXT_REMOVE
        page.apply_redactions(**kwargs)
    return count


def _pad_px_for_rect(rect: Rect, *, sy: float) -> int:
    """Pad scales with region height so descenders outside the OCR bbox clear."""
    height_px = abs(rect.y1 - rect.y0) * sy
    return max(_INPAINT_PAD_PX, int(_INPAINT_PAD_HEIGHT_RATIO * height_px))


def inpaint_scan_image(
    image_bytes: bytes,
    translatable: list[Rect],
    reserved: list[Rect],
    *,
    page_w: float,
    page_h: float,
) -> Optional[bytes]:
    """Inpaint glyph regions on a scanned page raster. Best-effort."""
    if not image_bytes or not translatable:
        return None
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:
        logger.debug("scan inpaint skipped: opencv/PIL unavailable", exc_info=True)
        return None

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    arr = np.array(img)
    h, w = arr.shape[:2]
    sx = w / max(page_w, 1.0)
    sy = h / max(page_h, 1.0)
    mask = np.zeros((h, w), dtype=np.uint8)

    def _px(rect: Rect) -> tuple[int, int, int, int]:
        pad = _pad_px_for_rect(rect, sy=sy)
        x0 = int(max(0, rect.x0 * sx) - pad)
        y0 = int(max(0, rect.y0 * sy) - pad)
        x1 = int(min(w, rect.x1 * sx) + pad)
        y1 = int(min(h, rect.y1 * sy) + pad)
        return x0, y0, x1, y1

    for rect in translatable:
        x0, y0, x1, y1 = _px(rect)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    for rect in reserved:
        x0, y0, x1, y1 = _px(rect)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0

    if int(mask.sum()) == 0:
        return image_bytes
    try:
        inpainted = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        logger.debug("cv2.inpaint failed", exc_info=True)
        return None
    from utils.image_utils import encode_scan_jpeg

    return encode_scan_jpeg(Image.fromarray(inpainted), quality=90)


def translatable_and_reserved(
    regions: list[Region],
    *,
    include_tables: bool = False,
) -> tuple[list[Rect], list[Rect]]:
    """Split regions into inpaint/redact targets vs protected ink.

    When ``include_tables`` is True (translation layout on scans), table
    interiors are inpainted so cell text can be redrawn without the source
    glyphs showing through. Figures and equations stay reserved.
    """
    trans: list[Rect] = []
    reserved: list[Rect] = []
    for r in regions:
        if r.role in {"figure", "equation"}:
            reserved.append(r.bbox)
            continue
        if r.role == "table":
            if include_tables:
                trans.append(r.bbox)
            else:
                reserved.append(r.bbox)
            continue
        if r.role == "vertical" or r.passthrough:
            continue
        trans.append(r.bbox)
    return trans, reserved
