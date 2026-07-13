"""
Layout-faithful PDF export: one source page → one PDF page, elements at bbox coordinates.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Any, Iterable, List, Literal, Optional

import fitz

from core.constants import OCR_EQUATION_LABELS
from utils.table_grid import build_table_grid, compact_empty_columns, table_text_to_cell_rows

_MIN_FONT_PT = 6.0
_MIN_FONT_PT_REPLACE = 5.0
_TABLE_IMAGE_FALLBACK = True
_TEXT_OVERLAY_GAP = 4.0

TextOverlayMode = Literal["skip", "replace"]

logger = logging.getLogger(__name__)

_IMAGE_LABELS = frozenset({"image", "figure", "chart", "graph", "picture"})
_HEADING_LABELS = frozenset({"title", "sub_title", "heading"})
# When a page scan is the background, skip re-drawing body text (already on scan).
_BACKGROUND_SKIP_LABELS = frozenset(
    {
        "text",
        "main_text",
        "section_heading",
        "abstract",
        "reference",
        "footer",
        "header",
        "list_item",
        "caption",
        "title",
        "sub_title",
        "heading",
    }
)
# Text whose bbox sits mostly inside a figure/chart must not be white-masked
# or redrawn on top of the image (axis labels, diagram callouts).
_FIGURE_INTERIOR_IOU = 0.45
_MIN_TEXTBOX_WIDTH_PT = 14.0
_IMG_PLACEHOLDER_RE = re.compile(
    r"^\(img_content\)|^\[?image[_\s]?\d*\]?$|^\[figure", re.IGNORECASE
)
_STRIP_HTML_RE = re.compile(r"<[^>]+>")

_font_path_cache: Optional[str] = None


def _overlay_pad() -> float:
    try:
        from config.settings import settings

        return float(settings.layout_pdf_text_overlay_pad)
    except Exception:
        return 1.5


def _expand_ratio() -> float:
    try:
        from config.settings import settings

        return float(settings.layout_pdf_text_expand_ratio)
    except Exception:
        return 0.8


def _pages_have_images(pages: Iterable[Any]) -> bool:
    for p in pages:
        if isinstance(p, dict):
            if p.get("image_key"):
                return True
        elif getattr(p, "image_key", None):
            return True
    return False


def _enrich_export_dict(elem: dict, *, document_id: Optional[str], page_number: int) -> dict:
    """Ensure dict payloads carry page_image_key for bbox crops."""
    from utils.storage_keys import page_image_key as default_page_image_key

    out = dict(elem)
    if document_id and not out.get("page_image_key"):
        out["page_image_key"] = default_page_image_key(document_id, page_number)
    bbox = out.get("bbox")
    if bbox and not isinstance(bbox, dict):
        out["bbox"] = dict(bbox)
    return out


def _resolve_font_path() -> Optional[str]:
    global _font_path_cache
    if _font_path_cache is not None:
        return _font_path_cache or None
    try:
        from babeldoc.assets.assets import get_font_and_metadata

        path, _meta = get_font_and_metadata("GoNotoKurrent-Regular.ttf")
        _font_path_cache = path.as_posix()
        return _font_path_cache
    except Exception:
        _font_path_cache = ""
        return None


def _element_dict(elem: Any, *, page_number: int) -> dict:
    if isinstance(elem, dict):
        bbox = elem.get("bbox") or {}
        return {
            "id": elem.get("id") or f"p{page_number}_{elem.get('sequence_order', 0)}",
            "page_number": page_number,
            "label": (elem.get("label") or "text").lower(),
            "text_content": elem.get("text_content") or "",
            "bbox_x1": bbox.get("x1", elem.get("bbox_x1", 0)),
            "bbox_y1": bbox.get("y1", elem.get("bbox_y1", 0)),
            "bbox_x2": bbox.get("x2", elem.get("bbox_x2", 0)),
            "bbox_y2": bbox.get("y2", elem.get("bbox_y2", 0)),
            "_view": elem,
        }
    return {
        "id": getattr(elem, "id", None) or f"p{page_number}_{getattr(elem, 'sequence_order', 0)}",
        "page_number": page_number,
        "label": (getattr(elem, "label", None) or "text").lower(),
        "text_content": getattr(elem, "text_content", None) or "",
        "bbox_x1": getattr(elem, "bbox_x1", 0),
        "bbox_y1": getattr(elem, "bbox_y1", 0),
        "bbox_x2": getattr(elem, "bbox_x2", 0),
        "bbox_y2": getattr(elem, "bbox_y2", 0),
        "_view": elem,
    }


def _strip_text(text: str) -> str:
    text = _STRIP_HTML_RE.sub("", text or "")
    return text.replace("\\n", "\n").strip()


def _bbox_rect(elem: dict, page_w: float, page_h: float) -> fitz.Rect:
    x1 = float(elem.get("bbox_x1", 0))
    y1 = float(elem.get("bbox_y1", 0))
    x2 = float(elem.get("bbox_x2", x1 + 1))
    y2 = float(elem.get("bbox_y2", y1 + 1))
    x1 = max(0.0, min(x1, page_w))
    x2 = max(x1 + 1.0, min(x2, page_w))
    y1 = max(0.0, min(y1, page_h))
    y2 = max(y1 + 1.0, min(y2, page_h))
    return fitz.Rect(x1, y1, x2, y2)


def _x_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    if overlap <= 0:
        return 0.0
    base = min(a.width, b.width)
    return overlap / base if base > 0 else 0.0


def _rect_area(r: fitz.Rect) -> float:
    return max(0.0, r.width) * max(0.0, r.height)


def _intersection_over_self(inner: fitz.Rect, outer: fitz.Rect) -> float:
    """Fraction of ``inner`` covered by intersection with ``outer``."""
    inter = inner & outer
    area = _rect_area(inner)
    if area <= 0:
        return 0.0
    return _rect_area(inter) / area


def _collect_image_rects(elements: List[dict], page_w: float, page_h: float) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    for elem in elements:
        if (elem.get("label") or "").lower() in _IMAGE_LABELS:
            rects.append(_bbox_rect(elem, page_w, page_h))
    return rects


def _text_inside_figure(
    rect: fitz.Rect,
    image_rects: List[fitz.Rect],
    *,
    threshold: float = _FIGURE_INTERIOR_IOU,
) -> bool:
    for img in image_rects:
        if _intersection_over_self(rect, img) >= threshold:
            return True
    return False


def _expand_text_rect_for_overlay(
    rect: fitz.Rect,
    elem: dict,
    page_elements: List[dict],
    page_h: float,
) -> fitz.Rect:
    """Grow rect downward for longer translated text, capped by next element or ratio."""
    orig_h = rect.height
    max_y2 = min(rect.y1 + orig_h * (1.0 + _expand_ratio()), page_h)
    elem_id = elem.get("id")
    found = False
    for other in page_elements:
        if other.get("id") == elem_id:
            found = True
            continue
        if not found:
            continue
        other_rect = fitz.Rect(
            float(other.get("bbox_x1", 0)),
            float(other.get("bbox_y1", 0)),
            float(other.get("bbox_x2", 0)),
            float(other.get("bbox_y2", 0)),
        )
        if other_rect.y0 <= rect.y0:
            continue
        if _x_overlap_ratio(rect, other_rect) < 0.5:
            continue
        max_y2 = min(max_y2, other_rect.y0 - _TEXT_OVERLAY_GAP)
        break
    if max_y2 > rect.y1:
        return fitz.Rect(rect.x0, rect.y0, rect.x1, max_y2)
    return rect


def _mask_rect(page: fitz.Page, rect: fitz.Rect, *, pad: float | None = None) -> None:
    p = _overlay_pad() if pad is None else pad
    r = rect + (-p, -p, p, p)
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)


def _load_element_image_bytes(elem: Any) -> bytes | None:
    import base64 as b64mod

    def _get(name, default=None):
        if isinstance(elem, dict):
            return elem.get(name, default)
        return getattr(elem, name, default)

    key = _get("crop_image_key")
    if key:
        try:
            from services.object_storage import get_object_storage

            return get_object_storage().get_bytes(key)
        except Exception:
            pass

    inline = _get("crop_image_base64")
    if inline:
        try:
            return b64mod.b64decode(inline)
        except Exception:
            pass

    page_key = _get("page_image_key")
    bbox = _get("bbox") or {}
    x1 = bbox.get("x1", _get("bbox_x1"))
    y1 = bbox.get("y1", _get("bbox_y1"))
    x2 = bbox.get("x2", _get("bbox_x2"))
    y2 = bbox.get("y2", _get("bbox_y2"))
    if not page_key or x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    try:
        from PIL import Image

        from services.object_storage import get_object_storage

        data = get_object_storage().get_bytes(page_key)
        img = Image.open(BytesIO(data))
        box = (int(x1), int(y1), int(x2), int(y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        crop = img.crop(box)
        buf = BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None


def _fit_font_size(
    text: str,
    width: float,
    height: float,
    *,
    min_pt: float,
    max_pt: float = 12.0,
    bold: bool = False,
) -> float:
    if not text or width <= 0 or height <= 0:
        return min_pt
    # Guard against near-zero width (chart axis labels): the old
    # ``len(text) // max(int(width/5), 1)`` inflated line count and stacked
    # characters vertically when width was only a few points.
    usable_width = max(width, _MIN_TEXTBOX_WIDTH_PT)
    chars_per_line = max(int(usable_width / 5), 1)
    lines = max(1, text.count("\n") + 1, (len(text) + chars_per_line - 1) // chars_per_line)
    by_height = height / (lines * 1.35)
    size = min(max_pt, max(min_pt, by_height))
    if bold:
        size = min(size, max_pt)
    return size


def _insert_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontsize: float,
    align: int = fitz.TEXT_ALIGN_LEFT,
    mask: bool = False,
    min_pt: float = _MIN_FONT_PT,
) -> None:
    text = _strip_text(text)
    if not text:
        return
    # Too-narrow boxes (axis labels) produce vertical single-char stacks;
    # skip rather than corrupt the page.
    if rect.width < _MIN_TEXTBOX_WIDTH_PT and len(text) > 2:
        logger.debug("skip narrow textbox (%.1fpt): %r", rect.width, text[:40])
        return
    if mask:
        _mask_rect(page, rect)
    fontfile = _resolve_font_path()
    kwargs = {
        "fontsize": fontsize,
        "align": align,
        "color": (0, 0, 0),
    }
    if fontfile:
        kwargs["fontfile"] = fontfile
        kwargs["fontname"] = "noto"
    else:
        kwargs["fontname"] = "helv"
    overflow = page.insert_textbox(rect, text, **kwargs)
    size = fontsize
    while overflow < 0 and size > min_pt:
        size = max(min_pt, size * 0.85)
        overflow = page.insert_textbox(rect, text, **{**kwargs, "fontsize": size})
    if overflow < 0:
        logger.debug("textbox overflow after shrink: %r", text[:80])


def _draw_table(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    mask_text: bool = False,
) -> bool:
    cell_rows = table_text_to_cell_rows(text)
    if not cell_rows:
        return False
    n_rows, n_cols, placements = build_table_grid(cell_rows)
    n_cols, placements = compact_empty_columns(n_cols, placements)
    if n_rows == 0 or n_cols == 0 or not placements:
        return False

    col_width = rect.width / n_cols
    row_height = rect.height / max(n_rows, 1)
    min_pt = _MIN_FONT_PT_REPLACE if mask_text else _MIN_FONT_PT

    for r0, c0, r1, c1, cell_text, header in placements:
        cell_rect = fitz.Rect(
            rect.x0 + c0 * col_width,
            rect.y0 + r0 * row_height,
            rect.x0 + (c1 + 1) * col_width,
            rect.y0 + (r1 + 1) * row_height,
        )
        page.draw_rect(cell_rect, color=(0.6, 0.6, 0.6), width=0.4)
        inner = cell_rect + (2, 2, -2, -2)
        fs = _fit_font_size(
            cell_text or "", inner.width, inner.height, min_pt=min_pt, max_pt=8.0, bold=header
        )
        _insert_textbox(
            page,
            inner,
            cell_text or "",
            fontsize=fs,
            align=fitz.TEXT_ALIGN_LEFT,
            mask=mask_text,
            min_pt=min_pt,
        )
    return True


def _sort_elements_for_render(elements: List[dict]) -> List[dict]:
    """Top-to-bottom, left-to-right paint order (absolute bbox placement)."""
    return sorted(
        elements,
        key=lambda e: (float(e.get("bbox_y1", 0)), float(e.get("bbox_x1", 0))),
    )


def _render_element(
    page: fitz.Page,
    elem: dict,
    page_w: float,
    page_h: float,
    *,
    skip_text_on_background: bool = False,
    text_overlay: TextOverlayMode = "skip",
    page_elements: Optional[List[dict]] = None,
    image_rects: Optional[List[fitz.Rect]] = None,
) -> None:
    view = elem.get("_view")
    label = elem.get("label", "text")
    text = elem.get("text_content") or ""
    rect = _bbox_rect(elem, page_w, page_h)
    replace = text_overlay == "replace"
    mask = replace
    min_pt = _MIN_FONT_PT_REPLACE if replace else _MIN_FONT_PT

    if skip_text_on_background and label in _BACKGROUND_SKIP_LABELS:
        return

    if label in _IMAGE_LABELS:
        img_bytes = _load_element_image_bytes(view) if view is not None else None
        caption = _strip_text(text)
        if img_bytes:
            page.insert_image(rect, stream=img_bytes, keep_proportion=True)
            if caption and not _IMG_PLACEHOLDER_RE.match(caption):
                cap_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, min(rect.y1 + 14, page_h))
                if cap_rect.height > 2:
                    _insert_textbox(
                        page,
                        cap_rect,
                        caption,
                        fontsize=7,
                        align=fitz.TEXT_ALIGN_CENTER,
                        mask=mask,
                        min_pt=min_pt,
                    )
            return
        if caption and not _IMG_PLACEHOLDER_RE.match(caption):
            _insert_textbox(
                page,
                rect,
                caption,
                fontsize=9,
                align=fitz.TEXT_ALIGN_CENTER,
                mask=mask,
                min_pt=min_pt,
            )
        return

    if label == "table" or "<table" in text.lower():
        if _TABLE_IMAGE_FALLBACK and view is not None:
            img_bytes = _load_element_image_bytes(view)
            if img_bytes:
                page.insert_image(rect, stream=img_bytes, keep_proportion=True)
                return
        if _draw_table(page, rect, text, mask_text=mask):
            return

    if label in OCR_EQUATION_LABELS:
        if view is not None:
            img_bytes = _load_element_image_bytes(view)
            if img_bytes:
                page.insert_image(rect, stream=img_bytes, keep_proportion=True)
                return
        eq_text = _strip_text(text).strip("$")
        fs = _fit_font_size(eq_text, rect.width, rect.height, min_pt=min_pt, max_pt=11.0)
        _insert_textbox(
            page,
            rect,
            eq_text,
            fontsize=fs,
            align=fitz.TEXT_ALIGN_CENTER,
            mask=mask,
            min_pt=min_pt,
        )
        return

    plain = _strip_text(text)
    if not plain:
        return
    # Do not white-mask / redraw text that lives inside a figure crop —
    # that paints translated labels onto diagrams and erases chart ink.
    if replace and image_rects and _text_inside_figure(rect, image_rects):
        return
    if replace and page_elements:
        rect = _expand_text_rect_for_overlay(rect, elem, page_elements, page_h)
    bold = label in _HEADING_LABELS
    max_pt = 14.0 if label == "title" else 12.0 if bold else 10.0
    fs = _fit_font_size(plain, rect.width, rect.height, min_pt=min_pt, max_pt=max_pt, bold=bold)
    align = fitz.TEXT_ALIGN_CENTER if label == "title" else fitz.TEXT_ALIGN_LEFT
    _insert_textbox(
        page,
        rect,
        plain,
        fontsize=fs,
        align=align,
        mask=mask,
        min_pt=min_pt,
    )


def _render_page(
    doc: fitz.Document,
    *,
    page_number: int,
    page_w: float,
    page_h: float,
    elements: List[dict],
    page_image_key: Optional[str] = None,
    page_background: bool = False,
    text_overlay: TextOverlayMode = "skip",
    background_bytes: Optional[bytes] = None,
) -> None:
    page = doc.new_page(width=page_w, height=page_h)

    if page_background and (background_bytes or page_image_key):
        try:
            if background_bytes is None:
                from services.object_storage import get_object_storage

                background_bytes = get_object_storage().get_bytes(page_image_key)
            page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=background_bytes)
        except Exception:
            logger.debug("page background failed for page %s", page_number, exc_info=True)

    ordered = _sort_elements_for_render(elements)
    skip_text = text_overlay == "skip" and page_background and bool(page_image_key)
    image_rects = (
        _collect_image_rects(ordered, page_w, page_h) if text_overlay == "replace" else None
    )
    for elem in ordered:
        _render_element(
            page,
            elem,
            page_w,
            page_h,
            skip_text_on_background=skip_text,
            text_overlay=text_overlay,
            page_elements=ordered if text_overlay == "replace" else None,
            image_rects=image_rects,
        )


def render_export_backgrounds(
    original_pdf_path: Optional[str],
    page_numbers: Iterable[int],
) -> dict[int, bytes]:
    """Render fresh, higher-DPI page backgrounds for export from the original
    PDF, independent of the OCR model's low-res input image (which is capped
    at ``core.constants.DEFAULT_OCR_PARAMS['max_image_size']`` — tuned for
    the vision model's tiling behavior, not human viewing/zooming).

    Best-effort: returns an empty dict (never raises) if the original file
    isn't a resolvable PDF, so callers fall back to the existing
    ``page_image_key``-based background.
    """
    if not original_pdf_path:
        return {}
    try:
        import base64

        from config.settings import settings
        from utils.image_utils import render_pdf_page_to_base64

        out: dict[int, bytes] = {}
        for pn in page_numbers:
            try:
                b64 = render_pdf_page_to_base64(
                    original_pdf_path,
                    pn,
                    target_dpi=settings.layout_pdf_export_dpi,
                    max_size=settings.layout_pdf_export_max_size,
                    quality=settings.layout_pdf_export_jpeg_quality,
                )
                out[pn] = base64.b64decode(b64)
            except Exception:
                logger.debug("export background render failed for page %s", pn, exc_info=True)
        return out
    except Exception:
        logger.debug("export background rendering unavailable", exc_info=True)
        return {}


def build_layout_pdf_bytes(
    elements: Iterable[Any],
    pages: Iterable[Any],
    *,
    document_id: Optional[str] = None,
    page_background: Optional[bool] = None,
    text_overlay: TextOverlayMode = "skip",
    page_backgrounds: Optional[dict[int, bytes]] = None,
) -> bytes:
    """
    Build a PDF with one page per source page; elements placed at stored bboxes.

    ``pages`` — ORM Page rows or dicts with page_number, image_width, image_height, image_key.
    ``text_overlay`` — ``skip``: OCR export (body text omitted on scan background);
    ``replace``: translation export (mask + redraw translated text).
    ``page_backgrounds`` — optional ``{page_number: image_bytes}`` override, e.g.
    from ``render_export_backgrounds()``, used instead of the stored
    ``page_image_key`` when present (higher resolution for export/zoom).
    """
    from utils.storage_keys import page_image_key as default_page_image_key

    pages_list = list(pages)
    if page_background is None:
        page_background = _pages_have_images(pages_list)

    page_meta: dict[int, dict] = {}
    for p in pages_list:
        if isinstance(p, dict):
            pn = p.get("page_number")
            page_meta[pn] = {
                "width": float(p.get("image_width") or 595),
                "height": float(p.get("image_height") or 842),
                "image_key": p.get("image_key"),
            }
        else:
            pn = getattr(p, "page_number", None)
            page_meta[pn] = {
                "width": float(getattr(p, "image_width", None) or 595),
                "height": float(getattr(p, "image_height", None) or 842),
                "image_key": getattr(p, "image_key", None),
            }

    by_page: dict[int, list] = {}
    for elem in elements:
        if isinstance(elem, dict):
            pn = elem.get("page_number", 1)
            enriched = _enrich_export_dict(elem, document_id=document_id, page_number=int(pn))
            by_page.setdefault(int(pn), []).append(enriched)
            continue
        pn = getattr(elem, "page_number", None)
        if pn is None and getattr(elem, "page", None) is not None:
            pn = getattr(elem.page, "page_number", 1)
        pn = pn or 1
        by_page.setdefault(int(pn), []).append(elem)

    if not page_meta and by_page:
        for pn in by_page:
            page_meta[pn] = {"width": 595.0, "height": 842.0, "image_key": None}

    doc = fitz.open()
    for pn in sorted(page_meta.keys()):
        meta = page_meta[pn]
        page_w = meta["width"]
        page_h = meta["height"]
        img_key = meta.get("image_key")
        if not img_key and document_id:
            img_key = default_page_image_key(document_id, pn)

        raw_elems = by_page.get(pn, [])
        dict_elems = [_element_dict(e, page_number=pn) for e in raw_elems]
        _render_page(
            doc,
            page_number=pn,
            page_w=page_w,
            page_h=page_h,
            elements=dict_elems,
            page_image_key=img_key,
            page_background=page_background,
            text_overlay=text_overlay,
            background_bytes=(page_backgrounds or {}).get(pn),
        )

    try:
        doc.subset_fonts(fallback=True)
    except Exception:
        logger.debug("Font subset skipped", exc_info=True)
    pdf_bytes = doc.tobytes(deflate=True, garbage=3, use_objstms=1)
    doc.close()
    return pdf_bytes
