"""Build a column-aware PageScene from stored layout elements."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from core.pdf_render.geometry import (
    ColumnBand,
    PageMeta,
    PageScene,
    Rect,
    Region,
    stored_to_page_rect,
)
from core.spatial.grouping import assign_column_membership, detect_columns_projection

FIGURE_LABELS = frozenset({"figure", "image", "picture", "chart", "graph"})
TABLE_LABELS = frozenset({"table"})
EQUATION_LABELS = frozenset({"equation", "formula", "isolate_formula"})
HEADING_LABELS = frozenset({"title", "sub_title", "heading", "section_heading"})
HEADER_FOOTER_LABELS = frozenset({"header", "footer", "page_number"})
CAPTION_LABELS = frozenset({"caption"})
AFFILIATION_RE = re.compile(
    r"\b(university|universidad|institute|instituto|department|departamento|"
    r"affiliation|email|orcid)\b",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(
    r"^(figure|fig\.?|table|tab\.?|hình|bảng|image|chart)\s*\d+",
    re.IGNORECASE,
)
NARROW_MIN_PT = 14.0
FULL_WIDTH_RATIO = 0.68
LINE_HEIGHT_RATIO = 0.035
COLUMN_X_OVERLAP = 0.5


def _label_of(elem: dict) -> str:
    return (elem.get("label") or "text").lower()


def _text_of(elem: dict) -> str:
    return (elem.get("text_content") or elem.get("text_full") or elem.get("text") or "").strip()


def _bbox_of(elem: dict, meta: PageMeta) -> Rect:
    bbox = elem.get("bbox")
    if bbox:
        stored = Rect.from_bbox_dict(bbox)
    else:
        stored = Rect.from_xyxy(
            elem.get("bbox_x1", elem.get("x1", 0)) or 0,
            elem.get("bbox_y1", elem.get("y1", 0)) or 0,
            elem.get("bbox_x2", elem.get("x2", 0)) or 0,
            elem.get("bbox_y2", elem.get("y2", 0)) or 0,
        )
    return stored_to_page_rect(
        stored,
        image_w=meta.storage_width,
        image_h=meta.storage_height,
        page_w=meta.width,
        page_h=meta.height,
        rotation=meta.rotation,
    )


def classify_role(label: str, text: str, bbox: Rect, page_w: float, page_h: float) -> str:
    lab = (label or "text").lower()
    if lab in FIGURE_LABELS:
        return "figure"
    if lab in TABLE_LABELS:
        return "table"
    if lab in EQUATION_LABELS:
        return "equation"
    if lab in HEADER_FOOTER_LABELS:
        return "header" if lab == "header" else "footer"
    if lab in CAPTION_LABELS or CAPTION_RE.match(text or ""):
        return "caption"
    if lab in HEADING_LABELS:
        return "heading"
    if is_narrow_vertical(bbox):
        return "vertical"
    if is_affiliation(bbox, page_h, text):
        return "affiliation"
    if bbox.width >= page_w * FULL_WIDTH_RATIO and bbox.y0 < page_h * 0.18:
        if lab in HEADING_LABELS:
            return "heading"
        return "banner"
    return "body"


def is_narrow_vertical(bbox: Rect) -> bool:
    return bbox.width < NARROW_MIN_PT and bbox.height > max(2.0 * bbox.width, 24.0)


def is_full_width(bbox: Rect, page_w: float) -> bool:
    return bbox.width >= page_w * FULL_WIDTH_RATIO


def is_affiliation(bbox: Rect, page_h: float, text: str) -> bool:
    if bbox.y0 > page_h * 0.28:
        return False
    if bbox.height > page_h * 0.22:
        return False
    sample = (text or "")[:240]
    return bool(AFFILIATION_RE.search(sample))


def is_passthrough_role(role: str) -> bool:
    return role in {"figure", "table", "equation", "vertical"}


def _looks_like_ocr_lines(regions: list[Region], page_h: float) -> bool:
    bodies = [r for r in regions if r.role in {"body", "caption", "affiliation"}]
    if len(bodies) < 6:
        return False
    heights = sorted(r.bbox.height for r in bodies)
    median = heights[len(heights) // 2]
    return median < page_h * LINE_HEIGHT_RATIO


def _figure_rects(regions: list[Region]) -> list[Rect]:
    return [r.bbox for r in regions if r.role in {"figure", "table", "equation"}]


def _blocked_by_reserved(a: Rect, b: Rect, reserved: list[Rect]) -> bool:
    gap = Rect(min(a.x0, b.x0), min(a.y1, b.y1), max(a.x1, b.x1), max(a.y0, b.y0))
    if gap.height <= 0:
        gap = Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
    return any(gap.intersects(r) and r.intersection_over_self(gap) < 0.95 for r in reserved)


def _group_ocr_lines(regions: list[Region], page_w: float, page_h: float) -> list[Region]:
    reserved = _figure_rects(regions)
    keep: list[Region] = []
    mergeable = [r for r in regions if r.role == "body" and not r.full_width]
    keep.extend(r for r in regions if r not in mergeable)

    by_col: dict[int, list[Region]] = {}
    for r in mergeable:
        by_col.setdefault(r.column_index, []).append(r)

    grouped: list[Region] = []
    seq = 0
    for col_idx, items in by_col.items():
        items = sorted(items, key=lambda r: (r.bbox.y0, r.bbox.x0))
        current: list[Region] = []

        def flush() -> None:
            nonlocal seq
            if not current:
                return
            bbox = current[0].bbox
            for extra in current[1:]:
                bbox = bbox.union(extra.bbox)
            text = "\n".join(x.text for x in current if x.text)
            grouped.append(
                Region(
                    id=f"blk_{col_idx}_{seq}",
                    label=current[0].label,
                    role="body",
                    text=text,
                    bbox=bbox,
                    column_index=col_idx,
                    full_width=False,
                    passthrough=False,
                    source=current[0].source,
                    sequence_order=seq,
                    page_number=current[0].page_number,
                )
            )
            seq += 1
            current.clear()

        for item in items:
            if not current:
                current.append(item)
                continue
            prev = current[-1]
            same_col = item.bbox.x_overlap_ratio(prev.bbox) >= COLUMN_X_OVERLAP
            gap = item.bbox.y0 - prev.bbox.y1
            median_h = max(prev.bbox.height, 8.0)
            if (
                same_col
                and gap <= median_h * 1.5
                and not _blocked_by_reserved(prev.bbox, item.bbox, reserved)
            ):
                current.append(item)
            else:
                flush()
                current.append(item)
        flush()

    return keep + grouped


def _elements_for_columns(regions: list[Region]) -> list[dict]:
    out = []
    for r in regions:
        if r.full_width or r.role in {"figure", "table", "equation", "header", "footer", "banner"}:
            continue
        out.append(
            {
                "bbox_x1": r.bbox.x0,
                "bbox_y1": r.bbox.y0,
                "bbox_x2": r.bbox.x1,
                "bbox_y2": r.bbox.y1,
                "label": r.label,
            }
        )
    return out


def build_page_scene(
    elements: Iterable[Any],
    meta: PageMeta,
    *,
    source_hint: Optional[str] = None,
) -> PageScene:
    raw: list[Region] = []
    for idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            elem = {
                "label": getattr(elem, "label", "text"),
                "text_content": getattr(elem, "text_content", "") or "",
                "bbox_x1": getattr(elem, "bbox_x1", 0),
                "bbox_y1": getattr(elem, "bbox_y1", 0),
                "bbox_x2": getattr(elem, "bbox_x2", 0),
                "bbox_y2": getattr(elem, "bbox_y2", 0),
                "crop_image_key": getattr(elem, "crop_image_key", None),
                "crop_image_base64": getattr(elem, "crop_image_base64", None),
                "sequence_order": getattr(elem, "sequence_order", idx),
                "source": getattr(elem, "source", source_hint or "unknown"),
                "page_number": getattr(elem, "page_number", meta.page_number),
            }
        bbox = _bbox_of(elem, meta)
        text = _text_of(elem)
        label = _label_of(elem)
        role = classify_role(label, text, bbox, meta.width, meta.height)
        full_width = is_full_width(bbox, meta.width) and role not in {
            "figure",
            "table",
            "equation",
        }
        raw.append(
            Region(
                id=str(elem.get("id") or f"p{meta.page_number}_{idx}"),
                label=label,
                role=role,
                text=text,
                bbox=bbox,
                full_width=full_width,
                passthrough=is_passthrough_role(role),
                source=elem.get("source") or source_hint or "unknown",
                crop_image_key=elem.get("crop_image_key"),
                crop_image_base64=elem.get("crop_image_base64"),
                sequence_order=int(elem.get("sequence_order") or idx),
                page_number=int(elem.get("page_number") or meta.page_number),
                extra={"bbox_raw": elem.get("bbox")},
            )
        )

    col_elems = _elements_for_columns(raw)
    detected = detect_columns_projection(col_elems, int(meta.width) or 595)
    columns = [
        ColumnBand(index=c.index, x0=float(c.x1), x1=float(c.x2), full_width=False)
        for c in detected
    ]
    if not columns:
        columns = [ColumnBand(index=0, x0=0.0, x1=meta.width, full_width=True)]

    assigned = assign_column_membership(
        [
            {
                "_i": i,
                "bbox_x1": r.bbox.x0,
                "bbox_y1": r.bbox.y0,
                "bbox_x2": r.bbox.x1,
                "bbox_y2": r.bbox.y1,
            }
            for i, r in enumerate(raw)
        ],
        detected,
    )
    col_by_i = {item["_i"]: item.get("column_index", 0) for item in assigned}
    regions: list[Region] = []
    for i, r in enumerate(raw):
        col = -1 if r.full_width else int(col_by_i.get(i, 0))
        regions.append(
            Region(
                id=r.id,
                label=r.label,
                role=r.role,
                text=r.text,
                bbox=r.bbox,
                column_index=col,
                full_width=r.full_width,
                passthrough=r.passthrough,
                source=r.source,
                crop_image_key=r.crop_image_key,
                crop_image_base64=r.crop_image_base64,
                sequence_order=r.sequence_order,
                page_number=r.page_number,
                extra=r.extra,
            )
        )

    source = source_hint or (regions[0].source if regions else "unknown")
    if source == "ocr" or _looks_like_ocr_lines(regions, meta.height):
        regions = _group_ocr_lines(regions, meta.width, meta.height)

    regions = _drop_contained_regions(regions)
    regions = _drop_duplicate_fragments(regions)
    _passthrough_text_on_figures(regions)
    regions = _restack_overlaps(regions, meta.height)
    regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0, r.sequence_order))
    return PageScene(meta=meta, regions=regions, columns=columns)


def _restack_overlaps(regions: list[Region], page_h: float) -> list[Region]:
    """Push later text boxes down when stored bboxes sit on top of each other."""
    from dataclasses import replace

    skip = {"figure", "table", "equation", "vertical"}
    ordered = sorted(regions, key=lambda r: (r.bbox.y0, r.sequence_order, r.bbox.x0))
    out: list[Region] = []
    for r in ordered:
        if r.passthrough or r.role in skip:
            out.append(r)
            continue
        bbox = r.bbox
        for prev in out:
            if prev.passthrough or prev.role in skip:
                continue
            if bbox.x_overlap_ratio(prev.bbox) < 0.45:
                continue
            if not bbox.intersects(prev.bbox, eps=0.8):
                continue
            height = max(bbox.height, 8.0)
            y0 = prev.bbox.y1 + 1.2
            if y0 + 8.0 > page_h:
                break
            bbox = Rect(bbox.x0, y0, bbox.x1, min(page_h - 0.5, y0 + height))
        out.append(replace(r, bbox=bbox))
    return out


def _drop_duplicate_fragments(regions: list[Region]) -> list[Region]:
    """Drop short boxes whose text is already contained in a larger region."""
    textual = [(r, " ".join((r.text or "").split())) for r in regions if (r.text or "").strip()]
    drop: set[str] = set()
    for r, text in textual:
        if r.passthrough or len(text) < 36:
            continue
        for other, other_text in textual:
            if other is r:
                continue
            if other_text == text:
                if (r.bbox.y0, r.bbox.x0) > (other.bbox.y0, other.bbox.x0):
                    drop.add(r.id)
                    break
                continue
            if len(other_text) <= len(text):
                continue
            if other_text.startswith(text[: min(80, len(text))]) and len(other_text) >= len(text) * 1.25:
                drop.add(r.id)
                break
    for r in regions:
        if r.id in drop:
            r.text = ""
    return regions


def _passthrough_text_on_figures(regions: list[Region]) -> None:
    """Keep native figure titles/captions instead of redrawing over them."""
    figs = [r.bbox for r in regions if r.role in {"figure", "table", "equation"}]
    if not figs:
        return
    for r in regions:
        if r.passthrough or r.role in {"figure", "table", "equation"}:
            continue
        if any(r.bbox.intersection_over_self(fig) >= 0.35 for fig in figs):
            r.passthrough = True


def _drop_contained_regions(regions: list[Region]) -> list[Region]:
    kept: list[Region] = []
    for r in regions:
        if r.passthrough or r.role in {"figure", "table", "equation"}:
            kept.append(r)
            continue
        contained = False
        for other in regions:
            if other is r:
                continue
            if (
                r.bbox.intersection_over_self(other.bbox) >= 0.72
                and other.bbox.area > r.bbox.area * 1.05
            ):
                contained = True
                break
        if not contained:
            kept.append(r)
    return kept


def page_meta_from_row(page, *, fallback_w: float = 595.0, fallback_h: float = 842.0) -> PageMeta:
    if isinstance(page, dict):
        return PageMeta(
            page_number=int(page.get("page_number") or 1),
            width=float(page.get("image_width") or fallback_w),
            height=float(page.get("image_height") or fallback_h),
            page_type=(page.get("page_type") or "text"),
            rotation=int(page.get("rotation") or 0),
            image_width=page.get("image_width"),
            image_height=page.get("image_height"),
            image_key=page.get("image_key"),
            image_bytes=page.get("image_bytes"),
        )
    return PageMeta(
        page_number=int(getattr(page, "page_number", 1) or 1),
        width=float(getattr(page, "image_width", None) or fallback_w),
        height=float(getattr(page, "image_height", None) or fallback_h),
        page_type=getattr(page, "page_type", None) or "text",
        rotation=int(getattr(page, "rotation", 0) or 0),
        image_width=getattr(page, "image_width", None),
        image_height=getattr(page, "image_height", None),
        image_key=getattr(page, "image_key", None),
    )
