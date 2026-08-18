"""Hybrid PDF renderer: native-clean + collision-free text, OCR facsimile layer."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

from core.pdf_render.cleaner import (
    inpaint_scan_image,
    redact_native_text,
    translatable_and_reserved,
)
from core.pdf_render.fonts import fitz_font, resolve_render_font_path
from core.pdf_render.geometry import (
    RENDERER_VERSION,
    PageMeta,
    PageScene,
    Rect,
    Region,
)
from core.pdf_render.quality import (
    PdfRenderQuality,
    QualityIssue,
    aggregate_quality,
    evaluate_page_layout,
)
from core.pdf_render.regions import (
    FIGURE_LABELS,
    build_page_scene,
    page_meta_from_row,
)
from core.pdf_render.text_layout import (
    MIN_FONT_PT,
    FittedText,
    expand_rect_in_column,
    fit_textbox,
)

logger = logging.getLogger(__name__)

PdfMode = Literal["auto", "layout", "facsimile", "clean", "reflow"]
TextKind = Literal["ocr", "translation"]


@dataclass
class RenderResult:
    pdf_bytes: bytes
    quality: PdfRenderQuality
    pdf_mode: str
    renderer_version: str = RENDERER_VERSION
    continuation_pages: int = 0


def _load_image_bytes(region: Region) -> Optional[bytes]:
    if region.crop_image_key:
        try:
            from services.object_storage import get_object_storage

            return get_object_storage().get_bytes(region.crop_image_key)
        except Exception:
            pass
    if region.crop_image_base64:
        import base64

        try:
            return base64.b64decode(region.crop_image_base64)
        except Exception:
            return None
    return None


def _load_page_image(meta: PageMeta) -> Optional[bytes]:
    if meta.image_bytes:
        return meta.image_bytes
    if not meta.image_key:
        return None
    try:
        from services.object_storage import get_object_storage

        return get_object_storage().get_bytes(meta.image_key)
    except Exception:
        logger.debug("page image load failed for page %s", meta.page_number, exc_info=True)
        return None


def _insert_image(page, rect: Rect, data: bytes) -> None:
    try:
        page.insert_image(rect.to_fitz(), stream=data, keep_proportion=True)
    except Exception:
        logger.debug("insert_image failed", exc_info=True)


def _draw_table(page, rect: Rect, text: str, fontfile: Optional[str]) -> bool:
    from utils.table_grid import build_table_grid, compact_empty_columns, table_text_to_cell_rows

    rows = table_text_to_cell_rows(text)
    if not rows:
        return False
    n_rows, n_cols, placements = build_table_grid(rows)
    n_cols, placements = compact_empty_columns(n_cols, placements)
    if n_rows == 0 or n_cols == 0 or not placements:
        return False
    col_w = rect.width / n_cols
    row_h = rect.height / max(n_rows, 1)
    for r0, c0, r1, c1, cell_text, header in placements:
        cell = Rect(
            rect.x0 + c0 * col_w,
            rect.y0 + r0 * row_h,
            rect.x0 + (c1 + 1) * col_w,
            rect.y0 + (r1 + 1) * row_h,
        )
        page.draw_rect(cell.to_fitz(), color=(0.6, 0.6, 0.6), width=0.4)
        inner = Rect(cell.x0 + 2, cell.y0 + 2, cell.x1 - 2, cell.y1 - 2)
        _write_visible(page, inner, cell_text or "", fontfile, fontsize=7.0, bold=header)
    return True


def _font_kwargs(fontfile: Optional[str], *, fontsize: float, visible: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "fontsize": fontsize,
        "color": (0, 0, 0),
        "overlay": True,
    }
    if not visible:
        kwargs["render_mode"] = 3
    if fontfile:
        kwargs["fontfile"] = fontfile
        kwargs["fontname"] = "noto"
    else:
        kwargs["fontname"] = "helv"
    return kwargs


def _write_fitted_lines(
    page,
    rect: Rect,
    fitted: FittedText,
    fontfile: Optional[str],
    font,
    *,
    visible: bool = True,
    align: int = 0,
) -> None:
    """Paint pre-wrapped lines with insert_text.

    insert_textbox re-wraps and, on overflow, can write nothing — so the
    fitter's line breaks are the source of truth.
    """
    if not fitted.lines or rect.width < 2 or rect.height < 2:
        return
    kwargs = _font_kwargs(fontfile, fontsize=fitted.fontsize, visible=visible)
    y = rect.y0 + fitted.fontsize * 0.92
    max_y = rect.y1 - 0.4
    lh = fitted.line_height or fitted.fontsize * 1.28
    for line in fitted.lines:
        if y > max_y:
            break
        x = rect.x0
        if align == 1 and line and font is not None:
            width = font.text_length(line, fontsize=fitted.fontsize)
            x = rect.x0 + max(0.0, (rect.width - width) / 2.0)
        page.insert_text((x, y), line, **kwargs)
        y += lh


def _write_visible(
    page,
    rect: Rect,
    text: str,
    fontfile: Optional[str],
    *,
    fontsize: float,
    bold: bool = False,
    align: int = 0,
) -> None:
    del bold
    if not (text or "").strip() or rect.width < 2 or rect.height < 2:
        return
    import fitz

    from core.pdf_render.fonts import fitz_font

    font = fitz.Font(fontfile=fontfile) if fontfile else fitz_font("en")
    fitted = FittedText(
        fontsize=fontsize,
        lines=text.splitlines() or [text],
        line_height=fontsize * 1.28,
        used_height=rect.height,
    )
    _write_fitted_lines(page, rect, fitted, fontfile, font, visible=True, align=align)


def _copy_page(src_doc, page_index: int, dest_doc):
    dest_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)


def _scene_for_page(
    page_number: int,
    elements_by_page: dict[int, list],
    meta: PageMeta,
    source_hint: Optional[str],
) -> PageScene:
    return build_page_scene(elements_by_page.get(page_number, []), meta, source_hint=source_hint)


def _group_elements(elements: Iterable[Any]) -> dict[int, list]:
    by_page: dict[int, list] = {}
    for elem in elements:
        if isinstance(elem, dict):
            pn = int(elem.get("page_number") or 1)
            by_page.setdefault(pn, []).append(elem)
            continue
        pn = getattr(elem, "page_number", None)
        if pn is None and getattr(elem, "page", None) is not None:
            pn = getattr(elem.page, "page_number", 1)
        by_page.setdefault(int(pn or 1), []).append(elem)
    return by_page


def _neighbors_for(region: Region, scene: PageScene) -> list[Rect]:
    out = []
    for other in scene.regions:
        if other.id == region.id:
            continue
        if other.column_index != region.column_index and not other.full_width:
            if region.bbox.x_overlap_ratio(other.bbox) < 0.5:
                continue
        out.append(other.bbox)
    return out


def _draw_passthrough(page, region: Region, fontfile: Optional[str]) -> None:
    img = _load_image_bytes(region)
    if region.role in {"figure"} or region.label in FIGURE_LABELS:
        if img:
            _insert_image(page, region.bbox, img)
        return
    if region.role == "table" or "<table" in (region.text or "").lower():
        if img:
            _insert_image(page, region.bbox, img)
            return
        if _draw_table(page, region.bbox, region.text, fontfile):
            return
    if region.role == "equation" and img:
        _insert_image(page, region.bbox, img)


def _layout_page_text(
    page,
    scene: PageScene,
    font,
    fontfile: Optional[str],
    *,
    visible: bool,
    lang: str,
) -> tuple[list[tuple[Region, Rect, FittedText]], list[str], int]:
    import fitz

    drawn: list[tuple[Region, Rect, FittedText]] = []
    leftovers: list[str] = []
    font_floor = 0
    skip_roles = {"figure", "table", "equation", "vertical"}
    for region in scene.regions:
        if region.role in skip_roles or region.passthrough:
            if visible:
                _draw_passthrough(page, region, fontfile)
            continue
        text = (region.text or "").strip()
        if not text:
            continue
        if region.role == "vertical":
            continue
        max_pt = 14.0 if region.role == "heading" and region.label == "title" else 12.0
        min_pt = MIN_FONT_PT
        fitted = fit_textbox(text, region.bbox, font, min_pt=min_pt, max_pt=max_pt)
        needed = max(fitted.used_height, region.bbox.height)
        if fitted.overflow:
            expanded = expand_rect_in_column(
                region.bbox,
                needed_height=needed + 24.0,
                page_h=scene.meta.height,
                neighbors=_neighbors_for(region, scene),
            )
            if expanded.height > region.bbox.height + 0.5:
                fitted = fit_textbox(text, expanded, font, min_pt=min_pt, max_pt=max_pt)
                draw_rect = expanded
            else:
                draw_rect = region.bbox
        else:
            draw_rect = region.bbox
        if fitted.fontsize <= min_pt + 0.05:
            font_floor += 1
        if fitted.overflow:
            leftovers.append(fitted.overflow)
        if visible:
            align = 1 if region.role == "heading" and region.label == "title" else 0
            _write_fitted_lines(
                page, draw_rect, fitted, fontfile, font, visible=True, align=align
            )
        else:
            _write_fitted_lines(
                page, draw_rect, fitted, fontfile, font, visible=False, align=0
            )
        drawn.append((region, draw_rect, fitted))
    return drawn, leftovers, font_floor


def _append_continuation(
    doc, leftovers: list[str], page_number: int, fontfile: Optional[str]
) -> int:
    if not leftovers:
        return 0
    import fitz

    page = doc.new_page(width=595, height=842)
    body = f"… (tiếp trang {page_number})\n\n" + "\n\n".join(leftovers)
    rect = fitz.Rect(48, 48, 547, 794)
    kwargs: dict[str, Any] = {"fontsize": 10, "color": (0, 0, 0)}
    if fontfile:
        kwargs["fontfile"] = fontfile
        kwargs["fontname"] = "noto"
    page.insert_textbox(rect, body, **kwargs)
    return 1


def _open_original(original_pdf_bytes: Optional[bytes], original_pdf_path: Optional[str]):
    import fitz

    if original_pdf_bytes:
        return fitz.open(stream=original_pdf_bytes, filetype="pdf")
    if original_pdf_path:
        return fitz.open(original_pdf_path)
    return None


def render_document_pdf(
    *,
    pages: Iterable[Any],
    elements: Iterable[Any],
    original_pdf_bytes: Optional[bytes] = None,
    original_pdf_path: Optional[str] = None,
    pdf_mode: PdfMode = "auto",
    text_kind: TextKind = "ocr",
    lang: str = "vi",
    source_hint: Optional[str] = None,
    page_backgrounds: Optional[dict[int, bytes]] = None,
) -> RenderResult:
    """Render a layout-faithful PDF. ``auto`` picks facsimile for OCR, layout for translation."""
    import fitz

    pages_list = list(pages)
    if pdf_mode == "auto":
        pdf_mode = "facsimile" if text_kind == "ocr" else "layout"
    if pdf_mode == "reflow":
        raise ValueError("reflow mode is handled by the export service")

    font = fitz_font(lang)
    fontfile = resolve_render_font_path(lang)
    by_page = _group_elements(elements)
    metas = [page_meta_from_row(p) for p in pages_list]
    if not metas and by_page:
        for pn in sorted(by_page):
            metas.append(PageMeta(page_number=pn, width=595.0, height=842.0))

    src = _open_original(original_pdf_bytes, original_pdf_path)
    dest = fitz.open()
    issues: list[QualityIssue] = []
    continuations = 0
    try:
        for meta in metas:
            scene = _scene_for_page(meta.page_number, by_page, meta, source_hint)
            page_index = meta.page_number - 1
            src_page = None
            if src is not None and 0 <= page_index < src.page_count:
                src_page = src[page_index]
                meta.width, meta.height = float(src_page.rect.width), float(src_page.rect.height)
                meta.rotation = int(src_page.rotation or 0)
                scene = _scene_for_page(meta.page_number, by_page, meta, source_hint)

            original_text = ""
            if src_page is not None and text_kind == "translation":
                original_text = src_page.get_text("text") or ""

            if pdf_mode == "facsimile":
                page = dest.new_page(width=meta.width, height=meta.height)
                bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
                if bg:
                    page.insert_image(page.rect, stream=bg)
                elif src_page is not None:
                    pix = src_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    page.insert_image(page.rect, pixmap=pix)
                drawn, leftovers, font_floor = _layout_page_text(
                    page, scene, font, fontfile, visible=False, lang=lang
                )
                output_text = " ".join(t.visible_text for _, _, t in drawn)
            elif pdf_mode == "clean":
                page = dest.new_page(width=meta.width, height=meta.height)
                bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
                trans, reserved = translatable_and_reserved(scene.regions)
                if bg:
                    cleaned = inpaint_scan_image(
                        bg, trans, reserved, page_w=meta.width, page_h=meta.height
                    )
                    page.insert_image(page.rect, stream=cleaned or bg)
                drawn, leftovers, font_floor = _layout_page_text(
                    page, scene, font, fontfile, visible=True, lang=lang
                )
                output_text = page.get_text("text") or ""
            else:
                # layout: native redact + redraw, or scan inpaint
                if src_page is not None and (meta.page_type or "text") != "scanned":
                    _copy_page(src, page_index, dest)
                    page = dest[-1]
                    trans, reserved = translatable_and_reserved(scene.regions)
                    redact_native_text(page, trans, reserved)
                else:
                    page = dest.new_page(width=meta.width, height=meta.height)
                    bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
                    trans, reserved = translatable_and_reserved(scene.regions)
                    if src_page is not None:
                        pix = src_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        raw = pix.tobytes("jpeg")
                        cleaned = inpaint_scan_image(
                            raw, trans, reserved, page_w=meta.width, page_h=meta.height
                        )
                        page.insert_image(page.rect, stream=cleaned or raw)
                    elif bg:
                        cleaned = inpaint_scan_image(
                            bg, trans, reserved, page_w=meta.width, page_h=meta.height
                        )
                        page.insert_image(page.rect, stream=cleaned or bg)
                drawn, leftovers, font_floor = _layout_page_text(
                    page, scene, font, fontfile, visible=True, lang=lang
                )
                output_text = page.get_text("text") or ""

            issues.extend(
                evaluate_page_layout(
                    page_number=meta.page_number,
                    drawn=drawn,
                    source_text=original_text if text_kind == "translation" else "",
                    output_text=output_text if text_kind == "translation" else "",
                    font_floor_hits=font_floor,
                )
            )
            continuations += _append_continuation(dest, leftovers, meta.page_number, fontfile)

        try:
            dest.subset_fonts(fallback=True)
        except Exception:
            logger.debug("Font subset skipped", exc_info=True)
        pdf_bytes = dest.tobytes(deflate=True, garbage=3, use_objstms=1)
    finally:
        dest.close()
        if src is not None:
            src.close()

    quality = aggregate_quality(issues, len(metas), pdf_mode)
    return RenderResult(
        pdf_bytes=pdf_bytes,
        quality=quality,
        pdf_mode=pdf_mode,
        continuation_pages=continuations,
    )


def render_to_tempfile(**kwargs) -> tuple[str, RenderResult]:
    """Render sequentially to a temp file so callers can stream via put_file."""
    result = render_document_pdf(**kwargs)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    import os

    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(result.pdf_bytes)
    return path, result
