"""Hybrid PDF renderer: native-clean + collision-free text, OCR facsimile layer."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

from core.constants import SCAN_LIKE_PAGE_TYPES
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
    TABLE_MIN_FONT_PT,
    FittedText,
    expand_rect_in_column,
    fit_textbox,
)
from utils.image_utils import encode_scan_jpeg
from utils.math_omml import inline_math_to_plain

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


def _as_jpeg(data: bytes) -> bytes:
    """Re-encode a crop the extractor handed over as PNG.

    Part of the extraction path base64s figure crops as RGB PNG; embedding
    those verbatim cost 23 MB of one 61 MB translation export. A crop PNG
    already stores more cheaply — a blank margin, a line drawing — is kept.
    """
    if data[:2] == b"\xff\xd8":
        return data
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as img:
            if img.mode in {"RGBA", "LA", "P"}:
                return data  # transparency would turn black
            encoded = encode_scan_jpeg(img.copy(), quality=85)
    except Exception:
        logger.debug("crop re-encode failed", exc_info=True)
        return data
    return encoded if len(encoded) < len(data) else data


def _insert_image(page, rect: Rect, data: bytes) -> None:
    try:
        page.insert_image(rect.to_fitz(), stream=_as_jpeg(data), keep_proportion=True)
    except Exception:
        logger.debug("insert_image failed", exc_info=True)


def _draw_table(page, rect: Rect, text: str, fontfile: Optional[str]) -> Optional[FittedText]:
    """Redraw a table with metric-fitted cell text. Never pastes a source crop.

    Returns an aggregate ``FittedText`` for quality (overflow / font floor).
    Cell overflow is reported but not promoted to continuation pages.
    """
    import fitz

    from core.pdf_render.fonts import fitz_font
    from utils.table_grid import build_table_grid, compact_empty_columns, table_text_to_cell_rows

    rows = table_text_to_cell_rows(text)
    if not rows:
        return None
    n_rows, n_cols, placements = build_table_grid(rows)
    n_cols, placements = compact_empty_columns(n_cols, placements)
    if n_rows == 0 or n_cols == 0 or not placements:
        return None

    font = fitz.Font(fontfile=fontfile) if fontfile else fitz_font("en")
    col_w = rect.width / n_cols
    row_h = rect.height / max(n_rows, 1)
    min_pt = TABLE_MIN_FONT_PT
    max_pt = 8.0
    overflows: list[str] = []
    smallest = max_pt

    for r0, c0, r1, c1, cell_text, header in placements:
        cell = Rect(
            rect.x0 + c0 * col_w,
            rect.y0 + r0 * row_h,
            rect.x0 + (c1 + 1) * col_w,
            rect.y0 + (r1 + 1) * row_h,
        )
        page.draw_rect(cell.to_fitz(), color=(0.6, 0.6, 0.6), width=0.4)
        inner = Rect(cell.x0 + 2, cell.y0 + 2, cell.x1 - 2, cell.y1 - 2)
        if inner.width < 2 or inner.height < 2:
            continue
        body = (cell_text or "").strip()
        if not body:
            continue
        fitted = fit_textbox(
            body,
            inner,
            font,
            min_pt=min_pt,
            max_pt=max_pt,
            bold=bool(header),
        )
        smallest = min(smallest, fitted.fontsize)
        if fitted.overflow:
            overflows.append(fitted.overflow)
        _write_fitted_lines(page, inner, fitted, fontfile, font, visible=True, align=0)

    return FittedText(
        fontsize=smallest if placements else min_pt,
        lines=[],
        overflow=" ".join(overflows).strip(),
        line_height=0.0,
        used_height=rect.height,
        missing_glyphs=0,
    )


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


def _render_source_page_jpeg(src_page) -> bytes:
    """Rasterise one source page at the configured export DPI/quality."""
    import fitz

    from config.settings import settings

    zoom = max(settings.layout_pdf_export_dpi, 72) / 72.0
    pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("jpeg", jpg_quality=settings.layout_pdf_export_jpeg_quality)


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


def _draw_passthrough(
    page,
    region: Region,
    fontfile: Optional[str],
    *,
    allow_table_crop: bool = True,
    allow_figure_crop: bool = True,
) -> Optional[FittedText]:
    """Draw non-body regions. Returns table fit aggregate when a table is redrawn."""
    img = _load_image_bytes(region)
    if region.role in {"figure"} or region.label in FIGURE_LABELS:
        if img and allow_figure_crop:
            _insert_image(page, region.bbox, img)
        return None
    if region.role == "table" or "<table" in (region.text or "").lower():
        # Translation layout: never paste the source table crop (glyphs stay
        # in the pixels). OCR facsimile skips this path (visible=False).
        if allow_table_crop and img:
            _insert_image(page, region.bbox, img)
            return None
        return _draw_table(page, region.bbox, region.text, fontfile)
    if region.role == "equation" and img and allow_figure_crop:
        _insert_image(page, region.bbox, img)
    return None


def _layout_page_text(
    page,
    scene: PageScene,
    font,
    fontfile: Optional[str],
    *,
    visible: bool,
    lang: str,
    allow_table_crop: bool = True,
    allow_figure_crop: bool = True,
) -> tuple[list[tuple[Region, Rect, FittedText]], list[str], int]:
    import fitz

    drawn: list[tuple[Region, Rect, FittedText]] = []
    leftovers: list[str] = []
    font_floor = 0
    skip_roles = {"figure", "table", "equation", "vertical"}
    for region in scene.regions:
        if region.role in skip_roles or region.passthrough:
            if visible:
                fitted = _draw_passthrough(
                    page,
                    region,
                    fontfile,
                    allow_table_crop=allow_table_crop,
                    allow_figure_crop=allow_figure_crop,
                )
                if fitted is not None:
                    drawn.append((region, region.bbox, fitted))
                    if fitted.fontsize <= TABLE_MIN_FONT_PT + 0.05:
                        font_floor += 1
                    # Table cell overflow stays a quality issue — do not
                    # spill into continuation pages.
            continue
        # A text box cannot typeset math: print $T_{s}$ as readable Unicode.
        text = inline_math_to_plain((region.text or "").strip())
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
            _write_fitted_lines(page, draw_rect, fitted, fontfile, font, visible=True, align=align)
        else:
            _write_fitted_lines(page, draw_rect, fitted, fontfile, font, visible=False, align=0)
        drawn.append((region, draw_rect, fitted))
    return drawn, leftovers, font_floor


def _attach_overflow_note(page, leftovers: list[str]) -> None:
    """Keep text that did not fit its box as a note on the same page.

    Continuation pages used to be appended instead, so page N of every export
    drifted away from page N of the source (Digital Control: 159 → 178 pages).
    The overflow is still reported as a quality issue by evaluate_page_layout.
    """
    if not leftovers:
        return
    import fitz

    annot = page.add_text_annot(
        fitz.Point(max(page.rect.width - 24, 0), 24), "\n\n".join(leftovers), icon="Note"
    )
    annot.set_info(title="Phần chữ không vừa khung")
    annot.update()


def _open_original(original_pdf_bytes: Optional[bytes], original_pdf_path: Optional[str]):
    import fitz

    if original_pdf_bytes:
        return fitz.open(stream=original_pdf_bytes, filetype="pdf")
    if original_pdf_path:
        return fitz.open(original_pdf_path)
    return None


def _render_page_fragment(
    meta: PageMeta,
    *,
    by_page: dict[int, list],
    original_pdf_bytes: Optional[bytes],
    original_pdf_path: Optional[str],
    pdf_mode: str,
    text_kind: TextKind,
    lang: str,
    source_hint: Optional[str],
    page_backgrounds: Optional[dict[int, bytes]],
    fontfile: Optional[str],
    src=None,
    dest=None,
) -> tuple[list[QualityIssue], int]:
    """Render one source page into ``dest`` (must be an open Document).

    When ``src``/``dest`` are provided the caller owns their lifecycle (batch
    workers open once per batch). Returns (issues, continuation_count).
    """
    import copy

    import fitz

    meta = copy.copy(meta)
    font = fitz_font(lang)
    owns_docs = dest is None
    if owns_docs:
        src = _open_original(original_pdf_bytes, original_pdf_path)
        dest = fitz.open()
    assert dest is not None
    issues: list[QualityIssue] = []
    continuations = 0
    try:
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

        include_tables = text_kind == "translation"
        allow_table_crop = text_kind != "translation"

        if pdf_mode == "facsimile":
            if src_page is not None:
                # The source page already holds the scan (or the vector text)
                # at its own resolution. Copying it keeps that quality and the
                # file near the original; re-rendering every page as a 150 DPI
                # JPEG turned a 6.3 MB book into a 178 MB export.
                _copy_page(src, page_index, dest)
                page = dest[-1]
            else:
                page = dest.new_page(width=meta.width, height=meta.height)
                bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
                if bg:
                    page.insert_image(page.rect, stream=bg)
            # A copied native page carries its own searchable text; adding the
            # OCR layer on top would double every hit.
            if src_page is not None and (src_page.get_text("text") or "").strip():
                drawn, leftovers, font_floor = [], [], 0
            else:
                drawn, leftovers, font_floor = _layout_page_text(
                    page,
                    scene,
                    font,
                    fontfile,
                    visible=False,
                    lang=lang,
                    allow_table_crop=allow_table_crop,
                )
            output_text = " ".join(t.visible_text for _, _, t in drawn)
        elif pdf_mode == "clean":
            page = dest.new_page(width=meta.width, height=meta.height)
            bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
            trans, reserved = translatable_and_reserved(
                scene.regions, include_tables=include_tables
            )
            if bg:
                cleaned = inpaint_scan_image(
                    bg, trans, reserved, page_w=meta.width, page_h=meta.height
                )
                page.insert_image(page.rect, stream=cleaned or bg)
            drawn, leftovers, font_floor = _layout_page_text(
                page,
                scene,
                font,
                fontfile,
                visible=True,
                lang=lang,
                allow_table_crop=allow_table_crop,
            )
            output_text = page.get_text("text") or ""
        else:
            # layout: native redact + redraw, or scan inpaint
            native_copy = (
                src_page is not None and (meta.page_type or "text") not in SCAN_LIKE_PAGE_TYPES
            )
            if native_copy:
                _copy_page(src, page_index, dest)
                page = dest[-1]
                trans, reserved = translatable_and_reserved(
                    scene.regions, include_tables=include_tables
                )
                redact_native_text(page, trans, reserved)
            else:
                page = dest.new_page(width=meta.width, height=meta.height)
                bg = (page_backgrounds or {}).get(meta.page_number) or _load_page_image(meta)
                trans, reserved = translatable_and_reserved(
                    scene.regions, include_tables=include_tables
                )
                if src_page is not None:
                    # The scan has to be re-rendered (text is masked out of the
                    # pixels), but at the export DPI/quality — 2× at JPEG 95
                    # cost ~1 MB a page.
                    raw = bg or _render_source_page_jpeg(src_page)
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
                page,
                scene,
                font,
                fontfile,
                visible=True,
                lang=lang,
                allow_table_crop=allow_table_crop,
                # The copied page already holds its figures; pasting the
                # stored crop on top duplicates the artwork at a larger size.
                allow_figure_crop=not native_copy,
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
        _attach_overflow_note(page, leftovers)
    finally:
        if owns_docs:
            dest.close()
            if src is not None:
                src.close()
    return issues, continuations


def _render_pages_batch(
    metas: list[PageMeta],
    *,
    by_page: dict[int, list],
    original_pdf_bytes: Optional[bytes],
    original_pdf_path: Optional[str],
    pdf_mode: str,
    text_kind: TextKind,
    lang: str,
    source_hint: Optional[str],
    page_backgrounds: Optional[dict[int, bytes]],
    fontfile: Optional[str],
) -> tuple[bytes, list[QualityIssue], int]:
    """Render a contiguous batch of pages with one shared src/dest Document."""
    import fitz

    src = _open_original(original_pdf_bytes, original_pdf_path)
    dest = fitz.open()
    issues: list[QualityIssue] = []
    continuations = 0
    try:
        for meta in metas:
            page_issues, cont = _render_page_fragment(
                meta,
                by_page=by_page,
                original_pdf_bytes=original_pdf_bytes,
                original_pdf_path=original_pdf_path,
                pdf_mode=pdf_mode,
                text_kind=text_kind,
                lang=lang,
                source_hint=source_hint,
                page_backgrounds=page_backgrounds,
                fontfile=fontfile,
                src=src,
                dest=dest,
            )
            issues.extend(page_issues)
            continuations += cont
        frag = dest.tobytes(deflate=True, garbage=4, use_objstms=1)
    finally:
        dest.close()
        if src is not None:
            src.close()
    return frag, issues, continuations


def _partition_metas(metas: list[PageMeta], workers: int) -> list[list[PageMeta]]:
    """Split metas into up to ``workers`` non-empty contiguous batches."""
    n = len(metas)
    if n == 0:
        return []
    w = max(1, min(workers, n))
    size, rem = divmod(n, w)
    batches: list[list[PageMeta]] = []
    idx = 0
    for i in range(w):
        take = size + (1 if i < rem else 0)
        if take <= 0:
            continue
        batches.append(metas[idx : idx + take])
        idx += take
    return batches


def _element_as_plain(elem: Any) -> dict:
    """Make layout elements picklable for process-pool workers."""
    if isinstance(elem, dict):
        return elem
    pn = getattr(elem, "page_number", None)
    if pn is None and getattr(elem, "page", None) is not None:
        pn = getattr(elem.page, "page_number", 1)
    from utils.translation_elements import layout_element_to_dict

    return layout_element_to_dict(elem, int(pn or 1))


def _plain_by_page(by_page: dict[int, list]) -> dict[int, list]:
    return {pn: [_element_as_plain(e) for e in elems] for pn, elems in by_page.items()}


def _process_pages_batch(payload: dict) -> tuple[bytes, list[QualityIssue], int]:
    """Top-level worker entry for ProcessPoolExecutor (must be picklable)."""
    return _render_pages_batch(
        payload["metas"],
        by_page=payload["by_page"],
        original_pdf_bytes=payload.get("original_pdf_bytes"),
        original_pdf_path=payload.get("original_pdf_path"),
        pdf_mode=payload["pdf_mode"],
        text_kind=payload["text_kind"],
        lang=payload["lang"],
        source_hint=payload.get("source_hint"),
        page_backgrounds=payload.get("page_backgrounds"),
        fontfile=payload.get("fontfile"),
    )


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

    fontfile = resolve_render_font_path(lang)
    by_page = _plain_by_page(_group_elements(elements))
    metas = [page_meta_from_row(p) for p in pages_list]
    if not metas and by_page:
        for pn in sorted(by_page):
            metas.append(PageMeta(page_number=pn, width=595.0, height=842.0))

    try:
        from config.settings import settings

        workers = max(1, int(getattr(settings, "layout_pdf_render_workers", 4) or 1))
    except Exception:
        workers = 4

    common = dict(
        by_page=by_page,
        original_pdf_bytes=original_pdf_bytes,
        original_pdf_path=original_pdf_path,
        pdf_mode=pdf_mode,
        text_kind=text_kind,
        lang=lang,
        source_hint=source_hint,
        page_backgrounds=page_backgrounds,
        fontfile=fontfile,
    )

    fragments: list[tuple[bytes, list[QualityIssue], int]]
    if len(metas) <= 2 or workers <= 1:
        fragments = [_render_pages_batch(metas, **common)]
    else:
        # Process pool: PyMuPDF/OpenCV release the GIL poorly for text-fit;
        # threads barely help — processes do.
        from concurrent.futures import ProcessPoolExecutor

        batches = _partition_metas(metas, workers)
        payloads = [{**common, "metas": batch} for batch in batches]
        try:
            with ProcessPoolExecutor(max_workers=len(batches)) as pool:
                futures = [pool.submit(_process_pages_batch, payload) for payload in payloads]
                fragments = [fut.result() for fut in futures]
        except Exception:
            logger.warning(
                "Process-pool PDF render failed; falling back to threads",
                exc_info=True,
            )
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                futures = [pool.submit(_render_pages_batch, batch, **common) for batch in batches]
                fragments = [fut.result() for fut in futures]

    issues: list[QualityIssue] = []
    continuations = 0
    dest = fitz.open()
    try:
        for frag, page_issues, cont in fragments:
            issues.extend(page_issues)
            continuations += cont
            part = fitz.open(stream=frag, filetype="pdf")
            try:
                dest.insert_pdf(part)
            finally:
                part.close()
        try:
            dest.subset_fonts(fallback=True)
        except Exception:
            logger.debug("Font subset skipped", exc_info=True)
        # garbage=4 (not 3) also deduplicates: page-by-page copying embeds the
        # same font programs again per page — 8.6 MB for a 1.6 MB source.
        pdf_bytes = dest.tobytes(deflate=True, garbage=4, use_objstms=1)
    finally:
        dest.close()

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
