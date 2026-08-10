"""
Layout-preserving PDF extraction via IBM Docling (full pipeline).

Uses DocumentConverter for reading order, table structure, and section
detection. Output is adapted to UnifiedElement for the existing storage
and export pipeline.
"""

from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from core.models import UnifiedElement

logger = logging.getLogger(__name__)

_HEADING_LABELS = frozenset({"section_header", "title", "field_heading"})
_FIGURE_LABELS = frozenset({"picture", "chart"})
_SKIP_LABELS = frozenset({"page_header", "page_footer", "document_index", "caption"})
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def prov_bbox_to_top_left(bbox, page_height: float) -> dict:
    """Convert docling BOTTOMLEFT bbox (l, t, r, b) → top-left {x1,y1,x2,y2}."""
    return {
        "x1": float(bbox.l),
        "y1": float(page_height - bbox.t),
        "x2": float(bbox.r),
        "y2": float(page_height - bbox.b),
    }


def _item_label_str(item) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    return label.value if hasattr(label, "value") else str(label)


def _item_self_ref(item) -> str:
    ref = getattr(item, "self_ref", None)
    return str(ref) if ref else ""


def _item_text(item, doc) -> str:
    text = getattr(item, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    if hasattr(item, "export_to_markdown"):
        try:
            exported = item.export_to_markdown(doc)
            if exported and str(exported).strip():
                return str(exported).strip()
        except Exception:
            logger.debug("export_to_markdown failed for %s", type(item).__name__, exc_info=True)
    return ""


def _formula_text(item, doc) -> str:
    for attr in ("latex", "text", "orig"):
        val = getattr(item, attr, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return _item_text(item, doc)


def _clean_figure_text(text: str) -> str:
    if not text:
        return ""
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MD_IMAGE_RE.sub("", text)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip()).strip()


def _build_caption_map(doc) -> Dict[str, str]:
    """Map picture self_ref (e.g. #/pictures/1) → caption text."""
    captions: Dict[str, str] = {}
    for item, _level in doc.iterate_items():
        if _item_label_str(item) != "caption":
            continue
        parent = getattr(item, "parent", None)
        if parent is None:
            continue
        cref = getattr(parent, "cref", None) or str(parent)
        text = _item_text(item, doc)
        if cref and text:
            captions[cref] = text
    return captions


def _pil_to_jpeg_b64(pil) -> Tuple[str, int, int]:
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil.width, pil.height


def _extract_picture_image(item, doc) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    get_image = getattr(item, "get_image", None)
    if not callable(get_image):
        return None, None, None
    try:
        pil = get_image(doc)
        if pil is None:
            return None, None, None
        return _pil_to_jpeg_b64(pil)
    except Exception:
        logger.debug("picture get_image failed", exc_info=True)
        return None, None, None


def _is_tiny_picture(width: Optional[int], height: Optional[int], min_px: int) -> bool:
    if width is None or height is None:
        return False
    return width < min_px and height < min_px


def _figure_display_text(caption: str, raw_text: str) -> str:
    if caption:
        return caption
    cleaned = _clean_figure_text(raw_text)
    if cleaned and not cleaned.startswith("!["):
        return cleaned
    return "(img_content)[figure]"


def _map_element_type(label: str, heading_level: Optional[int]) -> str:
    if label in _FIGURE_LABELS:
        return "figure"
    if label == "table":
        return "table"
    if label == "formula":
        return "equation"
    if label in _HEADING_LABELS or heading_level is not None:
        return "heading"
    return "text"


class DoclingLayoutExtractor:
    """Extract structured layout from text-layer PDFs using IBM Docling."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._converter = None
        self._result = None
        self._document = None
        self._page_range: Optional[Tuple[int, int]] = None
        self._elements_by_page: Dict[int, List[UnifiedElement]] = {}

    def _build_converter(self):
        """Build the DocumentConverter once — it caches its own pipeline, so
        reusing the instance across page ranges keeps the models loaded."""
        if self._converter is not None:
            return self._converter

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        from config.settings import settings

        pipeline_opts = PdfPipelineOptions(
            do_ocr=settings.docling_do_ocr,
            do_table_structure=settings.docling_table_structure,
            generate_picture_images=settings.docling_generate_picture_images,
            do_formula_enrichment=settings.docling_do_formula_enrichment,
            images_scale=settings.docling_images_scale,
        )
        if settings.docling_artifacts_path:
            pipeline_opts.artifacts_path = settings.docling_artifacts_path

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            }
        )
        return self._converter

    def convert(self, page_range: Optional[Tuple[int, int]] = None) -> None:
        """Convert the PDF — all of it, or one inclusive page range — and
        cache per-page elements for whatever was converted.

        A range REPLACES the cache rather than adding to it, so a caller
        walking a book range by range must read each range's pages back before
        asking for the next one. That is the trade that makes progress
        possible: one call over 761 pages is a single opaque hour with nothing
        to report and nothing persisted, and Docling gives no per-page hook.

        Page numbers stay absolute under `page_range` and per-page markdown is
        identical to a full-document conversion — both verified against the
        installed Docling before this was relied on.
        """
        if self._document is not None and page_range == self._page_range:
            return

        converter = self._build_converter()
        kwargs = {"page_range": page_range} if page_range else {}
        self._result = converter.convert(self.file_path, **kwargs)
        self._document = self._result.document
        self._page_range = page_range
        self._build_page_cache()

    def _ensure_converted(self) -> None:
        """Convert lazily for readers — but never re-convert the whole book
        just because the cache currently holds a range."""
        if self._document is None:
            self.convert()

    @property
    def total_pages(self) -> int:
        self._ensure_converted()
        return len(self._document.pages)

    def page_size(self, page_number: int) -> Tuple[float, float]:
        self._ensure_converted()
        page = self._document.pages.get(page_number)
        if page is None:
            return 595.0, 842.0
        return float(page.size.width), float(page.size.height)

    def extract_page(self, page_number: int) -> List[UnifiedElement]:
        self._ensure_converted()
        return list(self._elements_by_page.get(page_number, []))

    def page_markdown(self, page_number: int) -> str:
        self._ensure_converted()
        return self._document.export_to_markdown(page_no=page_number) or ""

    def _build_page_cache(self) -> None:
        from config.settings import settings

        doc = self._document
        pages: Dict[int, List[UnifiedElement]] = {}
        caption_map = _build_caption_map(doc)
        min_picture_px = settings.docling_min_picture_px

        for item, level in doc.iterate_items():
            prov_list = getattr(item, "prov", None) or []
            if not prov_list:
                continue

            label = _item_label_str(item)
            if label in _SKIP_LABELS:
                continue

            if label in _FIGURE_LABELS:
                self_ref = _item_self_ref(item)
                caption = caption_map.get(self_ref, "")
                raw_text = _item_text(item, doc)
                text = _figure_display_text(caption, raw_text)
                image_b64, img_w, img_h = _extract_picture_image(item, doc)
                if _is_tiny_picture(img_w, img_h, min_picture_px):
                    continue
                element_type = "figure"
                heading_level = None
            elif label == "formula":
                text = _formula_text(item, doc)
                if not text:
                    continue
                element_type = "equation"
                heading_level = None
                image_b64, img_w, img_h = None, None, None
                if not text.startswith("$"):
                    from utils.math_omml import wrap_as_equation_markdown

                    text = wrap_as_equation_markdown(text)
            else:
                text = _item_text(item, doc)
                if not text:
                    continue
                heading_level = None
                if label in _HEADING_LABELS:
                    heading_level = min(max(int(level), 1), 6)
                element_type = _map_element_type(label, heading_level)
                image_b64, img_w, img_h = None, None, None

            for prov in prov_list:
                page_no = prov.page_no
                page_h = float(doc.pages[page_no].size.height)
                bbox = prov_bbox_to_top_left(prov.bbox, page_h)
                pages.setdefault(page_no, []).append(
                    UnifiedElement(
                        element_type=element_type,
                        text=text,
                        page_number=page_no,
                        order=len(pages[page_no]),
                        source="docling_layout",
                        level=heading_level if element_type == "heading" else None,
                        bbox=bbox,
                        image_bytes_b64=image_b64,
                        image_width=img_w,
                        image_height=img_h,
                    )
                )

        self._elements_by_page = pages
