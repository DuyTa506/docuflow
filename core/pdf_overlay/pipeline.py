"""
PDF overlay translation pipeline.

Ported from PDFMathTranslate (pdf2zh/high_level.py) — re-draws translated text
at original coordinates while preserving formula glyphs.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from config.settings import settings
from core.pdf_overlay.converter import TranslateConverter
from core.pdf_overlay.interp import PDFPageInterpreterEx
from core.pdf_overlay.layout_model import OnnxModel
from core.pdf_overlay.llm_adapter import OverlayLLMAdapter

logger = logging.getLogger(__name__)

NOTO_NAME = "noto"

_layout_model: OnnxModel | None = None

# Labels that must never be translated/redrawn -- direct equivalent of the
# YOLO-based path's vcls set, mapped onto this project's own layout-element
# label vocabulary (see core/models.py UnifiedElement.to_layout_element_dict).
_RESERVED_LABELS = frozenset({
    "figure",
    "table",
    "equation",
    "image",
    "chart",
    "graph",
    "picture",
    "formula",
    "isolate_formula",
})


def get_layout_model() -> OnnxModel:
    global _layout_model
    if _layout_model is None:
        path = settings.doclayout_model_path or None
        _layout_model = OnnxModel.from_pretrained(path)
    return _layout_model


def _yolo_mask_for_page(pix) -> np.ndarray:
    """Fallback path: run the YOLO layout model on a rendered page image.

    Only used when a document has no already-extracted layout elements
    stored (e.g. translation requested on a document that skipped normal
    extraction) -- the common case reuses build_layout_mask_from_elements
    instead, so this model is loaded lazily and only when actually needed.
    """
    model = get_layout_model()
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
    page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]

    box = np.ones((pix.height, pix.width))
    h, w = box.shape
    vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
    for i, d in enumerate(page_layout.boxes):
        if page_layout.names[int(d.cls)] not in vcls:
            x0, y0, x1, y1 = d.xyxy.squeeze()
            x0, y0, x1, y1 = (
                np.clip(int(x0 - 1), 0, w - 1),
                np.clip(int(h - y1 - 1), 0, h - 1),
                np.clip(int(x1 + 1), 0, w - 1),
                np.clip(int(h - y0 + 1), 0, h - 1),
            )
            box[y0:y1, x0:x1] = i + 2
    for i, d in enumerate(page_layout.boxes):
        if page_layout.names[int(d.cls)] in vcls:
            x0, y0, x1, y1 = d.xyxy.squeeze()
            x0, y0, x1, y1 = (
                np.clip(int(x0 - 1), 0, w - 1),
                np.clip(int(h - y1 - 1), 0, h - 1),
                np.clip(int(x1 + 1), 0, w - 1),
                np.clip(int(h - y0 + 1), 0, h - 1),
            )
            box[y0:y1, x0:x1] = 0
    return box


def _fetch_page_elements(document_id: str) -> dict[int, list[dict]]:
    """Already-extracted layout elements for this document, grouped by page.

    Reuses Docling/OCR's own layout output instead of running a second,
    separate YOLO detection pass -- confirmed live to be at least as
    accurate (correctly reads styled/logo text that YOLO's purely visual
    classifier can misjudge as non-text "abandon", and preserves paragraph
    grouping YOLO tends to over-split). Returns {} if nothing is stored
    (e.g. this document was never extracted), signalling the caller to
    fall back to the YOLO path.
    """
    from data.database import get_db_manager
    from data.repositories import DocumentRepository

    db_manager = get_db_manager()
    by_page: dict[int, list[dict]] = {}
    with db_manager.session() as db:
        repo = DocumentRepository(db)
        elements = repo.get_elements(document_id)
        for e in elements:
            by_page.setdefault(e.page.page_number, []).append({
                "label": e.label,
                "bbox_x1": e.bbox_x1,
                "bbox_y1": e.bbox_y1,
                "bbox_x2": e.bbox_x2,
                "bbox_y2": e.bbox_y2,
            })
    return by_page


def build_layout_mask_from_elements(
    elements: list[dict], page_h: int, page_w: int
) -> np.ndarray:
    """Build the same pixel mask TranslateConverter expects (0 = reserved/
    non-translatable, >=2 = a distinct translatable text box, 1 = untouched
    background) from already-extracted layout elements instead of a fresh
    YOLO detection pass on a rendered image.

    Coordinate handling mirrors the YOLO path exactly: stored bboxes are in
    top-down (image-like) space, same as YOLO's raw detections, so the same
    y-flip converts them into the bottom-up indexing TranslateConverter's
    pdfminer-based character lookup (`child.y0`) expects.
    """
    box = np.ones((page_h, page_w))
    h, w = box.shape

    def _clip_bbox(elem: dict) -> tuple[int, int, int, int]:
        x0 = elem.get("bbox_x1") or 0
        y0 = elem.get("bbox_y1") or 0
        x1 = elem.get("bbox_x2") or 0
        y1 = elem.get("bbox_y2") or 0
        return (
            np.clip(int(x0 - 1), 0, w - 1),
            np.clip(int(h - y1 - 1), 0, h - 1),
            np.clip(int(x1 + 1), 0, w - 1),
            np.clip(int(h - y0 + 1), 0, h - 1),
        )

    for i, elem in enumerate(elements):
        if (elem.get("label") or "").lower() not in _RESERVED_LABELS:
            x0, y0, x1, y1 = _clip_bbox(elem)
            box[y0:y1, x0:x1] = i + 2
    for i, elem in enumerate(elements):
        if (elem.get("label") or "").lower() in _RESERVED_LABELS:
            x0, y0, x1, y1 = _clip_bbox(elem)
            box[y0:y1, x0:x1] = 0

    return box


def _download_target_font(lang: str) -> str:
    """Return a font file path for the target language.

    Vietnamese gets a dedicated, much smaller font (Times New Roman --
    confirmed to have full coverage of Vietnamese precomposed characters,
    ~322KB vs. GoNotoKurrent's ~15MB unsubsetted) instead of the multi-script
    fallback, matching the same convention already used for DOCX exports.
    GoNotoKurrent remains the fallback for every other supported script
    (zh/ja/ko/ar/hi/th/etc., which Times New Roman has no glyphs for at all).
    """
    import os

    code = (lang or "").lower().split("-")[0]
    if code == "vi" and os.path.isfile(settings.pdf_overlay_vi_font_path):
        return settings.pdf_overlay_vi_font_path

    try:
        from babeldoc.assets.assets import get_font_and_metadata

        font_name = "GoNotoKurrent-Regular.ttf"
        font_path, _ = get_font_and_metadata(font_name)
        return font_path.as_posix()
    except Exception:
        return ""


_OVERLAY_FONT_LANG_PREFIXES = frozenset(
    {"vi", "zh", "ja", "ko", "th", "ar", "he", "hi", "bn", "ta", "te", "km", "lo"}
)


def _overlay_requires_target_font(lang: str) -> bool:
    code = (lang or "").lower().split("-")[0]
    return code in _OVERLAY_FONT_LANG_PREFIXES


def translate_pdf_bytes(
    pdf_bytes: bytes,
    *,
    lang_in: str,
    lang_out: str,
    llm_adapter: OverlayLLMAdapter,
    document_id: str | None = None,
    thread: int | None = None,
    on_progress: Callable[..., None] | None = None,
) -> bytes:
    """
    Translate a text-layer PDF and return mono-language PDF bytes.

    Args:
        pdf_bytes: Raw PDF file content.
        lang_in: Source language code.
        lang_out: Target language code.
        llm_adapter: Sync paragraph translator.
        document_id: When given, reuse this document's already-extracted
            layout elements for the translatable/reserved-region mask
            instead of running a second YOLO detection pass. Falls back to
            YOLO per-page if no elements are stored (e.g. document was
            never extracted through the normal pipeline).
        thread: Worker threads for parallel paragraph translation.
        on_progress: Optional callback(done_pages, total_pages).

    Returns:
        Translated PDF as bytes.
    """
    thread = thread if thread is not None else settings.pdf_overlay_threads

    font_path = _download_target_font(lang_out)
    if _overlay_requires_target_font(lang_out) and not font_path:
        raise RuntimeError(
            f"Overlay target font unavailable for language '{lang_out}' — install babeldoc fonts or use element/flat translation"
        )
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path) if font_path else None
    font_list = [("tiro", None)]
    if font_path:
        font_list.append((noto_name, font_path))

    doc_zh = Document(stream=pdf_bytes)
    page_count = doc_zh.page_count
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            if font[1] is not None or font[0] == "tiro":
                font_id[font[0]] = page.insert_font(font[0], font[1])

    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:
            try:
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search(r"(\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""
                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()
    doc_zh.save(fp)
    obj_patch = _translate_patch(
        fp,
        doc_zh=doc_zh,
        document_id=document_id,
        lang_in=lang_in,
        lang_out=lang_out,
        llm_adapter=llm_adapter,
        noto_name=noto_name,
        noto=noto,
        thread=thread,
        on_progress=on_progress,
    )

    for obj_id, ops_new in obj_patch.items():
        doc_zh.update_stream(obj_id, ops_new.encode())

    try:
        doc_zh.subset_fonts(fallback=True)
    except Exception as exc:
        logger.warning("Font subset skipped: %s", exc)
    return doc_zh.tobytes(deflate=True, garbage=3, use_objstms=1)


def _translate_patch(
    inf: io.BytesIO,
    *,
    doc_zh: Document,
    document_id: str | None,
    lang_in: str,
    lang_out: str,
    llm_adapter: OverlayLLMAdapter,
    noto_name: str,
    noto: Font | None,
    thread: int,
    on_progress: Callable[[int, int], None] | None,
) -> dict:
    rsrcmgr = PDFResourceManager()
    layout: dict = {}
    obj_patch: dict = {}
    para_state = {"done": 0, "total": 0}

    elements_by_page = _fetch_page_elements(document_id) if document_id else {}

    def _on_para(done: int, total: int) -> None:
        para_state["done"] = done
        para_state["total"] = total

    device = TranslateConverter(
        rsrcmgr,
        thread=thread,
        layout=layout,
        lang_in=lang_in,
        lang_out=lang_out,
        noto_name=noto_name,
        noto=noto,
        llm_adapter=llm_adapter,
        on_para_progress=_on_para,
    )

    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    total_pages = doc_zh.page_count
    page_limit = settings.pdf_overlay_max_pages or total_pages
    page_limit = min(page_limit, total_pages)
    done = 0

    for pageno, page in enumerate(PDFPage.create_pages(doc)):
        if pageno >= page_limit:
            break
        page.pageno = pageno
        pix = doc_zh[pageno].get_pixmap()

        page_elements = elements_by_page.get(pageno + 1)  # stored pages are 1-indexed
        if page_elements:
            box = build_layout_mask_from_elements(page_elements, pix.height, pix.width)
        else:
            box = _yolo_mask_for_page(pix)
        layout[pageno] = box

        page.page_xref = doc_zh.get_new_xref()
        doc_zh.update_object(page.page_xref, "<<>>")
        doc_zh.update_stream(page.page_xref, b"")
        doc_zh[pageno].set_contents(page.page_xref)
        interpreter.process_page(page)

        done += 1
        if on_progress:
            on_progress(done, page_limit, para_state.get("done", 0), para_state.get("total", 0))

    device.close()
    return obj_patch
