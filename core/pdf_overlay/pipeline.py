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


def get_layout_model() -> OnnxModel:
    global _layout_model
    if _layout_model is None:
        path = settings.doclayout_model_path or None
        _layout_model = OnnxModel.from_pretrained(path)
    return _layout_model


def _download_target_font(lang: str) -> str:
    """Return a font file path for the target language."""
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
        thread: Worker threads for parallel paragraph translation.
        on_progress: Optional callback(done_pages, total_pages).

    Returns:
        Translated PDF as bytes.
    """
    thread = thread if thread is not None else settings.pdf_overlay_threads
    model = get_layout_model()

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
        model=model,
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
    model: OnnxModel,
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
