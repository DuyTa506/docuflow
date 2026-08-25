"""DeepSeek formula enrichment for Docling text-layer pages.

Docling remains responsible for layout detection.  Its CPU CodeFormula model is
deliberately disabled; detected formula regions are recognized by the already
loaded DeepSeek-OCR server instead.  Tight crops are best for short formulas,
while context-heavy/multiline pages use one full-page OCR request.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import replace
from typing import Iterable, Sequence

from config.settings import settings
from core.models import ServicePageResult, UnifiedElement

logger = logging.getLogger(__name__)

_FORMULA_LABELS = frozenset({"formula", "equation", "isolate_formula", "isolated_formula", "math"})
_MATH_MARKER_RE = re.compile(r"(?:\$\$|\\\[|\\\(|\\begin\s*\{)")


def is_complex_formula_page(
    elements: Sequence[UnifiedElement],
    page_height: float,
) -> bool:
    """Return whether formula recognition needs full-page context."""
    formulas = [element for element in elements if element.element_type == "equation"]
    if len(formulas) >= max(1, int(settings.formula_complex_min_count)):
        return True

    height = max(float(page_height), 1.0)
    for formula in formulas:
        if len(formula.text.strip()) >= max(1, int(settings.formula_complex_min_chars)):
            return True
        bbox = formula.bbox or {}
        bbox_height = max(0.0, float(bbox.get("y2", 0)) - float(bbox.get("y1", 0)))
        if bbox_height / height >= max(0.0, float(settings.formula_complex_min_height_ratio)):
            return True
    return False


def _normalize_formula(text: str) -> str:
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$"):
        return text
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    elif text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    return f"$${text}$$"


def _raw_text(element: dict) -> str:
    return str(
        element.get("text_full") or element.get("text_content") or element.get("text") or ""
    ).strip()


def _raw_bbox(element: dict) -> dict[str, float]:
    return {
        "x1": float(element.get("bbox_x1", element.get("x1", 0)) or 0),
        "y1": float(element.get("bbox_y1", element.get("y1", 0)) or 0),
        "x2": float(element.get("bbox_x2", element.get("x2", 0)) or 0),
        "y2": float(element.get("bbox_y2", element.get("y2", 0)) or 0),
    }


def _candidate_payloads(
    result: ServicePageResult,
    *,
    accept_single_plain_text: bool,
    include_marked_text: bool = True,
) -> list[tuple[str, str, dict[str, float]]]:
    payloads: list[tuple[str, str, dict[str, float]]] = []
    nonempty: list[tuple[str, str, dict[str, float]]] = []

    for raw in result.layout_elements or []:
        label = str(raw.get("label", "text")).lower()
        text = _raw_text(raw)
        if not text or label in {"image", "figure"}:
            continue
        payload = (label, text, _raw_bbox(raw))
        nonempty.append(payload)
        if label in _FORMULA_LABELS or label == "code":
            payloads.append(payload)
        elif include_marked_text and ("```" in text or _MATH_MARKER_RE.search(text)):
            payloads.append(payload)

    if not payloads and accept_single_plain_text and len(nonempty) == 1:
        payloads = nonempty
    return payloads


def _render_crop(
    file_path: str,
    page_number: int,
    bbox: dict,
    dpi: int,
) -> str:
    import fitz

    doc = fitz.open(file_path)
    try:
        page = doc[page_number - 1]
        page_rect = page.rect
        # A tiny anti-clipping pad is useful; a broad expansion was measured to
        # pull surrounding prose into the formula and reduce recognition quality.
        pad = 1.5
        x1, y1, x2, y2 = (
            float(bbox.get("x1", 0)),
            float(bbox.get("y1", 0)),
            float(bbox.get("x2", 0)),
            float(bbox.get("y2", 0)),
        )
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        clip = fitz.Rect(
            max(page_rect.x0, x1 - pad),
            max(page_rect.y0, y1 - pad),
            min(page_rect.x1, x2 + pad),
            min(page_rect.y1, y2 + pad),
        )
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            raise ValueError(f"Invalid formula bbox on page {page_number}: {bbox}")

        scale = max(72, int(dpi)) / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(pixmap.tobytes("png"))
            return tmp.name
    finally:
        doc.close()


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_y_center(bbox: dict[str, float]) -> float:
    return (float(bbox.get("y1", 0)) + float(bbox.get("y2", 0))) / 2.0


def _pair_formulas_spatially(
    docling_eqs: Sequence[UnifiedElement],
    ocr_formulas: Sequence[UnifiedElement],
    *,
    iou_threshold: float = 0.15,
) -> tuple[dict[int, UnifiedElement], list[UnifiedElement]]:
    """Greedy spatial match of OCR formulas onto Docling equations.

    Returns ``(docling_index → replacement, unmatched_ocr)``. Unmatched
    Docling equations keep their original text as fallback.
    """
    if not ocr_formulas:
        return {}, []

    pairs: list[tuple[float, int, int]] = []
    for di, doc_eq in enumerate(docling_eqs):
        doc_bbox = doc_eq.bbox or {}
        for oi, ocr_eq in enumerate(ocr_formulas):
            ocr_bbox = ocr_eq.bbox or {}
            iou = _bbox_iou(
                {
                    "x1": float(doc_bbox.get("x1", 0)),
                    "y1": float(doc_bbox.get("y1", 0)),
                    "x2": float(doc_bbox.get("x2", 0)),
                    "y2": float(doc_bbox.get("y2", 0)),
                },
                {
                    "x1": float(ocr_bbox.get("x1", 0)),
                    "y1": float(ocr_bbox.get("y1", 0)),
                    "x2": float(ocr_bbox.get("x2", 0)),
                    "y2": float(ocr_bbox.get("y2", 0)),
                },
            )
            if iou >= iou_threshold:
                score = iou
            else:
                # Prefer nearest Y when boxes barely overlap (common after OCR
                # scale drift) — still better than dumping everything at slot 0.
                dy = abs(_bbox_y_center(doc_bbox) - _bbox_y_center(ocr_bbox))
                score = -dy
            pairs.append((score, di, oi))

    pairs.sort(key=lambda item: item[0], reverse=True)
    used_doc: set[int] = set()
    used_ocr: set[int] = set()
    matched: dict[int, UnifiedElement] = {}
    for score, di, oi in pairs:
        if di in used_doc or oi in used_ocr:
            continue
        # Reject distant Y-only matches once every Docling eq has a nearer peer.
        if score < 0 and abs(score) > 80:
            continue
        used_doc.add(di)
        used_ocr.add(oi)
        ocr_eq = ocr_formulas[oi]
        matched[di] = replace(
            docling_eqs[di],
            text=ocr_eq.text,
            element_type="equation",
            source="deepseek_formula",
            bbox=ocr_eq.bbox or docling_eqs[di].bbox,
        )

    unmatched = [ocr_formulas[i] for i in range(len(ocr_formulas)) if i not in used_ocr]
    return matched, unmatched


def _sort_reading_order(elements: Sequence[UnifiedElement]) -> list[UnifiedElement]:
    ordered = sorted(
        elements,
        key=lambda el: (
            float((el.bbox or {}).get("y1", 0)),
            float((el.bbox or {}).get("x1", 0)),
            int(el.order or 0),
        ),
    )
    for index, element in enumerate(ordered):
        element.order = index
    return ordered


def _elements_markdown(elements: Iterable[UnifiedElement]) -> str:
    blocks: list[str] = []
    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        if element.element_type == "heading":
            text = f"{'#' * max(1, min(6, int(element.level or 1)))} {text}"
        blocks.append(text)
    return "\n\n".join(blocks)


def merge_formula_markdown(
    page_markdown: str,
    original: Sequence[UnifiedElement],
    enriched: Sequence[UnifiedElement],
) -> str:
    """Replace only formula blocks while retaining Docling's other markdown."""
    old_formulas = [element for element in original if element.element_type == "equation"]
    replacements = [element for element in enriched if element.source == "deepseek_formula"]
    if not old_formulas or not replacements:
        return page_markdown

    if len(original) == len(enriched):
        merged = page_markdown
        changed = False
        for index, old in enumerate(original):
            if old.element_type != "equation":
                continue
            new = enriched[index]
            if new.source != "deepseek_formula":
                continue
            updated = merged.replace(old.text, new.text, 1)
            changed = changed or updated != merged
            merged = updated
        return merged if changed else _elements_markdown(enriched)

    merged = page_markdown
    marker = "\x00DEEPSEEK_FORMULAS\x00"
    inserted = False
    for old in old_formulas:
        replacement = marker if not inserted else ""
        updated = merged.replace(old.text, replacement, 1)
        if updated != merged:
            inserted = True
            merged = updated
    if not inserted:
        return _elements_markdown(enriched)

    formula_markdown = "\n\n".join(element.text for element in replacements if element.text)
    return merged.replace(marker, formula_markdown).strip()


class DeepSeekFormulaEnricher:
    """Recognize Docling formula regions with bounded DeepSeek requests."""

    def __init__(self, client, file_path: str):
        self.client = client
        self.file_path = file_path
        self._semaphore = asyncio.Semaphore(max(1, int(settings.formula_ocr_parallelism)))

    async def _extract(self, file_path: str, page_number: int):
        from services.extractors.ocr_extractor import OcrExtractor
        from services.ocr_limiter import ocr_request_slot

        extractor = OcrExtractor(self.client, file_path)
        async with self._semaphore, ocr_request_slot():
            elements = await extractor.extract_page(page_number)
        return elements, extractor.page_result

    async def _enrich_crop(
        self,
        target: UnifiedElement,
        page_number: int,
    ) -> UnifiedElement:
        tmp_path = await asyncio.to_thread(
            _render_crop,
            self.file_path,
            page_number,
            target.bbox or {},
            settings.formula_crop_dpi,
        )
        try:
            _elements, result = await self._extract(tmp_path, 1)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

        if result is None:
            return target
        payloads = _candidate_payloads(result, accept_single_plain_text=True)
        if not payloads:
            return target

        all_formulas = all(label in _FORMULA_LABELS for label, _text, _bbox in payloads)
        texts = [
            _normalize_formula(text) if label in _FORMULA_LABELS else text
            for label, text, _bbox in payloads
        ]
        return replace(
            target,
            element_type="equation" if all_formulas else "text",
            text="\n\n".join(texts),
            source="deepseek_formula",
        )

    async def _enrich_full_page(
        self,
        elements: Sequence[UnifiedElement],
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> list[UnifiedElement]:
        _ocr_elements, result = await self._extract(self.file_path, page_number)
        if result is None:
            return list(elements)

        payloads = _candidate_payloads(
            result,
            accept_single_plain_text=False,
            include_marked_text=False,
        )
        # Only formula-labeled OCR payloads participate in the merge. Plain
        # text from full-page OCR must not displace Docling prose.
        formula_payloads = [
            (label, text, bbox) for label, text, bbox in payloads if label in _FORMULA_LABELS
        ]
        if not formula_payloads:
            return list(elements)

        from utils.image_utils import decode_base64_image

        image = decode_base64_image(result.image_base64)
        image_width, image_height = image.size
        scale_x = float(page_width) / max(1, image_width)
        scale_y = float(page_height) / max(1, image_height)

        ocr_formulas: list[UnifiedElement] = []
        for order, (_label, text, bbox) in enumerate(formula_payloads):
            ocr_formulas.append(
                UnifiedElement(
                    element_type="equation",
                    text=_normalize_formula(text),
                    page_number=page_number,
                    order=order,
                    source="deepseek_formula",
                    bbox={
                        "x1": bbox["x1"] * scale_x,
                        "y1": bbox["y1"] * scale_y,
                        "x2": bbox["x2"] * scale_x,
                        "y2": bbox["y2"] * scale_y,
                    },
                )
            )

        docling_eqs = [el for el in elements if el.element_type == "equation"]
        matched, unmatched_ocr = _pair_formulas_spatially(docling_eqs, ocr_formulas)

        merged: list[UnifiedElement] = []
        eq_index = 0
        for element in elements:
            if element.element_type == "equation":
                merged.append(matched.get(eq_index, element))
                eq_index += 1
            else:
                merged.append(element)

        # Extra OCR formulas (no Docling peer) insert by Y, not as a block at
        # the first equation slot — that was the DOC_016 p.122 ordering bug.
        merged.extend(unmatched_ocr)
        return _sort_reading_order(merged)

    async def enrich_page(
        self,
        elements: Sequence[UnifiedElement],
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> list[UnifiedElement]:
        formulas = [element for element in elements if element.element_type == "equation"]
        if not formulas or not settings.deepseek_formula_enrichment:
            return list(elements)

        try:
            if is_complex_formula_page(elements, page_height):
                return await self._enrich_full_page(
                    elements,
                    page_number,
                    page_width,
                    page_height,
                )

            replacements = await asyncio.gather(
                *(self._enrich_crop(formula, page_number) for formula in formulas)
            )
            by_id = {
                id(formula): replacement for formula, replacement in zip(formulas, replacements)
            }
            return [by_id.get(id(element), element) for element in elements]
        except Exception:
            # Formula recognition is an enrichment, never a reason to lose the
            # already extracted page or invalidate its retry checkpoint.
            logger.warning(
                "DeepSeek formula enrichment failed on page %s; using Docling text fallback",
                page_number,
                exc_info=True,
            )
            return list(elements)
