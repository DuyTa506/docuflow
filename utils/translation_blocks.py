"""Merge layout-element payloads into translation blocks (PDFMathTranslate-style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.spatial.grouping import (
    assign_column_membership,
    detect_columns_projection,
    group_into_lines,
    group_lines_to_blocks,
)
from utils.translation_elements import is_heading_label, should_skip_label

_PASSTHROUGH_LABELS = frozenset({
    "image",
    "figure",
    "table",
    "equation",
    "formula",
    "isolate_formula",
    "chart",
    "graph",
})


@dataclass
class TranslationBlock:
    """One logical unit to translate or pass through unchanged."""

    block_id: str
    page_number: int
    label: str
    text: str
    bbox: Dict[str, Any]
    passthrough: bool = False
    is_heading: bool = False
    source_payloads: List[dict] = field(default_factory=list)


def is_passthrough_label(label: str) -> bool:
    return (label or "").lower() in _PASSTHROUGH_LABELS


def _payload_bbox(payload: dict) -> dict:
    bbox = payload.get("bbox") or {}
    return {
        "x1": bbox.get("x1", 0),
        "y1": bbox.get("y1", 0),
        "x2": bbox.get("x2", 0),
        "y2": bbox.get("y2", 0),
    }


def _to_grouping_elem(payload: dict) -> dict:
    bbox = _payload_bbox(payload)
    return {
        "_payload": payload,
        "label": payload.get("label") or "text",
        "zone": payload.get("zone") or "main_text",
        "text_content": payload.get("text_content") or "",
        "sequence_order": payload.get("sequence_order", 0),
        "bbox_x1": bbox["x1"],
        "bbox_y1": bbox["y1"],
        "bbox_x2": bbox["x2"],
        "bbox_y2": bbox["y2"],
        "column_index": 0,
    }


def _text_from_elements(elements: List[dict]) -> str:
    if not elements:
        return ""
    lines = group_into_lines(elements)
    line_texts: List[str] = []
    for line in lines:
        line.sort(key=lambda e: e.get("bbox_x1", 0))
        parts = [
            (e.get("text_content") or "").strip()
            for e in line
            if (e.get("text_content") or "").strip()
        ]
        if parts:
            line_texts.append(" ".join(parts))
    return "\n".join(line_texts)


def _single_passthrough_block(payload: dict, idx: int) -> TranslationBlock:
    bbox = _payload_bbox(payload)
    return TranslationBlock(
        block_id=f"pass_{idx}",
        page_number=payload.get("page_number") or 1,
        label=payload.get("label") or "text",
        text=(payload.get("text_content") or "").strip(),
        bbox=bbox,
        passthrough=True,
        source_payloads=[payload],
    )


def _merge_grouping_elements(
    elements: List[dict],
    page_number: int,
    block_counter: int,
) -> List[TranslationBlock]:
    if not elements:
        return []

    page_width = max(int(e.get("bbox_x2", 0)) for e in elements) or 800
    columns = detect_columns_projection(elements, page_width)
    with_cols = assign_column_membership(elements, columns)

    by_col: Dict[int, List[dict]] = {}
    for el in with_cols:
        by_col.setdefault(el.get("column_index", 0), []).append(el)

    blocks: List[TranslationBlock] = []
    for col_idx in sorted(by_col.keys()):
        col_elems = sorted(
            by_col[col_idx],
            key=lambda e: (e.get("bbox_y1", 0), e.get("bbox_x1", 0)),
        )
        segments: List[tuple[str, List[dict]]] = []
        current: List[dict] = []
        for el in col_elems:
            label = el.get("label") or "text"
            if is_heading_label(label):
                if current:
                    segments.append(("text", current))
                    current = []
                segments.append(("heading", [el]))
            else:
                current.append(el)
        if current:
            segments.append(("text", current))

        for seg_type, seg_elems in segments:
            if seg_type == "heading":
                el = seg_elems[0]
                payload = el["_payload"]
                blocks.append(
                    TranslationBlock(
                        block_id=f"blk_{block_counter}",
                        page_number=page_number,
                        label=el.get("label") or "heading",
                        text=(el.get("text_content") or "").strip(),
                        bbox=_payload_bbox(payload),
                        is_heading=True,
                        source_payloads=[payload],
                    )
                )
                block_counter += 1
                continue

            lines = group_into_lines(seg_elems)
            spatial_blocks = group_lines_to_blocks(lines)
            for sb in spatial_blocks:
                if not sb.elements:
                    continue
                payloads = [e["_payload"] for e in sb.elements]
                blocks.append(
                    TranslationBlock(
                        block_id=f"blk_{block_counter}",
                        page_number=page_number,
                        label=sb.block_type if sb.block_type != "text" else "text",
                        text=_text_from_elements(sb.elements),
                        bbox=dict(sb.bbox),
                        source_payloads=payloads,
                    )
                )
                block_counter += 1

    return blocks


def merge_payloads_to_blocks(payloads: List[dict]) -> List[TranslationBlock]:
    """Group element payloads into blocks for batched translation."""
    if not payloads:
        return []

    sorted_payloads = sorted(
        payloads,
        key=lambda p: (p.get("page_number") or 0, p.get("sequence_order") or 0),
    )

    blocks: List[TranslationBlock] = []
    block_counter = 0
    i = 0
    while i < len(sorted_payloads):
        payload = sorted_payloads[i]
        label = payload.get("label") or "text"

        if is_passthrough_label(label) or should_skip_label(label):
            blocks.append(_single_passthrough_block(payload, block_counter))
            block_counter += 1
            i += 1
            continue

        page_number = payload.get("page_number") or 1
        mergeable: List[dict] = []
        j = i
        while j < len(sorted_payloads):
            pj = sorted_payloads[j]
            plabel = pj.get("label") or "text"
            if (pj.get("page_number") or 1) != page_number:
                break
            if is_passthrough_label(plabel) or should_skip_label(plabel):
                break
            mergeable.append(_to_grouping_elem(pj))
            j += 1

        new_blocks = _merge_grouping_elements(mergeable, page_number, block_counter)
        blocks.extend(new_blocks)
        block_counter += len(new_blocks)
        i = j

    return blocks


def block_to_translated_element(block: TranslationBlock, text: str) -> dict:
    """Build one translated-element dict from a merged block."""
    base = dict(block.source_payloads[0]) if block.source_payloads else {}
    out = {
        "page_number": block.page_number,
        "label": block.label,
        "text_content": text,
        "sequence_order": base.get("sequence_order", 0),
        "bbox": dict(block.bbox),
    }
    if base.get("crop_image_key"):
        out["crop_image_key"] = base["crop_image_key"]
    if base.get("crop_image_base64"):
        out["crop_image_base64"] = base["crop_image_base64"]
    return out


def block_to_export_element(block: TranslationBlock) -> dict:
    """Merged layout element for PDF export (original text, union bbox)."""
    return block_to_translated_element(block, block.text)


def merge_elements_for_layout_export(payloads: List[dict]) -> List[dict]:
    """Collapse line-level OCR elements into paragraph blocks for layout PDF."""
    if not payloads:
        return []
    blocks = merge_payloads_to_blocks(payloads)
    return [block_to_export_element(b) for b in blocks]
