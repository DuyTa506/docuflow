"""Helpers for structure-preserving translation element payloads."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional


_SKIP_LABELS = frozenset({"image", "figure", "chart", "graph", "picture"})
_HEADING_LABELS = frozenset({"title", "sub_title", "heading"})


class TranslatedElementView:
    """Lightweight view for render_layout_elements_to_docx (dict or ORM)."""

    def __init__(
        self,
        *,
        label: str,
        text_content: str,
        page_number: Optional[int] = None,
        page=None,
        bbox_x1: Optional[int] = None,
        bbox_y1: Optional[int] = None,
        bbox_x2: Optional[int] = None,
        bbox_y2: Optional[int] = None,
        crop_image_key: Optional[str] = None,
        crop_image_base64: Optional[str] = None,
        page_image_key: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        self.label = label
        self.text_content = text_content
        self.page_number = page_number
        self.page = page
        self.bbox_x1 = bbox_x1
        self.bbox_y1 = bbox_y1
        self.bbox_x2 = bbox_x2
        self.bbox_y2 = bbox_y2
        # Image sources for structure-preserving export (figures/charts).
        self.crop_image_key = crop_image_key
        self.crop_image_base64 = crop_image_base64
        self.page_image_key = page_image_key
        self.image_width = image_width
        self.image_height = image_height


def layout_element_to_dict(elem, page_number: int) -> dict:
    """Serialize a LayoutElement ORM row to a translation payload dict."""
    return {
        "page_number": page_number,
        "label": elem.label,
        "text_content": elem.text_content or "",
        "sequence_order": elem.sequence_order,
        "crop_image_key": getattr(elem, "crop_image_key", None),
        "bbox": {
            "x1": elem.bbox_x1,
            "y1": elem.bbox_y1,
            "x2": elem.bbox_x2,
            "y2": elem.bbox_y2,
        },
    }


def elements_to_views(
    elements: Iterable[Any],
    *,
    document_id: Optional[str] = None,
) -> List[TranslatedElementView]:
    """Convert stored JSON dicts or ORM rows into views for DOCX rendering.

    ``document_id`` lets dict payloads (translated_elements) reconstruct the
    per-page image key so figures can be cropped from the page render at export.
    """
    from utils.storage_keys import page_image_key as _page_image_key

    views: List[TranslatedElementView] = []
    for elem in elements:
        if isinstance(elem, dict):
            bbox = elem.get("bbox") or {}
            page_num = elem.get("page_number")
            page_img_key = elem.get("page_image_key")
            if not page_img_key and document_id and page_num is not None:
                page_img_key = _page_image_key(document_id, page_num)
            views.append(
                TranslatedElementView(
                    label=elem.get("label", "text"),
                    text_content=elem.get("text_content", ""),
                    page_number=page_num,
                    bbox_x1=bbox.get("x1"),
                    bbox_y1=bbox.get("y1"),
                    bbox_x2=bbox.get("x2"),
                    bbox_y2=bbox.get("y2"),
                    crop_image_key=elem.get("crop_image_key"),
                    crop_image_base64=elem.get("crop_image_base64"),
                    page_image_key=page_img_key,
                )
            )
        else:
            page_num = None
            page_rel = getattr(elem, "page", None)
            page_img_key = None
            img_w = img_h = None
            if page_rel is not None:
                page_num = getattr(page_rel, "page_number", None)
                page_img_key = getattr(page_rel, "image_key", None)
                img_w = getattr(page_rel, "image_width", None)
                img_h = getattr(page_rel, "image_height", None)
            views.append(
                TranslatedElementView(
                    label=getattr(elem, "label", "text"),
                    text_content=getattr(elem, "text_content", "") or "",
                    page_number=page_num,
                    page=page_rel,
                    bbox_x1=getattr(elem, "bbox_x1", None),
                    bbox_y1=getattr(elem, "bbox_y1", None),
                    bbox_x2=getattr(elem, "bbox_x2", None),
                    bbox_y2=getattr(elem, "bbox_y2", None),
                    crop_image_key=getattr(elem, "crop_image_key", None),
                    crop_image_base64=getattr(elem, "crop_image_base64", None),
                    page_image_key=page_img_key,
                    image_width=img_w,
                    image_height=img_h,
                )
            )
    return views


def flatten_translated_elements(elements: Iterable[dict]) -> str:
    """Join translated blocks in reading order for search / legacy API."""
    parts = []
    for elem in elements:
        text = (elem.get("text_content") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def serialize_translated_elements(elements: List[dict]) -> str:
    return json.dumps(elements, ensure_ascii=False)


def deserialize_translated_elements(raw: Optional[str]) -> List[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def should_skip_label(label: str) -> bool:
    return (label or "").lower() in _SKIP_LABELS


def is_heading_label(label: str) -> bool:
    return (label or "").lower() in _HEADING_LABELS
