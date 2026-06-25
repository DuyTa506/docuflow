"""Helpers for structure-preserving translation element payloads."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional


_SKIP_LABELS = frozenset({"image", "figure"})
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
    ):
        self.label = label
        self.text_content = text_content
        self.page_number = page_number
        self.page = page
        self.bbox_x1 = bbox_x1
        self.bbox_y1 = bbox_y1
        self.bbox_x2 = bbox_x2
        self.bbox_y2 = bbox_y2


def layout_element_to_dict(elem, page_number: int) -> dict:
    """Serialize a LayoutElement ORM row to a translation payload dict."""
    return {
        "page_number": page_number,
        "label": elem.label,
        "text_content": elem.text_content or "",
        "sequence_order": elem.sequence_order,
        "bbox": {
            "x1": elem.bbox_x1,
            "y1": elem.bbox_y1,
            "x2": elem.bbox_x2,
            "y2": elem.bbox_y2,
        },
    }


def elements_to_views(elements: Iterable[Any]) -> List[TranslatedElementView]:
    """Convert stored JSON dicts or ORM rows into views for DOCX rendering."""
    views: List[TranslatedElementView] = []
    for elem in elements:
        if isinstance(elem, dict):
            bbox = elem.get("bbox") or {}
            views.append(
                TranslatedElementView(
                    label=elem.get("label", "text"),
                    text_content=elem.get("text_content", ""),
                    page_number=elem.get("page_number"),
                    bbox_x1=bbox.get("x1"),
                    bbox_y1=bbox.get("y1"),
                    bbox_x2=bbox.get("x2"),
                    bbox_y2=bbox.get("y2"),
                )
            )
        else:
            page_num = None
            page_rel = getattr(elem, "page", None)
            if page_rel is not None:
                page_num = getattr(page_rel, "page_number", None)
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
