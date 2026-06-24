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
    ):
        self.label = label
        self.text_content = text_content
        self.page_number = page_number
        self.page = page


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
            views.append(
                TranslatedElementView(
                    label=elem.get("label", "text"),
                    text_content=elem.get("text_content", ""),
                    page_number=elem.get("page_number"),
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
