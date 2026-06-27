"""Preview / truncation helpers for large document text."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def preview_from_page_markdown(
    pages: Sequence[Any],
    preview_pages: int | None,
    *,
    field: str = "markdown_content",
    separator: str = "\n\n---\n\n",
) -> tuple[str, bool, int]:
    total = len(pages)
    if not pages:
        return "", False, 0
    if preview_pages is None or preview_pages <= 0 or preview_pages >= total:
        content = separator.join((getattr(p, field) or "") for p in pages)
        return content, False, total
    selected = pages[:preview_pages]
    content = separator.join((getattr(p, field) or "") for p in selected)
    return content, True, total


def preview_flat_text(
    text: str,
    preview_pages: int | None,
    *,
    chars_per_page: int = 3000,
) -> tuple[str, bool]:
    if not text:
        return "", False
    if preview_pages is None or preview_pages <= 0:
        return text, False
    limit = preview_pages * chars_per_page
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n…", True


def preview_translated_elements(
    elements: Iterable[dict],
    preview_pages: int | None,
) -> tuple[str, bool]:
    elems = list(elements)
    if not elems:
        return "", False
    if preview_pages is None or preview_pages <= 0:
        return _flatten_elements(elems), False
    max_page = max((e.get("page_number") or 1) for e in elems)
    if preview_pages >= max_page:
        return _flatten_elements(elems), False
    filtered = [e for e in elems if (e.get("page_number") or 1) <= preview_pages]
    return _flatten_elements(filtered), True


def _flatten_elements(elements: list[dict]) -> str:
    parts = []
    for e in elements:
        text = e.get("text_content") or e.get("text") or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)
