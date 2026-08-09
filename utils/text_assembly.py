"""Helpers for assembling document text from per-page storage."""

from __future__ import annotations

from typing import Iterable, Protocol


class _PageLike(Protocol):
    page_number: int
    markdown_content: str


def assemble_ocr_from_pages(pages: Iterable[_PageLike]) -> str:
    """Build full-book OCR text from ordered page markdown."""
    ordered = sorted(pages, key=lambda p: p.page_number)
    return "\n\n---\n\n".join(
        f"# Page {p.page_number}\n\n{p.markdown_content or ''}" for p in ordered
    )
