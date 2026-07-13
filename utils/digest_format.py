"""Helpers for digest template formatting (bilingual keywords, chapter lines)."""

from __future__ import annotations

from config.settings import normalize_lang_code


def format_keyword_bilingual(source_lang: str, keyword: str, display: str | None = None) -> str:
    """Format keyword for digest §2.3."""
    if display and display.strip():
        return display.strip()
    kw = (keyword or "").strip()
    if not kw:
        return ""
    lang = normalize_lang_code(source_lang or "en")
    if lang == "vi":
        return kw
    return kw


def join_catalog_items(items: list[str]) -> str:
    """Join CTĐT / NNC list for template field."""
    if not items:
        return ""
    return "; ".join(str(x).strip() for x in items if str(x).strip())


def is_chapter_schema(details: dict | None) -> bool:
    if not details or not isinstance(details, dict):
        return False
    chapters = details.get("chapters")
    return isinstance(chapters, list) and len(chapters) > 0


def bibliographic_defaults(title: str = "", pages: str | int | None = None) -> dict:
    return {
        "title_display": title or "",
        "authors": "",
        "publisher": "",
        "year": "",
        "isbn": "",
        "doi": "",
        "pages": str(pages) if pages is not None else "",
    }


def usage_scope_defaults() -> dict:
    return {
        "undergraduate": [],
        "master": [],
        "phd": [],
        "strong_research_groups": [],
    }
