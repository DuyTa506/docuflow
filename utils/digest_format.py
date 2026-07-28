"""Helpers for digest template formatting (bilingual keywords, chapter lines)."""

from __future__ import annotations

import re

# The digest template renders into Word paragraphs, which have no markdown.
# Anything the model emitted as markdown used to appear as literal characters.
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", flags=re.MULTILINE)
_MD_HEADING_TAIL_RE = re.compile(r"[ \t]+#{1,6}[ \t]*$", flags=re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", flags=re.DOTALL)
_MD_CODE_RE = re.compile(r"`([^`]+)`")


def plain_text(text: str | None) -> str:
    """Strip the markdown Word cannot render, leaving the words untouched.

    Deliberately conservative: headings, bold and inline code only. Single
    asterisks are left alone — in a technical digest they are far more likely
    to be multiplication than emphasis.
    """
    if not text:
        return ""
    out = _MD_HEADING_RE.sub("", str(text))
    out = _MD_HEADING_TAIL_RE.sub("", out)
    out = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", out)
    out = _MD_CODE_RE.sub(r"\1", out)
    return out.strip()


def strip_block_markdown(text: str | None) -> str:
    """Remove BLOCK markup only — `#` headings — and keep inline markup intact.

    `plain_text` throws away `**bold**` and `` `code` `` too, which was right
    while the digest could only emit flat strings. The renderer now builds Word
    runs, so inline markup should reach it and become real formatting; only the
    block-level markers, which have no run-level equivalent, are dropped here.
    """
    if not text:
        return ""
    out = _MD_HEADING_RE.sub("", str(text))
    out = _MD_HEADING_TAIL_RE.sub("", out)
    return out.strip()


def split_block_lines(text: str | None) -> list[str]:
    """Split into Word paragraphs, keeping inline markup for the renderer.

    Every line break becomes a paragraph boundary: a single `{{ abstract }}`
    placeholder collapsed the whole abstract into one unreadable block, because
    a newline inside a Word run renders as nothing at all.
    """
    return [line.strip() for line in strip_block_markdown(text).splitlines() if line.strip()]


def split_paragraphs(text: str | None) -> list[str]:
    """As above but flattened — for callers that can only take plain strings."""
    return [line.strip() for line in (plain_text(text)).splitlines() if line.strip()]


# Nhãn tiếng Việt cho từng loại đơn vị cấu trúc mà `match_chapter_heading` nhận
# ra. Một phụ lục in thành "Chương 10" là sai sự thật trong một văn bản chính
# thức, và loại đơn vị thì đã biết sẵn — không phải suy đoán gì thêm.
_UNIT_LABELS = {"chapter": "Chương", "appendix": "Phụ lục", "part": "Phần", "section": "Mục"}


def _unit_label(kind: str | None, number: int, ordinal: int | None) -> str:
    label = _UNIT_LABELS.get(kind or "", "Chương")
    if kind == "appendix" and ordinal and 1 <= ordinal <= 26:
        # Phụ lục đánh chữ; số thứ tự gốc có thể là А/Б/В (Cyrillic) — quy về
        # bảng chữ cái Latin để bản tiếng Việt đọc được.
        return f"{label} {chr(ord('A') + ordinal - 1)}"
    return f"{label} {ordinal or number}"


def chapter_heading(
    number: int,
    title_vi: str,
    title_original: str = "",
    *,
    doc_kind: str = "book",
    paper_count: int | None = None,
    heading_kind: str | None = None,
    heading_ordinal: int | None = None,
) -> str:
    """Dòng mở đầu một mục §2.2, theo đúng ba dạng của mẫu chính thức.

    - Book:            ``Chương 1. Giới thiệu (Introduction).``
    - Kỷ yếu, cụm:     ``Khoa học máy tính (Computer Science), gồm 5 BBKH.``
    - Kỷ yếu, đơn lẻ:  ``BBKH 1 - Giám sát chất lượng nước (Real-Time…).``

    ``paper_count`` < 2 rơi về dạng đơn lẻ: "gồm 1 BBKH" không phải một cụm.
    """
    vi = plain_text(title_vi)
    original = plain_text(title_original)
    # Cây trích xuất thường trả cùng một chuỗi cho cả hai trường; in nó hai
    # lần trong ngoặc đơn là nhiễu, không phải song ngữ.
    name = f"{vi} ({original})" if original and original != vi else vi

    if doc_kind == "proceedings":
        try:
            count = int(paper_count or 0)
        except (TypeError, ValueError):
            count = 0
        if count >= 2:
            return f"{name}, gồm {count} BBKH." if name else f"Cụm {number}, gồm {count} BBKH."
        return f"BBKH {number} - {name}." if name else f"BBKH {number}."

    label = _unit_label(heading_kind, number, heading_ordinal)
    return f"{label}. {name}." if name else f"{label}."


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
