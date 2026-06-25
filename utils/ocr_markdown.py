"""
Normalize OCR markdown/HTML before DOCX export.

DeepSeek-OCR outputs markdown mixed with HTML (tables, center tags, etc.).
This module cleans common artifacts while preserving structure.
"""

import re

from core.constants import OCR_EQUATION_LABELS

_PAGE_BREAK_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)
_EQUATION_LABELS = OCR_EQUATION_LABELS
_LATEX_DISPLAY_RE = re.compile(r"\$\$[\s\S]+?\$\$")
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")


def normalize_ocr_markdown(text: str) -> str:
    """Prepare raw OCR markdown for structured DOCX rendering."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_residual_grounding_tags(text)
    text = _normalize_html_tables(text)
    text = _normalize_center_tags(text)
    text = _collapse_excessive_blank_lines(text)
    return text.strip()


def is_structured_markdown(text: str) -> bool:
    """Return True when content should be rendered as markdown, not plain lines."""
    if not text:
        return False
    if re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
        return True
    if re.search(r"^\s*\|.+?\|\s*$", text, re.MULTILINE):
        return True
    if re.search(r"^\s*[-*+]\s+\S", text, re.MULTILINE):
        return True
    if re.search(r"^\s*\d+\.\s+\S", text, re.MULTILINE):
        return True
    if "<table" in text.lower():
        return True
    if re.search(r"</?(center|td|tr|th)\b", text, re.IGNORECASE):
        return True
    if _PAGE_BREAK_RE.search(text):
        return True
    if _LATEX_DISPLAY_RE.search(text) or _LATEX_INLINE_RE.search(text):
        return True
    return False


def split_pages(text: str) -> list[str]:
    """Split aggregated OCR text on page separators (---)."""
    parts = re.split(r"\n\s*---\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def element_heading_level(label: str) -> int | None:
    """Map OCR layout label to a Word heading level."""
    key = (label or "").lower()
    if key == "title":
        return 1
    if key == "sub_title":
        return 2
    if key == "heading":
        return 3
    return None


def element_export_text(label: str, text_content: str | None) -> str:
    """Pick the best text blob for an element during structured export."""
    text = (text_content or "").strip()
    if not text:
        return ""
    key = (label or "").lower()
    if key in _EQUATION_LABELS:
        from utils.math_omml import wrap_as_equation_markdown

        return wrap_as_equation_markdown(text)
    level = element_heading_level(label)
    if level is not None:
        prefix = "#" * level
        if not text.lstrip().startswith("#"):
            return f"{prefix} {text}"
    return text


def _strip_residual_grounding_tags(text: str) -> str:
    text = re.sub(
        r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<\|grounding\|>", "", text)
    text = re.sub(r"<\|/grounding\|>", "", text)
    return text


def _normalize_html_tables(text: str) -> str:
    """Ensure table rows are on separate lines for the HTML parser."""
    text = re.sub(r"(?i)</tr>\s*<tr>", "</tr>\n<tr>", text)
    text = re.sub(r"(?i)<table\b", "\n<table", text)
    text = re.sub(r"(?i)</table>", "</table>\n", text)
    return text


def _normalize_center_tags(text: str) -> str:
    """Convert <center> blocks to markdown-style centered lines."""
    def _repl(match: re.Match) -> str:
        inner = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return f"\n\n{center_line(inner)}\n\n"

    return re.sub(r"(?is)<center>(.*?)</center>", _repl, text)


def center_line(text: str) -> str:
    return text


def _collapse_excessive_blank_lines(text: str) -> str:
    return re.sub(r"\n{4,}", "\n\n\n", text)
