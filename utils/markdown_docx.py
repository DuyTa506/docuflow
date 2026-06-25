"""
Convert OCR markdown (with embedded HTML tables) into a structured python-docx document.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Iterable

from docx import Document as DocxDocument
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

from core.constants import OCR_EQUATION_LABELS
from utils.ocr_markdown import normalize_ocr_markdown, split_pages

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_OL_RE = re.compile(r"^(\s*)\d+\.\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_CODE_FENCE_RE = re.compile(r"^```")


class _HtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._cell_parts = []

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in ("td", "th") and self._current_row is not None:
            self._current_row.append("".join(self._cell_parts).strip())
            self._cell_parts = []
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str):
        if self._current_row is not None:
            self._cell_parts.append(data)


def render_markdown_to_docx(
    doc: DocxDocument,
    markdown: str,
    *,
    page_breaks: bool = True,
) -> None:
    """Append structured content from OCR markdown into an existing document."""
    markdown = normalize_ocr_markdown(markdown)
    if not markdown:
        return

    pages = split_pages(markdown) if page_breaks and "\n---\n" in f"\n{markdown}\n" else [markdown]
    for idx, page_text in enumerate(pages):
        if idx > 0 and page_breaks:
            doc.add_page_break()
        _render_block(doc, page_text)


def render_layout_elements_to_docx(
    doc: DocxDocument,
    elements: Iterable,
    *,
    page_break_between_pages: bool = True,
) -> None:
    """Render document from ordered layout elements (spatial reading order)."""
    from utils.ocr_markdown import element_export_text, element_heading_level

    views = list(elements)
    page_metrics = _build_page_metrics(views)

    current_page: int | None = None
    for elem in views:
        page_num = getattr(elem, "page_number", None)
        page_rel = getattr(elem, "page", None)
        if page_num is None and page_rel is not None:
            page_num = getattr(page_rel, "page_number", None)

        if (
            page_break_between_pages
            and page_num is not None
            and current_page is not None
            and page_num != current_page
        ):
            doc.add_page_break()
        if page_num is not None:
            current_page = page_num

        label = getattr(elem, "label", "text") or "text"
        text = element_export_text(label, getattr(elem, "text_content", None))
        if not text:
            continue

        level = element_heading_level(label)
        if level is not None:
            heading = doc.add_heading(_strip_inline_markdown(text), level=level)
            _apply_spatial_alignment(heading, elem, page_metrics.get(page_num))
            continue

        if label.lower() == "table" or "<table" in text.lower():
            _render_tables_and_text(doc, text)
            continue

        if label.lower() in OCR_EQUATION_LABELS:
            _add_equation_paragraph(doc, text, elem, page_metrics.get(page_num))
            continue

        p = doc.add_paragraph()
        _add_multiline_runs(p, text)
        _apply_spatial_alignment(p, elem, page_metrics.get(page_num))


def build_docx_bytes_from_markdown(
    markdown: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
) -> bytes:
    """Build a complete .docx byte stream from markdown content."""
    import io

    doc = DocxDocument()
    if title:
        doc.add_heading(title, level=1)
    for h in headings or []:
        doc.add_heading(h, level=2)
    render_markdown_to_docx(doc, markdown)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _render_block(doc: DocxDocument, text: str) -> None:
    remaining = text.strip()
    if not remaining:
        return

    while remaining:
        html_match = _HTML_TABLE_RE.search(remaining)
        if html_match and html_match.start() == 0:
            table_html = html_match.group(0)
            _add_html_table(doc, table_html)
            remaining = remaining[html_match.end() :].lstrip()
            continue

        if html_match and html_match.start() > 0:
            prefix = remaining[: html_match.start()].rstrip()
            if prefix:
                _render_text_lines(doc, prefix)
            remaining = remaining[html_match.start() :].lstrip()
            continue

        _render_text_lines(doc, remaining)
        break


def _render_tables_and_text(doc: DocxDocument, text: str) -> None:
    _render_block(doc, text)


def _render_text_lines(doc: DocxDocument, text: str) -> None:
    lines = text.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if _CODE_FENCE_RE.match(stripped):
            if in_code:
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Courier New"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
                run.font.size = Pt(10)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if _is_markdown_table_start(lines, i):
            table_lines, i = _collect_markdown_table(lines, i)
            _add_markdown_table(doc, table_lines)
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = min(len(heading.group(1)), 9)
            doc.add_heading(_strip_inline_markdown(heading.group(2)), level=level)
            i += 1
            continue

        ul = _UL_RE.match(line)
        if ul:
            p = doc.add_paragraph(style="List Bullet")
            _add_inline_runs(p, ul.group(2).strip())
            i += 1
            continue

        ol = _OL_RE.match(line)
        if ol:
            p = doc.add_paragraph(style="List Number")
            _add_inline_runs(p, ol.group(2).strip())
            i += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or _HEADING_RE.match(nxt)
                or _UL_RE.match(lines[i])
                or _OL_RE.match(lines[i])
                or _is_markdown_table_start(lines, i)
                or _HTML_TABLE_RE.match(nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1

        if _should_center_line_block(para_lines):
            for line in para_lines:
                p = doc.add_paragraph()
                _add_inline_runs(p, line)
                p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            continue

        p = doc.add_paragraph()
        _add_inline_runs(p, " ".join(para_lines))


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    row = lines[index].strip()
    sep = lines[index + 1].strip()
    return bool(_TABLE_ROW_RE.match(row) and _TABLE_SEP_RE.match(sep))


def _collect_markdown_table(lines: list[str], start: int) -> tuple[list[str], int]:
    collected = []
    i = start
    while i < len(lines):
        row = lines[i].strip()
        if not row or not _TABLE_ROW_RE.match(row):
            break
        collected.append(row)
        i += 1
    return collected, i


def _parse_markdown_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        if _TABLE_SEP_RE.match(line):
            continue
        inner = line.strip().strip("|")
        cells = [c.strip() for c in inner.split("|")]
        rows.append(cells)
    return rows


def _add_markdown_table(doc: DocxDocument, table_lines: list[str]) -> None:
    rows = _parse_markdown_table_rows(table_lines)
    _add_rows_to_docx_table(doc, rows)


def _add_html_table(doc: DocxDocument, html: str) -> None:
    parser = _HtmlTableParser()
    parser.feed(html)
    parser.close()
    if parser.rows:
        _add_rows_to_docx_table(doc, parser.rows)


def _add_rows_to_docx_table(doc: DocxDocument, rows: list[list[str]]) -> None:
    if not rows:
        return
    col_count = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=col_count)
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx in range(col_count):
            cell_text = row[c_idx] if c_idx < len(row) else ""
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            _add_inline_runs(p, _strip_html_inline(cell_text))


def _add_inline_runs(paragraph: Paragraph, text: str) -> None:
    text = _strip_html_inline(text)
    if not text:
        return

    pattern = re.compile(
        r"(\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_|`[^`]+`)"
    )
    parts = pattern.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("__") and part.endswith("__"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("_") and part.endswith("_"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Courier New"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
        else:
            paragraph.add_run(part)


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text.strip())
    return _strip_html_inline(text)


def _strip_html_inline(text: str) -> str:
    text = re.sub(r"(?i)</?center>", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _should_center_line_block(lines: list[str]) -> bool:
    """Heuristic: title-page style blocks — several short lines, no sentence end."""
    if len(lines) < 2:
        return False
    if any(len(line) > 90 for line in lines):
        return False
    if any(line.endswith((".", "?", "!")) and len(line) > 40 for line in lines):
        return False
    return True


def _add_multiline_runs(paragraph: Paragraph, text: str) -> None:
    """Preserve intentional line breaks inside a spatial text block."""
    text = _strip_html_inline(text)
    if not text:
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    if len(lines) == 1:
        _add_inline_runs(paragraph, lines[0])
        return
    for idx, line in enumerate(lines):
        if idx:
            paragraph.add_run().add_break()
        _add_inline_runs(paragraph, line)


class _PageMetrics:
    __slots__ = ("width", "min_x", "max_x")

    def __init__(self, width: float, min_x: float, max_x: float):
        self.width = width
        self.min_x = min_x
        self.max_x = max_x


def _build_page_metrics(elements: Iterable) -> dict[int, _PageMetrics]:
    buckets: dict[int, list[tuple[float, float]]] = {}
    for elem in elements:
        page_num = getattr(elem, "page_number", None)
        x1 = getattr(elem, "bbox_x1", None)
        x2 = getattr(elem, "bbox_x2", None)
        if page_num is None or x1 is None or x2 is None:
            continue
        buckets.setdefault(page_num, []).append((float(x1), float(x2)))

    metrics: dict[int, _PageMetrics] = {}
    for page_num, boxes in buckets.items():
        min_x = min(b[0] for b in boxes)
        max_x = max(b[1] for b in boxes)
        width = max(max_x - min_x, 1.0)
        metrics[page_num] = _PageMetrics(width=width, min_x=min_x, max_x=max_x)
    return metrics


def _add_equation_paragraph(doc: DocxDocument, text: str, elem, metrics: _PageMetrics | None) -> None:
    """Render equation as OMML when pandoc is available, else italic fallback."""
    from utils.math_omml import latex_to_omml_fragment, omml_fragment_for_docx, wrap_as_equation_markdown

    latex = wrap_as_equation_markdown(text).strip()
    inner = latex.strip("$").strip()
    omml_bytes = latex_to_omml_fragment(inner, display=True) if inner else None
    p = doc.add_paragraph()
    if omml_bytes:
        try:
            p._element.append(parse_xml(omml_fragment_for_docx(omml_bytes)))
        except Exception:
            _add_inline_runs(p, inner or text)
            if p.runs:
                p.runs[0].italic = True
    else:
        _add_inline_runs(p, inner or text)
        if p.runs:
            p.runs[0].italic = True
    _apply_spatial_alignment(p, elem, metrics)


def _apply_spatial_alignment(
    paragraph: Paragraph,
    elem,
    metrics: _PageMetrics | None,
) -> None:
    if metrics is None:
        return
    x1 = getattr(elem, "bbox_x1", None)
    x2 = getattr(elem, "bbox_x2", None)
    if x1 is None or x2 is None:
        return

    center_x = (float(x1) + float(x2)) / 2.0
    page_mid = metrics.min_x + metrics.width / 2.0
    rel_center = abs(center_x - page_mid) / metrics.width

    if rel_center < 0.08:
        paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        return

    rel_right = (float(x2) - metrics.min_x) / metrics.width
    if rel_right > 0.88 and (float(x2) - float(x1)) / metrics.width < 0.35:
        paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
        return

    indent_ratio = max(0.0, (float(x1) - metrics.min_x) / metrics.width)
    if indent_ratio > 0.12:
        paragraph.paragraph_format.left_indent = Inches(min(indent_ratio * 6.5, 2.0))
