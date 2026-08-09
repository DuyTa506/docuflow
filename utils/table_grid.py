"""Shared table grid parsing for DOCX and layout PDF export."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")


def span_int(value, default: int = 1) -> int:
    """Parse a colspan/rowspan attribute, clamped to a sane minimum of 1."""
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


class HtmlTableParser(HTMLParser):
    """Parse an HTML table into rows of cell dicts with colspan/rowspan."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[dict]] = []
        self.table_style: str = ""
        self._current_row: list[dict] | None = None
        self._cell_parts: list[str] = []
        self._cell: dict | None = None

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag == "table" and not self.table_style:
            self.table_style = attr_map.get("style", "")
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._cell_parts = []
            self._cell = {
                "colspan": span_int(attr_map.get("colspan")),
                "rowspan": span_int(attr_map.get("rowspan")),
                "header": tag == "th",
            }

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in ("td", "th") and self._current_row is not None and self._cell is not None:
            self._cell["text"] = "".join(self._cell_parts).strip()
            self._current_row.append(self._cell)
            self._cell = None
            self._cell_parts = []
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str):
        if self._cell is not None:
            self._cell_parts.append(data)


def build_table_grid(rows: list[list[dict]]) -> tuple[int, int, list[tuple]]:
    """Return (n_rows, n_cols, placements) for grid-aware table layout."""
    n_rows = len(rows)
    if n_rows == 0:
        return 0, 0, []
    n_cols = max((sum(span_int(c.get("colspan")) for c in r) for r in rows), default=0)
    if n_cols == 0:
        return 0, 0, []

    occ: list[list[bool]] = [[False] * n_cols for _ in range(n_rows)]
    placements: list[tuple] = []
    for ri, row in enumerate(rows):
        ci = 0
        for cell in row:
            while ci < n_cols and occ[ri][ci]:
                ci += 1
            if ci >= n_cols:
                n_cols += 1
                for rr in occ:
                    rr.append(False)
            cs = min(span_int(cell.get("colspan")), n_cols - ci)
            rs = min(span_int(cell.get("rowspan")), n_rows - ri)
            for dr in range(rs):
                for dc in range(cs):
                    occ[ri + dr][ci + dc] = True
            placements.append(
                (ri, ci, ri + rs - 1, ci + cs - 1, cell.get("text", ""), bool(cell.get("header")))
            )
            ci += cs
    return n_rows, n_cols, placements


def compact_empty_columns(n_cols: int, placements: list[tuple]) -> tuple[int, list[tuple]]:
    """Drop columns with no non-empty text anywhere."""
    if n_cols == 0:
        return n_cols, placements
    used = [False] * n_cols
    for _r0, c0, _r1, c1, text, _h in placements:
        if (text or "").strip():
            for c in range(c0, min(c1 + 1, n_cols)):
                used[c] = True
    if all(used):
        return n_cols, placements
    new_index: dict[int, int] = {}
    k = 0
    for c in range(n_cols):
        if used[c]:
            new_index[c] = k
            k += 1
    if k == 0:
        return n_cols, placements
    new_placements: list[tuple] = []
    for r0, c0, r1, c1, text, header in placements:
        cols = [c for c in range(c0, c1 + 1) if c in new_index]
        if not cols:
            continue
        new_placements.append((r0, new_index[cols[0]], r1, new_index[cols[-1]], text, header))
    return k, new_placements


def parse_html_table(html: str) -> tuple[list[list[dict]], str]:
    parser = HtmlTableParser()
    parser.feed(html)
    parser.close()
    return parser.rows, parser.table_style


def parse_markdown_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        if _TABLE_SEP_RE.match(line):
            continue
        inner = line.strip().strip("|")
        cells = [c.strip() for c in inner.split("|")]
        rows.append(cells)
    return rows


def string_rows_to_cell_rows(rows: list[list[str]]) -> list[list[dict]]:
    return [
        [{"text": c, "colspan": 1, "rowspan": 1, "header": ri == 0} for c in row]
        for ri, row in enumerate(rows)
    ]


def table_text_to_cell_rows(text: str) -> list[list[dict]] | None:
    """Parse table element text (HTML or markdown pipes) into cell rows."""
    if not text or not text.strip():
        return None
    html_match = _HTML_TABLE_RE.search(text)
    if html_match:
        rows, _style = parse_html_table(html_match.group(0))
        return rows if rows else None
    if "|" in text:
        lines = [ln for ln in text.splitlines() if _TABLE_ROW_RE.match(ln.strip())]
        if lines:
            str_rows = parse_markdown_table_rows(lines)
            return string_rows_to_cell_rows(str_rows) if str_rows else None
    return None
