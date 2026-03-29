"""
DOCX Extractor.

Uses python-docx + direct XML walking to extract structured content from .docx files.

Handles:
- Regular paragraphs and top-level tables (standard documents)
- Content inside drawing text boxes (wps:txbx / w:txbx) — common in
  layout-heavy Vietnamese CV / form documents
- Linked text box deduplication (Word links boxes to allow text to flow
  across frames; the same XML content appears in multiple anchors)
- Heading detection via:
    1. Word style name  (Heading 1 / Heading1 / Title / etc.)
    2. Font size (w:sz in half-points) relative to body size, when style = Normal
    3. Bold + larger-than-body → level 4 heading fallback
- Tables → GitHub-Flavored Markdown
- Page break detection for page_number tracking
"""
import re
from typing import List, Optional, Set

from services.extractors.base import BaseExtractor
from core.models import UnifiedElement

# ── Namespace constants ─────────────────────────────────────────────────────
_NS_W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_WPS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"

# Map Word built-in style names → heading level (1-based)
_STYLE_LEVEL_MAP = {
    "title": 1,
    "heading 1": 1,  "heading1": 1,
    "heading 2": 2,  "heading2": 2,
    "heading 3": 3,  "heading3": 3,
    "heading 4": 4,  "heading4": 4,
    "heading 5": 5,  "heading5": 5,
    "heading 6": 6,  "heading6": 6,
    "subtitle": 2,
}

# Font-size tier multipliers (sz values are in half-points; body ≈ 24 hp = 12pt)
_SZ_TIER1 = 1.6   # → level 1
_SZ_TIER2 = 1.3   # → level 2
_SZ_TIER3 = 1.15  # → level 3
_SZ_TIER4 = 1.05  # → level 4 (bold required)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_level_from_style(style_name: str) -> Optional[int]:
    """Map a Word style name to heading level 1-6, or None for body text."""
    key = style_name.strip().lower()
    if key in _STYLE_LEVEL_MAP:
        return _STYLE_LEVEL_MAP[key]
    m = re.match(r'heading\s*(\d)', key)
    if m:
        return min(int(m.group(1)), 6)
    return None


def _get_para_sz(p_el) -> Optional[float]:
    """
    Return the dominant font size for a paragraph XML element (w:p) in half-points,
    or None if not specified.  Checks rPr on runs; falls back to pPr/rPr.
    """
    # Try run-level sz first (most reliable)
    for r in p_el.findall(f"{{{_NS_W}}}r"):
        rpr = r.find(f"{{{_NS_W}}}rPr")
        if rpr is not None:
            sz = rpr.find(f"{{{_NS_W}}}sz")
            if sz is not None:
                val = sz.get(f"{{{_NS_W}}}val")
                if val:
                    return float(val)
    # Fall back to paragraph-level rPr
    pPr = p_el.find(f"{{{_NS_W}}}pPr")
    if pPr is not None:
        rPr = pPr.find(f"{{{_NS_W}}}rPr")
        if rPr is not None:
            sz = rPr.find(f"{{{_NS_W}}}sz")
            if sz is not None:
                val = sz.get(f"{{{_NS_W}}}val")
                if val:
                    return float(val)
    return None


def _is_bold(p_el) -> bool:
    """Return True if the paragraph or its first run is bold."""
    for r in p_el.findall(f"{{{_NS_W}}}r"):
        rpr = r.find(f"{{{_NS_W}}}rPr")
        if rpr is not None:
            b = rpr.find(f"{{{_NS_W}}}b")
            if b is not None:
                val = b.get(f"{{{_NS_W}}}val", "true")
                if val.lower() not in ("false", "0"):
                    return True
    return False


def _get_para_text(p_el) -> str:
    """Concatenate all w:t text within a paragraph element."""
    parts = []
    for t in p_el.findall(f".//{{{_NS_W}}}t"):
        if t.text:
            parts.append(t.text)
    return "".join(parts).strip()


def _has_page_break(p_el) -> bool:
    """Return True if the paragraph contains a w:br w:type='page'."""
    for br in p_el.findall(f".//{{{_NS_W}}}br"):
        typ = br.get(f"{{{_NS_W}}}type", "")
        if typ == "page":
            return True
    return False


def _xml_table_to_markdown(tbl_el) -> str:
    """Convert a w:tbl XML element to a GFM markdown table string."""
    rows = []
    for tr in tbl_el.findall(f"{{{_NS_W}}}tr"):
        cells = []
        for tc in tr.findall(f"{{{_NS_W}}}tc"):
            cell_texts = []
            for p in tc.findall(f".//{{{_NS_W}}}p"):
                t = _get_para_text(p)
                if t:
                    cell_texts.append(t)
            cells.append(" ".join(cell_texts).replace("|", "\\|"))
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")

    if not rows:
        return ""
    col_count = max(r.count("|") - 1 for r in rows)
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    if len(rows) > 1:
        rows.insert(1, separator)
    return "\n".join(rows)


def _analyze_body_sz(body_el) -> float:
    """
    Scan all w:sz values in the document body to find the modal (body) font size.
    Returns half-points (e.g. 24 = 12pt).  Falls back to 24 if nothing found.
    """
    from statistics import mode, StatisticsError
    sizes = []
    for sz in body_el.findall(f".//{{{_NS_W}}}sz"):
        val = sz.get(f"{{{_NS_W}}}val")
        if val and val.isdigit():
            sizes.append(int(val))
    if not sizes:
        return 24.0
    try:
        return float(mode(sizes))
    except StatisticsError:
        sizes.sort()
        return float(sizes[len(sizes) // 2])


def _detect_level_by_size(sz: float, body_sz: float, bold: bool) -> Optional[int]:
    """Classify a font size relative to body size into heading level 1-4, or None."""
    ratio = sz / body_sz if body_sz > 0 else 1.0
    if ratio > _SZ_TIER1:
        return 1
    if ratio > _SZ_TIER2:
        return 2
    if ratio > _SZ_TIER3:
        return 3
    if bold and ratio > _SZ_TIER4:
        return 4
    return None


# ── Main extractor ───────────────────────────────────────────────────────────

class DocxExtractor(BaseExtractor):
    """
    Extract structured content from .docx files.

    Produces UnifiedElements with heading levels from Word styles or font sizes,
    tables as GFM Markdown, and page numbers from explicit page-break runs.
    Handles both standard paragraph documents and layout-heavy text-box documents.
    """

    def extract(self, file_path: str) -> List[UnifiedElement]:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "python-docx is required. Install with: pip install python-docx"
            )

        doc = DocxDocument(file_path)
        body = doc.element.body

        # Pass 1: determine body font size for size-based heading detection
        body_sz = _analyze_body_sz(body)

        elements: List[UnifiedElement] = []
        page_number = 1
        order = 0

        # Collect all content regions to walk:
        #   1. Top-level body
        #   2. Each txbxContent (drawing text boxes)
        # Deduplicate txbxContent by their XML id to handle linked text boxes.
        regions = [body]
        seen_txbx_ids: Set[int] = set()

        for txbx in body.findall(
            f".//{{{_NS_WPS}}}txbx"
        ) + body.findall(
            f".//{{{_NS_W}}}txbx"
        ):
            content = txbx.find(f"{{{_NS_W}}}txbxContent")
            if content is None:
                # txbxContent may be directly the txbx's child under a different ns
                for child in txbx:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "txbxContent":
                        content = child
                        break
            if content is not None and id(content) not in seen_txbx_ids:
                seen_txbx_ids.add(id(content))
                regions.append(content)

        # Walk each region
        for region in regions:
            for child in region:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

                if tag == "p":
                    if _has_page_break(child):
                        page_number += 1

                    text = _get_para_text(child)
                    if not text:
                        continue

                    # Heading detection: style first, then size/bold
                    pPr = child.find(f"{{{_NS_W}}}pPr")
                    style_el = (
                        pPr.find(f"{{{_NS_W}}}pStyle") if pPr is not None else None
                    )
                    style_val = (
                        style_el.get(f"{{{_NS_W}}}val", "Normal")
                        if style_el is not None else "Normal"
                    )
                    level = _get_level_from_style(style_val)

                    if level is None:
                        sz = _get_para_sz(child)
                        if sz is not None:
                            bold = _is_bold(child)
                            level = _detect_level_by_size(sz, body_sz, bold)

                    elements.append(UnifiedElement(
                        element_type="heading" if level is not None else "text",
                        text=text,
                        page_number=page_number,
                        order=order,
                        source="docx",
                        level=level,
                        bbox=None,
                        font_size=None,
                        style_name=style_val,
                    ))
                    order += 1

                elif tag == "tbl":
                    md = _xml_table_to_markdown(child)
                    if md.strip():
                        elements.append(UnifiedElement(
                            element_type="table",
                            text=md,
                            page_number=page_number,
                            order=order,
                            source="docx",
                            level=None,
                            bbox=None,
                            font_size=None,
                            style_name=None,
                        ))
                        order += 1

        return elements
