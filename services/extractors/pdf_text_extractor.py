"""
PDF Text Extractor.

Uses PyMuPDF (fitz) for heading detection via font-size analysis and
pdfplumber for precise table extraction from text-based PDF pages.

Two-pass approach:
  Pass 1 — analyze_fonts() : Collect all font sizes → body_font_size (mode).
  Pass 2 — extract_page()  : Per-span heading detection + pdfplumber table overlay.

Also exposes classify_pages() to decide per page whether direct text
extraction is viable or OCR is required.
"""
from statistics import mode, StatisticsError
from typing import List, Dict, Optional, Set, Tuple

from services.extractors.base import BaseExtractor
from core.models import UnifiedElement

# Font-size multipliers for heading tier detection
_TIER_1_MULT = 1.6    # → level 1
_TIER_2_MULT = 1.3    # → level 2
_TIER_3_MULT = 1.15   # → level 3
_TIER_4_MULT = 1.05   # → level 4  (bold + slightly bigger than body)

# Text-presence threshold (chars): pages with fewer chars are considered scanned
DEFAULT_TEXT_THRESHOLD = 50


def classify_pages(pdf_path: str, threshold: int = DEFAULT_TEXT_THRESHOLD) -> Dict[int, str]:
    """
    Classify each page of a PDF as "text" or "scanned".

    Args:
        pdf_path: Path to the PDF file.
        threshold: Minimum character count to consider a page as text-based.

    Returns:
        Dict mapping 1-based page_number → "text" | "scanned"
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF text extraction. "
            "Install with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)
    result: Dict[int, str] = {}
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        result[i] = "text" if len(text) >= threshold else "scanned"
    doc.close()
    return result


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """
    Convert a pdfplumber table (list of rows, each a list of cell strings)
    into a GitHub-Flavored Markdown table string.
    """
    if not table:
        return ""

    # Normalise cells: replace None with empty string, collapse newlines
    rows = []
    for row in table:
        cells = [
            (cell or "").replace("\n", " ").strip()
            for cell in row
        ]
        rows.append(cells)

    if not rows:
        return ""

    # Build markdown
    col_count = max(len(r) for r in rows)
    lines = []
    for i, row in enumerate(rows):
        # Pad short rows
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines)


def _bbox_overlaps(bbox_a: Tuple, bbox_b: Tuple, threshold: float = 0.5) -> bool:
    """
    Return True when bbox_a overlaps bbox_b by more than `threshold` of bbox_a's area.
    Bboxes are (x0, y0, x1, y1) tuples.
    """
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
    return (intersection / area_a) >= threshold


class PdfTextExtractor(BaseExtractor):
    """
    Extract text and headings from PDF pages that have embedded text.

    Strategy:
    1. PyMuPDF span-based pass: heading detection via font-size tiers.
    2. pdfplumber table pass: extract tables as Markdown; suppress the
       raw text spans that fall inside table bounding boxes.
    """

    def __init__(self, pdf_path: str):
        try:
            import fitz  # noqa
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF text extraction. "
                "Install with: pip install pymupdf"
            )
        self.pdf_path = pdf_path
        self._body_font_size: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────────

    def extract(self, file_path: str) -> List[UnifiedElement]:
        """Implements BaseExtractor.extract(); file_path is ignored (uses self.pdf_path)."""
        return self.extract_all_pages()

    def extract_all_pages(self, page_numbers: Optional[List[int]] = None) -> List[UnifiedElement]:
        """Extract text from the given 1-based page numbers (all pages if None)."""
        import fitz
        doc = fitz.open(self.pdf_path)
        if self._body_font_size is None:
            self._body_font_size = self._analyze_fonts(doc)

        pages_to_process = page_numbers or list(range(1, doc.page_count + 1))
        all_elements: List[UnifiedElement] = []
        for pnum in pages_to_process:
            page = doc[pnum - 1]
            elements = self._extract_page(page, pnum)
            all_elements.extend(elements)
        doc.close()
        return all_elements

    def extract_page(self, page_number: int) -> List[UnifiedElement]:
        """Extract a single 1-based page."""
        import fitz
        doc = fitz.open(self.pdf_path)
        if self._body_font_size is None:
            self._body_font_size = self._analyze_fonts(doc)
        page = doc[page_number - 1]
        elements = self._extract_page(page, page_number)
        doc.close()
        return elements

    # ── Internal helpers ────────────────────────────────────────────

    def _analyze_fonts(self, doc) -> float:
        """
        Pass 1: collect all span font sizes across the document and return
        the mode (= body font size).  Falls back to 12.0 if no spans found.
        """
        sizes: List[int] = []
        for page in doc:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = span.get("size", 0)
                        if sz > 0:
                            sizes.append(round(sz * 2) / 2)

        if not sizes:
            return 12.0
        try:
            return float(mode(sizes))
        except StatisticsError:
            sizes.sort()
            return float(sizes[len(sizes) // 2])

    def _extract_tables_pdfplumber(self, page_number: int) -> Tuple[List[UnifiedElement], List[Tuple]]:
        """
        Use pdfplumber to extract tables from a page.

        Returns:
            (table_elements, table_bboxes)
            table_bboxes are (x0, y0, x1, y1) tuples in PDF point coords,
            used to suppress duplicate span text from PyMuPDF.
        """
        table_elements: List[UnifiedElement] = []
        table_bboxes: List[Tuple] = []

        try:
            import pdfplumber
        except ImportError:
            # pdfplumber not installed — skip table extraction gracefully
            return table_elements, table_bboxes

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_number > len(pdf.pages):
                    return table_elements, table_bboxes

                plumber_page = pdf.pages[page_number - 1]
                tables = plumber_page.extract_tables()
                table_objects = plumber_page.find_tables()

                for idx, (tbl_data, tbl_obj) in enumerate(zip(tables, table_objects)):
                    md = _table_to_markdown(tbl_data)
                    if not md.strip():
                        continue

                    # pdfplumber bbox: (x0, top, x1, bottom) in page coords
                    bbox = tbl_obj.bbox  # (x0, top, x1, bottom)
                    x0, top, x1, bottom = bbox
                    bbox_dict = {'x1': x0, 'y1': top, 'x2': x1, 'y2': bottom}

                    table_elements.append(UnifiedElement(
                        element_type="table",
                        text=md,
                        page_number=page_number,
                        order=idx,           # re-ordered later
                        source="pdf_text",
                        level=None,
                        bbox=bbox_dict,
                        font_size=None,
                        style_name=None,
                    ))
                    table_bboxes.append((x0, top, x1, bottom))

        except Exception:
            # Never let table extraction break the main text pass
            pass

        return table_elements, table_bboxes

    def _extract_page(self, page, page_number: int) -> List[UnifiedElement]:
        """
        Pass 2: extract all content from a single PyMuPDF page.

        - Tables are extracted first via pdfplumber (Markdown format).
        - PyMuPDF spans that overlap table bounding boxes are suppressed
          to avoid duplicate content.
        - Remaining spans are classified as heading or body text by font size.
        - All elements are merged into a single reading-order list.
        """
        body_size = self._body_font_size or 12.0

        # ── Step A: extract tables ───────────────────────────────────
        table_elements, table_bboxes = self._extract_tables_pdfplumber(page_number)

        # ── Step B: extract text spans (PyMuPDF) ────────────────────
        text_elements: List[UnifiedElement] = []
        order = 0

        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                line_texts = []
                line_level: Optional[int] = None
                line_bbox: Optional[tuple] = None
                line_font_size: Optional[float] = None

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    size = float(span.get("size", body_size))
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)
                    span_bbox = span.get("bbox")  # (x0, y0, x1, y1)

                    # Suppress spans that sit inside a table region
                    if span_bbox and table_bboxes:
                        if any(_bbox_overlaps(span_bbox, tb) for tb in table_bboxes):
                            continue

                    level = self._detect_level(size, body_size, is_bold)
                    line_texts.append(text)
                    if line_level is None:
                        line_level = level
                    if line_bbox is None and span_bbox:
                        line_bbox = span_bbox
                    if line_font_size is None:
                        line_font_size = size

                full_text = " ".join(line_texts).strip()
                if not full_text:
                    continue

                bbox_dict: Optional[Dict] = None
                if line_bbox:
                    x0, y0, x1, y1 = line_bbox
                    bbox_dict = {'x1': x0, 'y1': y0, 'x2': x1, 'y2': y1}

                text_elements.append(UnifiedElement(
                    element_type="heading" if line_level is not None else "text",
                    text=full_text,
                    page_number=page_number,
                    order=order,
                    source="pdf_text",
                    level=line_level,
                    bbox=bbox_dict,
                    font_size=line_font_size,
                    style_name=None,
                ))
                order += 1

        # ── Step C: merge & sort by vertical position ────────────────
        # Tables get their y1 from bbox; text elements already have y1.
        def _y1(elem: UnifiedElement) -> float:
            return elem.bbox['y1'] if elem.bbox else float('inf')

        all_elements = text_elements + table_elements
        all_elements.sort(key=_y1)

        # Re-assign reading order after sort
        for i, elem in enumerate(all_elements):
            elem.order = i

        return all_elements

    @staticmethod
    def _detect_level(size: float, body_size: float, is_bold: bool) -> Optional[int]:
        """
        Classify a font size relative to body_size into a heading level (1-4)
        or None for body text.
        """
        ratio = size / body_size if body_size > 0 else 1.0
        if ratio > _TIER_1_MULT:
            return 1
        if ratio > _TIER_2_MULT:
            return 2
        if ratio > _TIER_3_MULT:
            return 3
        if is_bold and ratio > _TIER_4_MULT:
            return 4
        return None
