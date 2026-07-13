"""
PDF Text Extractor.

Exposes classify_pages() to decide per page whether direct text extraction
is viable or OCR is required.
"""

from typing import Dict

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
            "PyMuPDF is required for PDF text extraction. " "Install with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)
    result: Dict[int, str] = {}
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        result[i] = "text" if len(text) >= threshold else "scanned"
    doc.close()
    return result
