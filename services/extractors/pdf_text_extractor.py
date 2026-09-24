"""
PDF Text Extractor.

Exposes classify_pages() to decide per page whether direct text extraction
is viable or OCR is required. Length is the first cut; an optional character
n-gram quality gate rejects long-but-unreadable text layers.
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

    from config.settings import settings
    from services.extractors.text_layer_quality import classify_extracted_text

    quality_gate = bool(settings.pdf_text_quality_gate)
    doc = fitz.open(pdf_path)
    result: Dict[int, str] = {}
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        result[i] = classify_extracted_text(text, min_chars=threshold, quality_gate=quality_gate)
    doc.close()
    return result
