"""
Unified extraction layer.

Every extractor (DOCX, PDF layout via Docling, OCR) produces
List[UnifiedElement] which is then converted to layout_element dicts
for build_spatial_tree().
"""

from services.extractors.base import BaseExtractor
from services.extractors.docling_layout_extractor import DoclingLayoutExtractor
from services.extractors.docling_pdf_extractor import DoclingPdfExtractor, classify_pages
from services.extractors.docx_extractor import DocxExtractor
from services.extractors.ocr_extractor import OcrExtractor

__all__ = [
    "BaseExtractor",
    "DocxExtractor",
    "DoclingPdfExtractor",
    "DoclingLayoutExtractor",
    "classify_pages",
    "OcrExtractor",
    "convert_doc_to_docx",
]
