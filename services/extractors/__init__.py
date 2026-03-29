"""
Unified extraction layer.

Every extractor (DOCX, PDF-text, OCR) produces List[UnifiedElement]
which is then converted to layout_element dicts for build_spatial_tree().
"""
from services.extractors.base import BaseExtractor
from services.extractors.docx_extractor import DocxExtractor
from services.extractors.pdf_text_extractor import PdfTextExtractor, classify_pages
from services.extractors.ocr_extractor import OcrExtractor
from services.extractors.doc_converter import convert_doc_to_docx

__all__ = [
    "BaseExtractor",
    "DocxExtractor",
    "PdfTextExtractor",
    "classify_pages",
    "OcrExtractor",
    "convert_doc_to_docx",
]
