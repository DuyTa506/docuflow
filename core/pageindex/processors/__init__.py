"""
High-level orchestrators for document processing.

NOTE: PDFProcessor has been moved to legacy folder.
Use MarkdownProcessor with OCR-generated .md files instead.
"""

from .md_processor import MarkdownProcessor

__all__ = [
    'MarkdownProcessor',
]
