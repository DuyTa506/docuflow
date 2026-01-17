"""
DEPRECATED: Legacy PDF-specific core components.

These components are only used by the deprecated PDFProcessor.
DO NOT USE in new code.

Components:
- TOCDetector: Detects table of contents in PDF
- TOCExtractor: Extracts TOC content
- TOCTransformer: Transforms TOC to JSON
- PageMapper: Maps logical to physical page numbers
- TOCVerifier: Verifies TOC accuracy
"""

from .toc_detector import TOCDetector
from .toc_extractor import TOCExtractor
from .toc_transformer import TOCTransformer
from .page_mapper import PageMapper
from .verifier import TOCVerifier

__all__ = [
    'TOCDetector',
    'TOCExtractor',
    'TOCTransformer',
    'PageMapper',
    'TOCVerifier',
]
