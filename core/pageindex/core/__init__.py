"""
Core business logic components for PageIndex.

This module contains the refactored, class-based implementations of
Markdown processing logic.

NOTE: PDF-specific components (TOCDetector, TOCExtractor, TOCTransformer,
PageMapper, TOCVerifier) have been moved to legacy/core.
"""

from .markdown_parser import MarkdownParser
from .markdown_tree_builder import MarkdownTreeBuilder
from .tree_optimizer import TreeOptimizer

__all__ = [
    "MarkdownParser",
    "TreeOptimizer",
    "MarkdownTreeBuilder",
]
