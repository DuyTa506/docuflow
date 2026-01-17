"""
Core business logic components for PageIndex.

This module contains the refactored, class-based implementations of 
Markdown processing logic.

NOTE: PDF-specific components (TOCDetector, TOCExtractor, TOCTransformer, 
PageMapper, TOCVerifier) have been moved to legacy/core.
"""

from .tree_builder import TreeBuilder
from .markdown_parser import MarkdownParser
from .tree_optimizer import TreeOptimizer
from .markdown_tree_builder import MarkdownTreeBuilder

__all__ = [
    'TreeBuilder',  # Base tree builder, still used
    'MarkdownParser',
    'TreeOptimizer',
    'MarkdownTreeBuilder',
]
