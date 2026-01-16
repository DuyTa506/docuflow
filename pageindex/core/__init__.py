"""
Core business logic components for PageIndex.

This module contains the refactored, class-based implementations of 
PDF and Markdown processing logic.
"""

from .toc_detector import TOCDetector
from .toc_extractor import TOCExtractor
from .toc_transformer import TOCTransformer
from .page_mapper import PageMapper
from .verifier import TOCVerifier
from .tree_builder import TreeBuilder
from .markdown_parser import MarkdownParser
from .tree_optimizer import TreeOptimizer
from .markdown_tree_builder import MarkdownTreeBuilder

__all__ = [
    'TOCDetector',
    'TOCExtractor',
    'TOCTransformer',
    'PageMapper',
    'TOCVerifier',
    'TreeBuilder',
    'MarkdownParser',
    'TreeOptimizer',
    'MarkdownTreeBuilder',
]
