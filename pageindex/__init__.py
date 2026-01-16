"""
PageIndex - Document Structure Extraction Library.

Provides tools for extracting hierarchical structure from PDF and Markdown documents.
"""

from .entry_points import page_index_main, md_to_tree, page_index, config
from .utils import ConfigLoader

__all__ = [
    'page_index_main',
    'md_to_tree', 
    'page_index',
    'config',
    'ConfigLoader',
]