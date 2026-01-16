"""Data access layer - Database models and connections."""

from .db_models import Base, Document, Page, LayoutElement, TreeIndex, TreeNode
from .database import (
    DatabaseManager,
    get_db_manager,
    session_scope,
    init_database,
    get_db
)

__all__ = [
    # Models
    'Base',
    'Document',
    'Page',
    'LayoutElement',
    'TreeIndex',
    'TreeNode',
    
    # Database
    'DatabaseManager',
    'get_db_manager',
    'session_scope',
    'init_database',
    'get_db'
]
