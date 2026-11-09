"""Data access layer - Database models and connections."""

from .database import (
    DatabaseManager,
    get_db_manager,
    init_database,
    session_scope,
)
from .db_models import (
    Base,
    DigitizedText,
    Document,
    DocumentKeyword,
    DocumentResearchDirection,
    IdSequence,
    Keyword,
    LayoutElement,
    MainContent,
    Page,
    ResearchDirection,
    Summary,
    Task,
    Translation,
    TreeIndex,
    TreeNode,
    User,
)
from .id_generator import IdGenerator

__all__ = [
    # Models
    "Base",
    "Document",
    "Page",
    "LayoutElement",
    "TreeIndex",
    "TreeNode",
    "User",
    "IdSequence",
    "DigitizedText",
    "Translation",
    "Summary",
    "MainContent",
    "Keyword",
    "DocumentKeyword",
    "ResearchDirection",
    "DocumentResearchDirection",
    "Task",
    # Database
    "DatabaseManager",
    "get_db_manager",
    "session_scope",
    "init_database",
    # get_db is now in api/dependencies.py — use that for FastAPI injection
    # Utilities
    "IdGenerator",
]
