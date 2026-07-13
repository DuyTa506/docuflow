"""
data/repositories — per-aggregate repository classes.

Each class handles DB queries for one aggregate only; no business logic.
"""

from .document_repo import DocumentRepository
from .keyword_repo import KeywordRepository
from .main_content_repo import MainContentRepository
from .research_repo import ResearchRepository
from .summary_repo import SummaryRepository
from .task_repo import TaskRepository
from .translation_repo import TranslationRepository

__all__ = [
    "DocumentRepository",
    "SummaryRepository",
    "KeywordRepository",
    "TranslationRepository",
    "ResearchRepository",
    "TaskRepository",
    "MainContentRepository",
]
