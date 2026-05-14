"""
data/repositories — per-aggregate repository classes.

Each class handles DB queries for one aggregate only; no business logic.
"""
from .document_repo import DocumentRepository
from .summary_repo import SummaryRepository
from .keyword_repo import KeywordRepository
from .translation_repo import TranslationRepository
from .research_repo import ResearchRepository
from .task_repo import TaskRepository
from .main_content_repo import MainContentRepository

__all__ = [
    "DocumentRepository",
    "SummaryRepository",
    "KeywordRepository",
    "TranslationRepository",
    "ResearchRepository",
    "TaskRepository",
    "MainContentRepository",
]
