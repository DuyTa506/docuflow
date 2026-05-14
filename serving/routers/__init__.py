"""
Serving routers package — all v2 API routers for DocuFlow.
"""
from .auth_router import router as auth_router
from .documents_router import router as documents_router
from .tasks_router import router as tasks_router
from .translation_router import router as translation_router
from .summarization_router import router as summarization_router
from .main_content_router import router as main_content_router
from .keywords_router import router as keywords_router
from .research_router import router as research_router
from .search_router import router as search_router
from .tree_index_router import router as tree_index_router

__all__ = [
    "auth_router",
    "documents_router",
    "tasks_router",
    "translation_router",
    "summarization_router",
    "main_content_router",
    "keywords_router",
    "research_router",
    "search_router",
    "tree_index_router",
]
