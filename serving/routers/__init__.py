"""
Serving routers package — all v2 API routers for DocuFlow.
"""

from .analysis_router import router as analysis_router
from .auth_router import router as auth_router
from .catalog_router import router as catalog_router
from .documents_router import router as documents_router
from .events_router import router as events_router
from .keywords_router import router as keywords_router
from .main_content_router import router as main_content_router
from .pipeline_router import router as pipeline_router
from .research_router import router as research_router
from .search_router import router as search_router
from .summarization_router import router as summarization_router
from .tasks_router import router as tasks_router
from .translation_router import router as translation_router
from .tree_index_router import router as tree_index_router

__all__ = [
    "auth_router",
    "catalog_router",
    "documents_router",
    "tasks_router",
    "translation_router",
    "summarization_router",
    "main_content_router",
    "keywords_router",
    "research_router",
    "search_router",
    "tree_index_router",
    "analysis_router",
    "pipeline_router",
    "events_router",
]
