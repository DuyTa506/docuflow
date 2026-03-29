"""
DocuFlow API — application shell.

Instantiates the FastAPI app, registers all routers (v1 + v2),
and runs the startup initialisation.  No endpoint logic lives here.
"""
from fastapi import FastAPI

from data.database import init_database
from serving.routers import (
    auth_router,
    documents_router,
    tasks_router,
    translation_router,
    summarization_router,
    main_content_router,
    keywords_router,
    research_router,
    search_router,
    v1_router,
)
from serving.routers.digest_router import router as digest_router


# ── Create FastAPI app ──────────────────────────────────────────────

workflow_app = FastAPI(
    title="DocuFlow API",
    description="OCR processing + library management AI services",
    version="2.0.0",
)


# ── Register all routers ────────────────────────────────────────────

for _router in [
    v1_router,           # GET/POST /process-document, /build-index, /documents (v1)
    auth_router,
    documents_router,
    tasks_router,
    translation_router,
    summarization_router,
    main_content_router,
    keywords_router,
    research_router,
    search_router,
    digest_router,
]:
    workflow_app.include_router(_router)


# ── Startup ─────────────────────────────────────────────────────────

@workflow_app.on_event("startup")
async def startup_event():
    """Initialize database tables + seed sequences on startup."""
    init_database()
    print("DocuFlow API initialized")


# ── Discovery endpoint ───────────────────────────────────────────────

@workflow_app.get("/")
async def root():
    """API root — endpoint discovery."""
    return {
        "name": "DocuFlow API",
        "version": "2.0.0",
        "v1_endpoints": {
            "process_document": "POST /process-document",
            "build_index": "POST /build-index/{document_id}",
            "get_document": "GET /documents/{document_id}",
            "get_elements": "GET /documents/{document_id}/elements",
            "get_tree": "GET /documents/{document_id}/tree",
            "list_documents": "GET /documents",
        },
        "v2_endpoints": {
            "auth": "/api/v2/auth/*",
            "documents": "/api/v2/documents/*",
            "tasks": "/api/v2/tasks/*",
            "translations": "/api/v2/documents/{id}/translations",
            "summaries": "/api/v2/documents/{id}/summaries",
            "main_content": "/api/v2/documents/{id}/main-content",
            "keywords": "/api/v2/documents/{id}/keywords",
            "research_directions": "/api/v2/documents/{id}/research-directions",
            "digest_json": "POST /api/v2/documents/{id}/digest",
            "digest_download": "GET  /api/v2/documents/{id}/digest/download",
            "search": "/api/v2/search?q=...",
        },
    }


# Export app for uvicorn
app = workflow_app
