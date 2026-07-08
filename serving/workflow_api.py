"""
DocuFlow API — application shell.

Instantiates the FastAPI app, registers all v2 routers,
and runs the startup initialisation.  No endpoint logic lives here.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    tree_index_router,
    pipeline_router,
)
from serving.routers.digest_router import router as digest_router
from serving.routers.analysis_router import router as analysis_router


# ── Create FastAPI app ──────────────────────────────────────────────

workflow_app = FastAPI(
    title="DocuFlow API",
    description="OCR processing + library management AI services",
    version="2.0.0",
)


# ── CORS ─────────────────────────────────────────────────────────────
# Allow all origins so ui.html can be opened as a local file (file://)
# or served from any dev host.  Tighten in production.

workflow_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register all routers ────────────────────────────────────────────

for _router in [
    auth_router,
    documents_router,
    tasks_router,
    translation_router,
    summarization_router,
    main_content_router,
    keywords_router,
    research_router,
    search_router,
    tree_index_router,
    digest_router,
    analysis_router,
    pipeline_router,
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
        "endpoints": {
            "auth":                "/api/v2/auth/*",
            "documents":           "/api/v2/documents/*",
            "tasks":               "/api/v2/tasks/*",
            "translations":        "/api/v2/documents/{id}/translations",
            "summaries":           "/api/v2/documents/{id}/summaries",
            "main_content":        "/api/v2/documents/{id}/main-content",
            "keywords":            "/api/v2/documents/{id}/keywords",
            "research_directions": "/api/v2/documents/{id}/research-directions",
            "digest_json":         "POST /api/v2/documents/{id}/digest",
            "digest_download":     "GET  /api/v2/documents/{id}/digest/download",
            "analysis":            "POST /api/v2/documents/{id}/analysis",
            "pipeline_status":     "GET  /api/v2/documents/{id}/pipeline-status",
            "search":              "/api/v2/search?q=...",
        },
    }


# Export app for uvicorn
app = workflow_app
