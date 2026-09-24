"""
DocuFlow API — application shell.

Instantiates the FastAPI app, registers all v2 routers,
and runs the startup initialisation.  No endpoint logic lives here.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from data.database import init_database
from services.pipeline.admission import AdmissionRejected, http_exception
from serving.health import router as health_router
from serving.routers import (
    auth_router,
    catalog_router,
    documents_router,
    events_router,
    keywords_router,
    main_content_router,
    pipeline_router,
    research_router,
    search_router,
    summarization_router,
    tasks_router,
    translation_router,
    tree_index_router,
)
from serving.routers.analysis_router import router as analysis_router
from serving.routers.digest_router import router as digest_router
from serving.spa import mount_spa

# ── Create FastAPI app ──────────────────────────────────────────────

_prod = os.environ.get("DOCUFLOW_PROD", "").strip().lower() in ("1", "true", "yes")

workflow_app = FastAPI(
    title="DocuFlow API",
    description="OCR processing + library management AI services",
    version="2.0.0",
    docs_url=None if _prod else "/docs",
    redoc_url=None if _prod else "/redoc",
    openapi_url=None if _prod else "/openapi.json",
)


def _cors_origins() -> list[str]:
    raw = (settings.cors_allow_origins or "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()] or ["*"]


# ── CORS ─────────────────────────────────────────────────────────────
# Same-origin SPA at :8022 does not need *. Keep * for local file:// / :4200
# unless CORS_ALLOW_ORIGINS is set.

_origins = _cors_origins()
workflow_app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register all routers ────────────────────────────────────────────

workflow_app.include_router(health_router)


@workflow_app.exception_handler(AdmissionRejected)
async def _admission_rejected_handler(_request: Request, exc: AdmissionRejected):
    http = http_exception(exc)
    return JSONResponse(
        status_code=http.status_code,
        content=http.detail,
        headers=dict(http.headers or {}),
    )


for _router in [
    auth_router,
    documents_router,
    tasks_router,
    translation_router,
    summarization_router,
    main_content_router,
    keywords_router,
    research_router,
    catalog_router,
    search_router,
    tree_index_router,
    digest_router,
    analysis_router,
    pipeline_router,
    events_router,
]:
    workflow_app.include_router(_router)


# ── Startup ─────────────────────────────────────────────────────────


@workflow_app.on_event("startup")
async def startup_event():
    """Initialize database tables + seed sequences on startup."""
    init_database()

    # In-process asyncio tasks died with the previous process — fail their
    # DB rows now so documents don't sit at RUNNING/IN_PROGRESS forever.
    from data.database import get_db_manager
    from services.task_manager import fail_orphaned_tasks

    with get_db_manager().session() as db:
        swept = fail_orphaned_tasks(db)
    if swept:
        print(f"Failed {swept} orphaned task/translation row(s) from previous run")

    try:
        from services.pipeline.job_queue import drain_waiting_queues

        await drain_waiting_queues()
    except Exception as exc:
        print(f"WARNING: waiting-job drain skipped: {exc}")

    # Say so loudly at boot rather than letting a DOCX export quietly drop its
    # formulas weeks later.
    from utils.native_deps import log_native_dependency_warnings

    for message in log_native_dependency_warnings():
        print(f"WARNING: {message}")

    print("DocuFlow API initialized")


# ── Discovery endpoint ───────────────────────────────────────────────
# On /api rather than /, because / now belongs to the frontend: users type the
# server's address and expect the app, not a JSON index.


@workflow_app.get("/api")
async def root():
    """API root — endpoint discovery."""
    return {
        "name": "DocuFlow API",
        "version": "2.0.0",
        "endpoints": {
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
            "analysis": "POST /api/v2/documents/{id}/analysis",
            "pipeline_status": "GET  /api/v2/documents/{id}/pipeline-status",
            "search": "/api/v2/search?q=...",
        },
    }


# ── Frontend ─────────────────────────────────────────────────────────
# Last, because it claims every path the routers above did not: one origin for
# app and API is what lets the frontend address the API relatively, and a
# relative URL is the only one that is correct from a LAN IP, a port-forward
# and a hostname at the same time.

mount_spa(workflow_app)


# Export app for uvicorn
app = workflow_app
