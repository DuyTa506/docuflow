"""
Library-system integration API.

A library system authenticates with ``X-API-Key`` (issued by an admin), submits
a document with the tasks it wants, and receives the outcome three ways: by
polling the job, by a signed callback when it finishes, and — when
``INTEGRATION_MSSQL_URL`` is set — as rows in its own MSSQL database.

  GET    /api/v2/integration/ping                    check the key
  POST   /api/v2/integration/jobs                    submit a file + tasks
  GET    /api/v2/integration/jobs                    list (filter by external_id, status)
  GET    /api/v2/integration/jobs/{id}               status of each stage
  GET    /api/v2/integration/jobs/{id}/result        ?format=json|marcxml|dc
  GET    /api/v2/integration/jobs/{id}/files/{kind}  kind=ocr|digest|translation, ?format=docx|pdf
  POST   /api/v2/integration/clients                 (ADMIN, JWT) issue a key
  GET    /api/v2/integration/clients                 (ADMIN, JWT)
  DELETE /api/v2/integration/clients/{id}            (ADMIN, JWT) revoke
"""

import asyncio
import logging
import math
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.dependencies import get_db, require_role
from config.settings import settings
from data.db_models import Document, IntegrationClient, IntegrationJob, Translation
from services import integration_service as svc
from services.export_service import export_service
from services.pipeline.temporal_client import start_integration_workflow
from utils.file_download import build_bytes_file_response, build_stored_file_response
from utils.library_formats import to_dublin_core, to_marcxml
from utils.url_safety import UnsafeUrl, validate_callback_url
from utils.upload_convert import UPLOAD_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/integration", tags=["integration"])

_FILE_KINDS = {"ocr": "ocr", "digest": "digest", "translation": "translate"}


def _base_url() -> str:
    return settings.integration_public_base_url


_api_key = APIKeyHeader(name="X-API-Key", auto_error=False, description="Khoá do quản trị DocuFlow cấp")


def get_integration_client(
    x_api_key: Optional[str] = Security(_api_key),
    db: Session = Depends(get_db),
) -> IntegrationClient:
    client = svc.authenticate(db, x_api_key)
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return client


def _own_job(job_id: str, client: IntegrationClient, db: Session) -> IntegrationJob:
    job = db.get(IntegrationJob, job_id)
    if job is None or job.client_id != client.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/ping")
async def ping(client: IntegrationClient = Depends(get_integration_client)):
    return {"client_id": client.id, "client_name": client.name, "status": client.status}


# ── Jobs ─────────────────────────────────────────────────────────────


@router.post("/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    external_id: Optional[str] = Form(None, max_length=100),
    title: Optional[str] = Form(None),
    source_language: str = Form("en"),
    tasks: Optional[str] = Form(None, description="Comma-separated: ocr, digest, translate"),
    callback_url: Optional[str] = Form(None),
    client: IntegrationClient = Depends(get_integration_client),
    db: Session = Depends(get_db),
):
    """Submit a document. Returns the job at once; stages run in the background."""
    try:
        wanted = svc.normalize_tasks(tasks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if callback_url:
        try:
            callback_url = validate_callback_url(callback_url)
        except UnsafeUrl as exc:
            raise HTTPException(status_code=400, detail=f"Invalid callback_url: {exc}") from exc
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(UPLOAD_EXTENSIONS)}",
        )

    try:
        doc = await asyncio.to_thread(
            svc.store_upload,
            db,
            file.file,
            file.filename,
            user_id=client.service_user_id,
            title=title,
            source_language=source_language,
        )
    except ValueError as exc:
        status = 413 if "upload limit" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = svc.create_job(
        db,
        client,
        document_id=doc.id,
        external_id=external_id,
        tasks=wanted,
        callback_url=callback_url,
    )
    try:
        await svc.submit_stages(db, job)
        await start_integration_workflow(job.id)
    except Exception as exc:
        logger.exception("Integration job %s could not be started", job.id)
        db.rollback()
        job.status, job.error = "FAILED", f"Could not start processing: {exc}"
        db.commit()
        raise HTTPException(
            status_code=503, detail=f"Job {job.id}: could not start processing"
        ) from exc
    return svc.job_view(job, _base_url())


@router.get("/jobs")
async def list_jobs(
    external_id: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    client: IntegrationClient = Depends(get_integration_client),
    db: Session = Depends(get_db),
):
    q = db.query(IntegrationJob).filter(IntegrationJob.client_id == client.id)
    if external_id:
        q = q.filter(IntegrationJob.external_id == external_id)
    if status:
        q = q.filter(IntegrationJob.status == status.upper())
    total = q.count()
    jobs = q.order_by(IntegrationJob.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "items": [svc.job_view(j, _base_url()) for j in jobs],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": math.ceil(total / limit) if total else 0,
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    client: IntegrationClient = Depends(get_integration_client),
    db: Session = Depends(get_db),
):
    job = _own_job(job_id, client, db)
    svc.refresh_job(db, job)  # current now, not as of the workflow's last poll
    return svc.job_view(job, _base_url())


@router.get("/jobs/{job_id}/result")
async def get_result(
    job_id: str,
    format: str = Query("json", pattern="^(json|marcxml|dc)$"),
    client: IntegrationClient = Depends(get_integration_client),
    db: Session = Depends(get_db),
):
    """Everything produced so far (json), or the catalogue record (marcxml, dc)."""
    job = _own_job(job_id, client, db)
    svc.refresh_job(db, job)
    if format == "json":
        return await asyncio.to_thread(svc.build_result, db, job, _base_url())

    if (job.stages or {}).get("digest") != "COMPLETED":
        raise HTTPException(status_code=409, detail="The digest of this job is not finished")
    digest = await asyncio.to_thread(svc.assemble_digest, db, job)
    if format == "marcxml":
        body, media = to_marcxml(digest), "application/marcxml+xml"
    else:
        body, media = to_dublin_core(digest), "application/xml"
    return Response(content=body, media_type=f"{media}; charset=utf-8")


@router.get("/jobs/{job_id}/files/{kind}")
async def get_file(
    job_id: str,
    kind: str,
    format: str = Query("docx", pattern="^(docx|pdf)$"),
    client: IntegrationClient = Depends(get_integration_client),
    db: Session = Depends(get_db),
):
    """The same DOCX/PDF exports the web app offers for OCR, digest and translation."""
    job = _own_job(job_id, client, db)
    if kind not in _FILE_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown file kind '{kind}'")
    svc.refresh_job(db, job)
    if (job.stages or {}).get(_FILE_KINDS[kind]) != "COMPLETED":
        raise HTTPException(status_code=409, detail=f"The {kind} stage of this job is not finished")

    doc = db.get(Document, job.document_id)
    try:
        if kind == "digest":
            built = await asyncio.to_thread(
                export_service.get_or_build_digest_export, db, doc.id, format
            )
        elif kind == "translation":
            tr = db.get(Translation, job.translation_id)
            built = await asyncio.to_thread(
                export_service.get_or_build_translation_export,
                db, doc, tr, source="auto", fmt=format, pdf_mode="auto",
            )
        else:
            built = await asyncio.to_thread(
                export_service.get_or_build_ocr_export,
                db, doc, content_type="ocr", mode="auto", fmt=format, source="extracted",
            )
    except Exception as exc:
        logger.exception("Integration export failed for %s (%s/%s)", job.id, kind, format)
        raise HTTPException(status_code=500, detail="Export failed") from exc

    key, filename, media_type, data = built
    if data is not None:
        export_service.schedule_export_put(key, data, content_type=media_type)
        return build_bytes_file_response(data, filename, media_type)
    return await asyncio.to_thread(
        build_stored_file_response, key, download_name=filename, content_type=media_type
    )


# ── Clients (admin) ──────────────────────────────────────────────────


class ClientCreate(BaseModel):
    name: str
    callback_url: Optional[str] = None


def _client_view(client: IntegrationClient) -> dict:
    return {
        "client_id": client.id,
        "name": client.name,
        "api_key_prefix": client.api_key_prefix,
        "callback_url": client.callback_url,
        "status": client.status,
        "service_user_id": client.service_user_id,
        "created_at": svc._iso(client.created_at),
        "last_used_at": svc._iso(client.last_used_at),
    }


@router.post("/clients", status_code=201, dependencies=[Depends(require_role("ADMIN"))])
async def create_client(body: ClientCreate, db: Session = Depends(get_db)):
    """Issue an API key. The key and webhook secret are shown only in this response."""
    if body.callback_url:
        try:
            body.callback_url = validate_callback_url(body.callback_url)
        except UnsafeUrl as exc:
            raise HTTPException(status_code=400, detail=f"Invalid callback_url: {exc}") from exc
    client, key, secret = svc.create_client(db, body.name, callback_url=body.callback_url)
    return {**_client_view(client), "api_key": key, "webhook_secret": secret}


@router.get("/clients", dependencies=[Depends(require_role("ADMIN"))])
async def list_clients(db: Session = Depends(get_db)):
    clients = db.query(IntegrationClient).order_by(IntegrationClient.created_at).all()
    return [_client_view(c) for c in clients]


@router.delete("/clients/{client_id}", status_code=204, dependencies=[Depends(require_role("ADMIN"))])
async def revoke_client(client_id: str, db: Session = Depends(get_db)):
    client = db.get(IntegrationClient, client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    client.status = "REVOKED"
    db.commit()
    return Response(status_code=204)
