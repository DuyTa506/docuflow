"""Library-system integration: API clients, jobs and their results.

A library system (identified by an API key) submits a file with the tasks it
wants — ``ocr``, ``digest``, ``translate`` — and gets a job id back. The job's
stages are the document's own processing states read back, never a second
state machine: OCR is ``Document.processing_status``, the digest is
``Document.pipeline_state``, translation is the ``Translation`` row. The
IntegrationJobWorkflow re-reads them until every stage is final, then delivers
the result (MSSQL tables, signed callback).
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Union

from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import Document, IntegrationClient, IntegrationJob, Task, Translation, User
from data.id_generator import IdGenerator
from data.repositories import DocumentRepository
from services.auth_service import AuthService
from services.digest_service import DigestService, digest_to_dict
from services.document_service import DocumentService
from services.pipeline.job_queue import submit_digest
from services.translation_service import TranslationService
from utils.content_storage import read_text_field
from utils.integration_security import generate_api_key, generate_webhook_secret, hash_api_key
from utils.library_formats import to_dublin_core, to_marcxml
from utils.upload_convert import convert_upload

logger = logging.getLogger(__name__)

TASKS = ("ocr", "digest", "translate")
DEFAULT_TASKS = ["ocr", "digest"]
FINAL = frozenset({"COMPLETED", "FAILED", "SKIPPED"})
API_PREFIX = "/api/v2/integration"

_TASK_TYPE = {"ocr": "EXTRACT", "digest": "DIGEST_PIPELINE", "translate": "TRANSLATE"}
_OCR_STATE = {"EXTRACTED": "COMPLETED", "FAILED": "FAILED", "EXTRACT_IN_PROGRESS": "RUNNING"}
_DIGEST_STATE = {"DONE": "COMPLETED", "FAILED": "FAILED", "RUNNING": "RUNNING"}
_TRANSLATION_STATE = {"COMPLETED": "COMPLETED", "FAILED": "FAILED", "IN_PROGRESS": "RUNNING"}

_digest_svc = DigestService()
_doc_svc = DocumentService()
_translation_svc = TranslationService()


# ── Clients ──────────────────────────────────────────────────────────


def create_client(db: Session, name: str, callback_url: Optional[str] = None):
    """Register a library system. Returns (client, api_key, webhook_secret);
    the key is not recoverable afterwards."""
    key, prefix, key_hash = generate_api_key()
    user = User(
        id=IdGenerator.next_id(db, "users"),
        username=f"svc_{uuid.uuid4().hex[:10]}",
        password_hash=AuthService.hash_password(secrets.token_urlsafe(24)),
        full_name=f"Tích hợp: {name}",
        group="LIBRARY",
        role="MEMBER",
        status="ACTIVE",
    )
    db.add(user)
    client = IntegrationClient(
        id=IdGenerator.next_id(db, "integration_clients"),
        name=name,
        api_key_prefix=prefix,
        api_key_hash=key_hash,
        webhook_secret=generate_webhook_secret(),
        callback_url=callback_url,
        service_user_id=user.id,
        status="ACTIVE",
    )
    db.add(client)
    db.commit()
    return client, key, client.webhook_secret


def authenticate(db: Session, api_key: Optional[str]) -> Optional[IntegrationClient]:
    if not api_key:
        return None
    client = (
        db.query(IntegrationClient)
        .filter(IntegrationClient.api_key_hash == hash_api_key(api_key))
        .first()
    )
    if client is None or client.status != "ACTIVE":
        return None
    client.last_used_at = datetime.utcnow()
    db.commit()
    return client


# ── Jobs ─────────────────────────────────────────────────────────────


def normalize_tasks(raw: Union[None, str, Sequence[str]]) -> List[str]:
    """Canonical task list. OCR is always included: every stage reads its text."""
    items = raw.split(",") if isinstance(raw, str) else list(raw or [])
    wanted = {t.strip().lower() for t in items if t and t.strip()}
    unknown = wanted - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown))}. Allowed: {', '.join(TASKS)}")
    if not wanted:
        return list(DEFAULT_TASKS)
    wanted.add("ocr")
    return [t for t in TASKS if t in wanted]


def store_upload(db: Session, fileobj, filename: str, *, user_id: str, title: Optional[str],
                 source_language: str) -> Document:
    """Save an uploaded stream and register it — the same path the UI upload takes."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = os.path.splitext(filename)[1].lower()
    dest = os.path.join(settings.upload_dir, f"_upload_{uuid.uuid4().hex}{ext}")
    stored = dest
    try:
        copied = 0
        with open(dest, "wb") as out:
            while chunk := fileobj.read(1024 * 1024):
                copied += len(chunk)
                if copied > settings.max_upload_bytes:
                    raise ValueError(f"File exceeds the {settings.max_upload_bytes} byte upload limit.")
                out.write(chunk)
        stored = convert_upload(dest)
        return _doc_svc.upload_document(
            db,
            file_path_on_disk=stored,
            original_filename=filename,
            user_id=user_id,
            title=title,
            source_language=source_language,
        )
    except Exception:
        for path in {dest, stored}:
            try:
                os.remove(path)
            except OSError:
                pass
        raise


def create_job(db: Session, client: IntegrationClient, *, document_id: str,
               external_id: Optional[str], tasks: List[str],
               callback_url: Optional[str]) -> IntegrationJob:
    job = IntegrationJob(
        id=IdGenerator.next_id(db, "integration_jobs"),
        client_id=client.id,
        external_id=external_id,
        document_id=document_id,
        tasks=list(tasks),
        stages={t: "PENDING" for t in tasks},
        callback_url=callback_url or client.callback_url,
        status="QUEUED",
    )
    db.add(job)
    db.commit()
    return job


async def submit_stages(db: Session, job: IntegrationJob) -> None:
    """Start every requested stage now; each one waits for OCR on its own."""
    owner = job.client.service_user_id
    doc = db.get(Document, job.document_id)
    await _doc_svc.submit_extraction_async(db, doc.id, fairness_key=owner)
    if "digest" in job.tasks:
        await submit_digest(db, doc.id, fairness_key=owner)
    if "translate" in job.tasks:
        if (doc.source_language or "").lower().startswith("vi"):
            job.stages = {**(job.stages or {}), "translate": "SKIPPED"}
        else:
            _, translation_id, _ = await _translation_svc.submit_async(
                db, doc.id, target_language="vi", fairness_key=owner
            )
            job.translation_id = translation_id
    db.commit()


def derive_stages(tasks: Iterable[str], doc, translation, *, skipped: Iterable[str]) -> dict:
    skipped = set(skipped)
    ocr = _OCR_STATE.get(doc.processing_status, "PENDING")
    derived = {
        "ocr": ocr,
        "digest": _DIGEST_STATE.get(doc.pipeline_state, "PENDING"),
        "translate": _TRANSLATION_STATE.get(translation.status, "PENDING") if translation else "PENDING",
    }
    stages = {}
    for task in tasks:
        if task in skipped:
            stages[task] = "SKIPPED"
        elif task != "ocr" and ocr == "FAILED":
            stages[task] = "FAILED"
        else:
            stages[task] = derived[task]
    return stages


def job_status(stages: dict) -> str:
    values = list(stages.values())
    if all(v in FINAL for v in values):
        if all(v == "FAILED" for v in values):
            return "FAILED"
        return "PARTIAL" if "FAILED" in values else "COMPLETED"
    if all(v == "PENDING" for v in values):
        return "QUEUED"
    return "PROCESSING"


def _failure_reason(db: Session, document_id: str, stages: dict) -> Optional[str]:
    reasons = []
    for stage, state in stages.items():
        if state != "FAILED":
            continue
        task = (
            db.query(Task)
            .filter(Task.document_id == document_id, Task.task_type == _TASK_TYPE[stage],
                    Task.status == "FAILED")
            .order_by(Task.created_at.desc())
            .first()
        )
        detail = (task.error or task.message) if task else None
        reasons.append(f"{stage}: {detail or 'failed'}")
    return "; ".join(reasons) or None


def refresh_job(db: Session, job: IntegrationJob) -> bool:
    """Re-read the document's states into the job. True when anything changed."""
    doc = db.get(Document, job.document_id)
    translation = db.get(Translation, job.translation_id) if job.translation_id else None
    skipped = [k for k, v in (job.stages or {}).items() if v == "SKIPPED"]
    stages = derive_stages(job.tasks, doc, translation, skipped=skipped)
    status = job_status(stages)
    if stages == job.stages and status == job.status:
        return False
    job.stages = stages
    job.status = status
    job.error = _failure_reason(db, doc.id, stages)
    if status in ("COMPLETED", "PARTIAL", "FAILED") and job.completed_at is None:
        job.completed_at = datetime.utcnow()
    db.commit()
    return True


# ── Views ────────────────────────────────────────────────────────────


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(timespec="seconds") if value else None


def job_view(job: IntegrationJob, base_url: str) -> dict:
    root = f"{base_url.rstrip('/')}{API_PREFIX}/jobs/{job.id}"
    links = {"self": root, "result": f"{root}/result"}
    if "digest" in job.tasks:
        links["result_marcxml"] = f"{root}/result?format=marcxml"
        links["result_dc"] = f"{root}/result?format=dc"
    for kind, task in (("ocr", "ocr"), ("digest", "digest"), ("translation", "translate")):
        if task in job.tasks:
            for fmt in ("docx", "pdf"):
                links[f"{kind}_{fmt}"] = f"{root}/files/{kind}?format={fmt}"
    return {
        "job_id": job.id,
        "external_id": job.external_id,
        "document_id": job.document_id,
        "status": job.status,
        "tasks": list(job.tasks),
        "stages": dict(job.stages or {}),
        "error": job.error,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
        "completed_at": _iso(job.completed_at),
        "links": links,
    }


def _ocr_text(db: Session, document_id: str) -> str:
    dt = DocumentRepository(db).get_digitized_text(document_id)
    if dt is None:
        return ""
    return read_text_field(inline=dt.normalized_content, key=dt.normalized_content_key) or read_text_field(
        inline=dt.ocr_content, key=dt.ocr_content_key
    )


def assemble_digest(db: Session, job: IntegrationJob):
    return _digest_svc.assemble(db, job.document_id)


def build_result(db: Session, job: IntegrationJob, base_url: str) -> dict:
    """Everything the job has produced so far; unfinished stages are null."""
    doc = db.get(Document, job.document_id)
    stages = job.stages or {}
    result = {
        "job": job_view(job, base_url),
        "document": {
            "id": doc.id,
            "title": doc.title,
            "source_language": doc.source_language,
            "original_filename": doc.original_filename,
            "total_pages": doc.total_pages,
        },
        "ocr": None,
        "digest": None,
        "catalog": None,
        "translation": None,
    }
    if stages.get("ocr") == "COMPLETED":
        result["ocr"] = {"text": _ocr_text(db, doc.id)}
    if stages.get("digest") == "COMPLETED":
        digest = assemble_digest(db, job)
        result["digest"] = digest_to_dict(digest)
        result["catalog"] = {"marcxml": to_marcxml(digest), "dublin_core": to_dublin_core(digest)}
    if stages.get("translate") == "COMPLETED" and job.translation_id:
        tr = db.get(Translation, job.translation_id)
        result["translation"] = {
            "target_language": tr.target_language,
            "text": read_text_field(inline=tr.translated_content, key=tr.translated_content_key),
        }
    return result
