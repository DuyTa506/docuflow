"""/api/v2/integration — the contract a library system codes against."""

import io
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import get_current_user, get_db
from data.db_models import Base, Document
from data.id_generator import IdGenerator
from serving.routers import integration_router as ir
from services import integration_service as svc
from services.digest_service import DigestResult


@pytest.fixture
def db():
    # TestClient serves requests on another thread: one shared connection.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(session)
    session.commit()
    yield session
    session.close()


@pytest.fixture
def admin():
    return SimpleNamespace(id="USR_ADMIN", role="ADMIN", group="LIBRARY", status="ACTIVE")


@pytest.fixture
def app(db, admin):
    app = FastAPI()
    app.include_router(ir.router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: admin
    return app


@pytest.fixture
def http(app):
    return TestClient(app)


@pytest.fixture
def key(db):
    _, key, _ = svc.create_client(db, "Thư viện A", callback_url="http://lib/cb")
    return key


@pytest.fixture
def stubs(db):
    """Upload stores a Document row; stage submission and Temporal are stubbed."""

    def fake_store(db_, fileobj, filename, *, user_id, title, source_language):
        doc = Document(id=f"DOC_{uuid.uuid4().hex[:8]}", user_id=user_id, title=title or filename,
                       original_filename=filename, source_language=source_language,
                       processing_status="INIT", pipeline_state="IDLE")
        db_.add(doc)
        db_.commit()
        return doc

    with patch.object(ir.svc, "store_upload", side_effect=fake_store), \
         patch.object(ir.svc, "submit_stages", AsyncMock()) as submit, \
         patch.object(ir, "start_integration_workflow", AsyncMock(return_value="wf")) as start:
        yield SimpleNamespace(submit=submit, start=start)


def _submit(http, key, **form):
    data = {"external_id": "LIB-42", "tasks": "ocr,digest", "source_language": "en", **form}
    return http.post(
        "/api/v2/integration/jobs",
        headers={"X-API-Key": key},
        files={"file": ("book.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        data=data,
    )


class TestAuth:
    def test_key_required(self, http):
        assert http.get("/api/v2/integration/ping").status_code == 401
        assert http.get("/api/v2/integration/ping", headers={"X-API-Key": "dfk_nope"}).status_code == 401

    def test_ping(self, http, key):
        r = http.get("/api/v2/integration/ping", headers={"X-API-Key": key})
        assert r.status_code == 200 and r.json()["client_name"] == "Thư viện A"


class TestJobs:
    def test_submit(self, http, key, stubs):
        r = _submit(http, key)

        assert r.status_code == 202, r.text
        body = r.json()
        assert body["job_id"].startswith("JOB_")
        assert body["external_id"] == "LIB-42"
        assert body["tasks"] == ["ocr", "digest"]
        assert body["status"] == "QUEUED"
        assert body["links"]["result"].endswith(f"/jobs/{body['job_id']}/result")
        stubs.submit.assert_awaited_once()
        stubs.start.assert_awaited_once_with(body["job_id"])

    def test_bad_requests(self, http, key, stubs):
        assert _submit(http, key, tasks="ocr,summary").status_code == 422
        r = http.post("/api/v2/integration/jobs", headers={"X-API-Key": key},
                      files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
        assert r.status_code == 400

    def test_submit_failure_marks_the_job_failed(self, http, key, stubs, db):
        stubs.submit.side_effect = RuntimeError("Temporal unavailable")
        r = _submit(http, key)
        assert r.status_code == 503
        assert "Temporal unavailable" in r.json()["detail"]

    def test_status_list_and_isolation(self, http, key, stubs, db):
        job_id = _submit(http, key).json()["job_id"]
        _, other_key, _ = svc.create_client(db, "Thư viện B")

        r = http.get(f"/api/v2/integration/jobs/{job_id}", headers={"X-API-Key": key})
        assert r.status_code == 200 and r.json()["job_id"] == job_id
        assert http.get(f"/api/v2/integration/jobs/{job_id}",
                        headers={"X-API-Key": other_key}).status_code == 404

        page = http.get("/api/v2/integration/jobs", params={"external_id": "LIB-42"},
                        headers={"X-API-Key": key}).json()
        assert set(page) == {"items", "total", "page", "limit", "total_pages"}
        assert [j["job_id"] for j in page["items"]] == [job_id]
        assert http.get("/api/v2/integration/jobs", headers={"X-API-Key": other_key}).json()["total"] == 0


class TestResult:
    def _finished(self, http, key, db, pipeline_state):
        job_id = _submit(http, key).json()["job_id"]
        job = db.get(svc.IntegrationJob, job_id)
        doc = db.get(Document, job.document_id)
        doc.processing_status, doc.pipeline_state = "EXTRACTED", pipeline_state
        db.commit()
        return job_id

    def test_json_result(self, http, key, stubs, db):
        job_id = self._finished(http, key, db, "RUNNING")
        with patch.object(svc, "_ocr_text", return_value="văn bản"):
            r = http.get(f"/api/v2/integration/jobs/{job_id}/result", headers={"X-API-Key": key})
        assert r.status_code == 200
        assert r.json()["ocr"] == {"text": "văn bản"} and r.json()["digest"] is None

    def test_catalogue_formats_need_a_finished_digest(self, http, key, stubs, db):
        job_id = self._finished(http, key, db, "RUNNING")
        r = http.get(f"/api/v2/integration/jobs/{job_id}/result", params={"format": "marcxml"},
                     headers={"X-API-Key": key})
        assert r.status_code == 409

    @pytest.mark.parametrize("fmt,media,marker", [
        ("marcxml", "application/marcxml+xml", "<marc:record"),
        ("dc", "application/xml", "<oai_dc:dc"),
    ])
    def test_catalogue_formats(self, http, key, stubs, db, fmt, media, marker):
        job_id = self._finished(http, key, db, "DONE")
        digest = DigestResult(document_id="D", title="T", source_language="en", original_filename=None,
                              bibliographic={"title_display": "Đạn đạo"})
        with patch.object(svc._digest_svc, "assemble", return_value=digest):
            r = http.get(f"/api/v2/integration/jobs/{job_id}/result", params={"format": fmt},
                         headers={"X-API-Key": key})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(media)
        assert marker in r.text and "Đạn đạo" in r.text

    def test_files_need_a_finished_stage(self, http, key, stubs, db):
        job_id = self._finished(http, key, db, "RUNNING")
        r = http.get(f"/api/v2/integration/jobs/{job_id}/files/digest", headers={"X-API-Key": key})
        assert r.status_code == 409
        r = http.get(f"/api/v2/integration/jobs/{job_id}/files/translation", headers={"X-API-Key": key})
        assert r.status_code == 409  # not requested at all


class TestClientsAdmin:
    def test_admin_issues_lists_and_revokes(self, http, db):
        r = http.post("/api/v2/integration/clients", json={"name": "Thư viện C",
                                                             "callback_url": "http://c/cb"})
        assert r.status_code == 201
        created = r.json()
        assert created["api_key"].startswith("dfk_") and created["webhook_secret"].startswith("whsec_")

        listed = http.get("/api/v2/integration/clients").json()
        mine = next(c for c in listed if c["client_id"] == created["client_id"])
        assert "api_key" not in mine and "webhook_secret" not in mine
        assert mine["api_key_prefix"] == created["api_key"][:12]

        assert http.get("/api/v2/integration/ping", headers={"X-API-Key": created["api_key"]}).status_code == 200
        assert http.delete(f"/api/v2/integration/clients/{created['client_id']}").status_code == 204
        assert http.get("/api/v2/integration/ping", headers={"X-API-Key": created["api_key"]}).status_code == 401

    def test_members_cannot_manage_clients(self, app, http):
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="USR_M", role="MEMBER", group="TEACHER", status="ACTIVE")
        assert http.post("/api/v2/integration/clients", json={"name": "x"}).status_code == 403
        assert http.get("/api/v2/integration/clients").status_code == 403
