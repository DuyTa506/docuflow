"""Integration jobs: clients, the stage/status model, and the result document."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from data.db_models import DigitizedText, Document, IntegrationClient, IntegrationJob, Task, Translation, User
from data.id_generator import IdGenerator
from services import integration_service as svc
from services.digest_service import DigestResult, KeywordEntry


@pytest.fixture
def db(test_db_session):
    IdGenerator.seed_sequences(test_db_session)
    test_db_session.commit()
    return test_db_session


@pytest.fixture
def client_and_key(db):
    client, key, secret = svc.create_client(db, "Thư viện Trường", callback_url="http://lib/cb")
    return client, key


def _doc(db, owner, **over):
    fields = dict(id=f"DOC_{uuid.uuid4().hex[:8]}", user_id=owner, title="Ballistics", source_language="en",
                  processing_status="INIT", pipeline_state="IDLE")
    fields.update(over)
    doc = Document(**fields)
    db.add(doc)
    db.commit()
    return doc


def _job(db, client, doc, tasks=("ocr", "digest"), **over):
    job = svc.create_job(db, client, document_id=doc.id, external_id="LIB-1", tasks=list(tasks),
                         callback_url=None)
    for k, v in over.items():
        setattr(job, k, v)
    db.commit()
    return job


class TestClients:
    def test_create_client_issues_key_once_and_a_service_user(self, db, client_and_key):
        client, key = client_and_key

        assert client.id.startswith("CLI_")
        assert key.startswith("dfk_") and key not in client.api_key_hash
        assert client.webhook_secret.startswith("whsec_")
        user = db.get(User, client.service_user_id)
        assert (user.group, user.role, user.status) == ("LIBRARY", "MEMBER", "ACTIVE")

    def test_authenticate(self, db, client_and_key):
        client, key = client_and_key

        assert svc.authenticate(db, key).id == client.id
        assert client.last_used_at is not None
        assert svc.authenticate(db, key + "x") is None
        assert svc.authenticate(db, "") is None

        client.status = "REVOKED"
        db.commit()
        assert svc.authenticate(db, key) is None


class TestTasks:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, ["ocr", "digest"]),
            ("", ["ocr", "digest"]),
            ("ocr", ["ocr"]),
            ("digest", ["ocr", "digest"]),  # every stage reads the OCR text
            ("translate, digest", ["ocr", "digest", "translate"]),
            (["TRANSLATE"], ["ocr", "translate"]),
        ],
    )
    def test_normalize(self, raw, expected):
        assert svc.normalize_tasks(raw) == expected

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="summary"):
            svc.normalize_tasks("ocr,summary")


def _state(processing="INIT", pipeline="IDLE", translation=None):
    doc = SimpleNamespace(processing_status=processing, pipeline_state=pipeline)
    tr = SimpleNamespace(status=translation) if translation else None
    return doc, tr


class TestStages:
    @pytest.mark.parametrize(
        "processing,pipeline,translation,expected",
        [
            ("INIT", "PENDING", "PENDING", ("PENDING", "PENDING", "PENDING")),
            ("EXTRACT_IN_PROGRESS", "PENDING", "PENDING", ("RUNNING", "PENDING", "PENDING")),
            ("EXTRACTED", "RUNNING", "IN_PROGRESS", ("COMPLETED", "RUNNING", "RUNNING")),
            ("EXTRACTED", "DONE", "COMPLETED", ("COMPLETED", "COMPLETED", "COMPLETED")),
            ("EXTRACTED", "FAILED", "COMPLETED", ("COMPLETED", "FAILED", "COMPLETED")),
            # Nothing downstream can run without text, whatever its row says yet.
            ("FAILED", "PENDING", "PENDING", ("FAILED", "FAILED", "FAILED")),
        ],
    )
    def test_derive(self, processing, pipeline, translation, expected):
        doc, tr = _state(processing, pipeline, translation)
        stages = svc.derive_stages(["ocr", "digest", "translate"], doc, tr, skipped=())
        assert (stages["ocr"], stages["digest"], stages["translate"]) == expected

    def test_only_requested_stages_and_skips(self):
        doc, _ = _state("EXTRACTED", "IDLE")
        assert svc.derive_stages(["ocr"], doc, None, skipped=()) == {"ocr": "COMPLETED"}
        assert svc.derive_stages(["ocr", "translate"], doc, None, skipped=("translate",)) == {
            "ocr": "COMPLETED",
            "translate": "SKIPPED",
        }

    @pytest.mark.parametrize(
        "stages,status",
        [
            ({"ocr": "PENDING", "digest": "PENDING"}, "QUEUED"),
            ({"ocr": "RUNNING", "digest": "PENDING"}, "PROCESSING"),
            ({"ocr": "COMPLETED", "digest": "PENDING"}, "PROCESSING"),
            ({"ocr": "COMPLETED", "digest": "COMPLETED"}, "COMPLETED"),
            ({"ocr": "COMPLETED", "translate": "SKIPPED"}, "COMPLETED"),
            ({"ocr": "COMPLETED", "digest": "FAILED"}, "PARTIAL"),
            ({"ocr": "FAILED", "digest": "FAILED"}, "FAILED"),
        ],
    )
    def test_job_status(self, stages, status):
        assert svc.job_status(stages) == status


class TestRefresh:
    def test_refresh_tracks_the_document_and_reports_changes(self, db, client_and_key):
        client, _ = client_and_key
        doc = _doc(db, client.service_user_id, processing_status="EXTRACT_IN_PROGRESS",
                   pipeline_state="PENDING")
        job = _job(db, client, doc)

        assert svc.refresh_job(db, job) is True
        assert job.status == "PROCESSING" and job.stages == {"ocr": "RUNNING", "digest": "PENDING"}
        assert svc.refresh_job(db, job) is False

        doc.processing_status, doc.pipeline_state = "EXTRACTED", "DONE"
        db.commit()
        assert svc.refresh_job(db, job) is True
        assert job.status == "COMPLETED" and job.completed_at is not None

    def test_failure_reason_comes_from_the_failed_task(self, db, client_and_key):
        client, _ = client_and_key
        doc = _doc(db, client.service_user_id, processing_status="FAILED")
        db.add(Task(id=f"TASK_{uuid.uuid4().hex[:8]}", document_id=doc.id, task_type="EXTRACT", status="FAILED",
                    error="PDF is password-protected"))
        job = _job(db, client, doc, tasks=["ocr"])

        svc.refresh_job(db, job)
        assert job.status == "FAILED"
        assert "password-protected" in job.error


class TestViewsAndResult:
    def test_job_view_links(self, db, client_and_key):
        client, _ = client_and_key
        job = _job(db, client, _doc(db, client.service_user_id), tasks=["ocr", "digest", "translate"])
        view = svc.job_view(job, "http://df")

        assert view["job_id"] == job.id and view["external_id"] == "LIB-1"
        assert view["links"]["result"] == f"http://df/api/v2/integration/jobs/{job.id}/result"
        assert view["links"]["digest_docx"].endswith("/files/digest?format=docx")
        assert view["links"]["translation_docx"].endswith("/files/translation?format=docx")

    def test_result_holds_every_finished_output(self, db, client_and_key):
        client, _ = client_and_key
        doc = _doc(db, client.service_user_id, processing_status="EXTRACTED", pipeline_state="DONE")
        tr_id = uuid.uuid4().hex
        db.add(DigitizedText(document_id=doc.id, ocr_content="raw", normalized_content="Chapter 1 …"))
        db.add(Translation(id=tr_id, document_id=doc.id, target_language="vi", status="COMPLETED",
                           translated_content="Chương 1 …"))
        job = _job(db, client, doc, tasks=["ocr", "digest", "translate"], translation_id=tr_id)
        svc.refresh_job(db, job)

        digest = DigestResult(document_id=doc.id, title="Ballistics", source_language="en",
                              original_filename="b.pdf", bibliographic={"title_display": "Ballistics"},
                              keywords=[KeywordEntry("ballistics", 0.9, "đạn đạo")])
        with patch.object(svc._digest_svc, "assemble", return_value=digest):
            result = svc.build_result(db, job, "http://df")

        assert result["job"]["status"] == "COMPLETED"
        assert result["ocr"]["text"] == "Chapter 1 …"
        assert result["translation"] == {"target_language": "vi", "text": "Chương 1 …"}
        assert result["digest"]["keywords"][0]["display"] == "đạn đạo"
        assert result["catalog"]["marcxml"].startswith("<marc:record")
        assert "oai_dc:dc" in result["catalog"]["dublin_core"]

    def test_unfinished_stages_are_null(self, db, client_and_key):
        client, _ = client_and_key
        doc = _doc(db, client.service_user_id, processing_status="EXTRACTED", pipeline_state="RUNNING")
        db.add(DigitizedText(document_id=doc.id, ocr_content="raw text"))
        job = _job(db, client, doc)
        svc.refresh_job(db, job)

        result = svc.build_result(db, job, "http://df")
        assert result["ocr"]["text"] == "raw text"
        assert result["digest"] is None and result["catalog"] is None
        assert result["translation"] is None


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submits_each_requested_stage(self, db, client_and_key):
        client, _ = client_and_key
        doc = _doc(db, client.service_user_id)
        job = _job(db, client, doc, tasks=["ocr", "digest", "translate"])
        with patch.object(svc._doc_svc, "submit_extraction_async", AsyncMock(return_value=("T1", False))) as ex, \
             patch.object(svc, "submit_digest", AsyncMock(return_value=("wf", "T2", False))) as dg, \
             patch.object(svc._translation_svc, "submit_async", AsyncMock(return_value=("T3", "TR_9", False))) as tr:
            await svc.submit_stages(db, job)

        ex.assert_awaited_once_with(db, doc.id, fairness_key=client.service_user_id)
        dg.assert_awaited_once_with(db, doc.id, fairness_key=client.service_user_id)
        tr.assert_awaited_once_with(db, doc.id, target_language="vi", fairness_key=client.service_user_id)
        assert job.translation_id == "TR_9"

    @pytest.mark.asyncio
    async def test_vietnamese_source_skips_translation(self, db, client_and_key):
        client, _ = client_and_key
        job = _job(db, client, _doc(db, client.service_user_id, source_language="vi"),
                   tasks=["ocr", "translate"])
        with patch.object(svc._doc_svc, "submit_extraction_async", AsyncMock(return_value=("T1", False))), \
             patch.object(svc._translation_svc, "submit_async", AsyncMock()) as tr:
            await svc.submit_stages(db, job)

        tr.assert_not_awaited()
        svc.refresh_job(db, job)
        assert job.stages["translate"] == "SKIPPED"
