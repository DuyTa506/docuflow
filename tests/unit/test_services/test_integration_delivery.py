"""Delivering a job's outcome: the signed callback and the MSSQL write.

Delivery never changes the job's processing status — the pull API stays the
source of truth — but each attempt is recorded on the job so an operator can
see why a library did not receive its result.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa

from data.db_models import Document
from data.id_generator import IdGenerator
from services import integration_delivery as delivery
from services import integration_service as svc
from services.integration_sink import MssqlSink
from utils.integration_security import verify_callback


@pytest.fixture
def db(test_db_session):
    IdGenerator.seed_sequences(test_db_session)
    test_db_session.commit()
    return test_db_session


@pytest.fixture
def job(db):
    client, _, _ = svc.create_client(db, "Thư viện", callback_url="http://lib/cb")
    doc = Document(id=f"DOC_{uuid.uuid4().hex[:8]}", user_id=client.service_user_id, title="T",
                   source_language="en", processing_status="EXTRACTED", pipeline_state="IDLE")
    db.add(doc)
    db.commit()
    job = svc.create_job(db, client, document_id=doc.id, external_id="LIB-7", tasks=["ocr"],
                         callback_url=None)
    svc.refresh_job(db, job)
    return job


class TestCallback:
    def test_signed_post_of_the_job_view(self, db, job):
        post = MagicMock(return_value=SimpleNamespace(status_code=204, text=""))
        assert delivery.deliver_callback(db, job, "http://df", post=post, attempt=1) == "DELIVERED"

        url, = post.call_args.args
        body, headers = post.call_args.kwargs["content"], post.call_args.kwargs["headers"]
        assert url == "http://lib/cb"
        assert verify_callback(job.client.webhook_secret, headers["X-DocuFlow-Timestamp"], body,
                               headers["X-DocuFlow-Signature"])
        payload = json.loads(body)
        assert payload["event"] == "job.completed"
        assert payload["job"]["job_id"] == job.id and payload["job"]["external_id"] == "LIB-7"
        assert headers["X-DocuFlow-Event"] == "job.completed"
        assert headers["X-DocuFlow-Delivery"]
        assert job.delivery["callback"]["status"] == "DELIVERED"

    def test_rejected_callback_is_recorded_and_raised_for_retry(self, db, job):
        post = MagicMock(return_value=SimpleNamespace(status_code=500, text="boom"))
        with pytest.raises(delivery.DeliveryError, match="500"):
            delivery.deliver_callback(db, job, "http://df", post=post, attempt=3)

        assert job.delivery["callback"]["status"] == "FAILED"
        assert job.delivery["callback"]["attempts"] == 3
        assert "500" in job.delivery["callback"]["error"]
        assert job.status == "COMPLETED"  # processing outcome untouched

    def test_no_callback_url(self, db, job):
        job.callback_url = None
        db.commit()
        post = MagicMock()
        assert delivery.deliver_callback(db, job, "http://df", post=post, attempt=1) == "SKIPPED"
        post.assert_not_called()


class TestMssql:
    def test_not_configured(self, db, job):
        assert delivery.deliver_mssql(db, job, "http://df", final=True, sink=None) == "SKIPPED"

    def test_final_write_includes_the_result(self, db, job):
        engine = sa.create_engine("sqlite://")
        sink = MssqlSink(engine=engine, schema=None)
        sink.metadata.create_all(engine)

        with patch.object(svc, "_ocr_text", return_value="văn bản"):
            assert delivery.deliver_mssql(db, job, "http://df", final=True, sink=sink) == "DELIVERED"

        with engine.connect() as conn:
            row = conn.execute(sa.select(sink.tables["DocuFlowResult"])).mappings().one()
        assert row["OcrText"] == "văn bản"
        assert job.delivery["mssql"]["status"] == "DELIVERED"

    def test_intermediate_write_is_status_only(self, db, job):
        sink = MagicMock()
        delivery.deliver_mssql(db, job, "http://df", final=False, sink=sink)
        (job_view,), kwargs = sink.write.call_args
        assert job_view["job_id"] == job.id and "result" not in kwargs

    def test_failure_is_recorded_and_raised(self, db, job):
        sink = MagicMock()
        sink.write.side_effect = RuntimeError("Login failed for user 'docuflow_writer'")
        with pytest.raises(RuntimeError):
            delivery.deliver_mssql(db, job, "http://df", final=True, sink=sink, attempt=2)
        assert job.delivery["mssql"]["status"] == "FAILED"
        assert "Login failed" in job.delivery["mssql"]["error"]


def test_parallel_channels_do_not_overwrite_each_others_record(tmp_path):
    """MSSQL and callback are delivered side by side, each in its own session.
    A session holding a stale copy of the job must not wipe the other channel's
    record when it writes its own."""
    from sqlalchemy.orm import sessionmaker

    from data.db_models import Base, IntegrationJob

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'df.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    setup = Session()
    IdGenerator.seed_sequences(setup)
    setup.commit()
    client, _, _ = svc.create_client(setup, "Thư viện", callback_url="http://lib/cb")
    doc = Document(id="DOC_P", user_id=client.service_user_id, title="T", source_language="en",
                   processing_status="EXTRACTED", pipeline_state="IDLE")
    setup.add(doc)
    setup.commit()
    job_id = svc.create_job(setup, client, document_id="DOC_P", external_id=None, tasks=["ocr"],
                            callback_url=None).id
    setup.close()

    a, b = Session(), Session()
    job_a, job_b = a.get(IntegrationJob, job_id), b.get(IntegrationJob, job_id)  # both loaded now
    delivery._record(a, job_a, "mssql", "DELIVERED", 1)
    delivery._record(b, job_b, "callback", "DELIVERED", 1)

    check = Session()
    assert set(check.get(IntegrationJob, job_id).delivery) == {"mssql", "callback"}
