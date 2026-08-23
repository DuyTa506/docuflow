"""Submit-layer admission counts OPEN tasks against the host capacity profile."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.capacity import SLOT_DIGEST, SLOT_EXTRACT, capacity_profile
from data.db_models import Base, Document, Task
from services.pipeline.admission import AdmissionRejected, assert_can_admit, count_open


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _task(db, tid, doc_id, task_type, status="RUNNING"):
    db.add(
        Task(
            id=tid,
            document_id=doc_id,
            task_type=task_type,
            status=status,
            progress=0,
        )
    )


def _doc(db, doc_id, user_id="USR_1"):
    db.add(
        Document(
            id=doc_id,
            user_id=user_id,
            title=doc_id,
            original_filename=f"{doc_id}.pdf",
            format="pdf",
            file_path=f"documents/{doc_id}/original/x.pdf",
            file_type="pdf",
            processing_status="EXTRACTED",
        )
    )


def test_admits_when_under_cap():
    db = _session()
    _doc(db, "DOC_1")
    db.commit()
    assert_can_admit(db, SLOT_DIGEST, user_id="USR_1")


def test_rejects_when_digest_slots_full():
    db = _session()
    cap = capacity_profile()
    _doc(db, "DOC_1")
    for i in range(cap.max_digest_pipelines):
        _task(db, f"TASK_{i}", "DOC_1", "DIGEST_PIPELINE")
    db.commit()
    try:
        assert_can_admit(db, SLOT_DIGEST)
        raise AssertionError("expected AdmissionRejected")
    except AdmissionRejected as exc:
        assert exc.slot == SLOT_DIGEST
        assert exc.current == cap.max_digest_pipelines
        assert exc.as_detail()["error"] == "admission_rejected"
        assert "Máy đang xử lý" in str(exc)
        assert f"{cap.max_digest_pipelines}/{cap.max_digest_pipelines}" in str(exc)


def test_excluding_task_id_does_not_count_the_new_row():
    db = _session()
    cap = capacity_profile()
    _doc(db, "DOC_1")
    for i in range(cap.max_digest_pipelines):
        _task(db, f"TASK_{i}", "DOC_1", "DIGEST_PIPELINE")
    db.commit()
    assert_can_admit(db, SLOT_DIGEST, excluding_task_id="TASK_0")


def test_per_user_cap(monkeypatch):
    from config.capacity import CapacityProfile
    from services.pipeline import admission as admission_mod

    monkeypatch.setattr(
        admission_mod,
        "capacity_profile",
        lambda: CapacityProfile(
            max_digest_pipelines=20,
            max_extractions=20,
            max_translations=20,
            max_jobs_per_user=2,
            digest_group_a_parallelism=2,
            digest_group_b_parallel=True,
            gpu_docling_slots=1,
            gpu_lease_ttl_seconds=90,
            gpu_lease_wait_seconds=5,
        ),
    )
    db = _session()
    _doc(db, "DOC_1", user_id="USR_1")
    _task(db, "TASK_u0", "DOC_1", SLOT_EXTRACT)
    _task(db, "TASK_u1", "DOC_1", SLOT_EXTRACT)
    db.commit()
    try:
        assert_can_admit(db, SLOT_EXTRACT, user_id="USR_1")
        raise AssertionError("expected AdmissionRejected")
    except AdmissionRejected as exc:
        assert "đang chạy tối đa" in str(exc)


def test_count_open_ignores_completed():
    db = _session()
    _doc(db, "DOC_1")
    _task(db, "TASK_done", "DOC_1", "DIGEST_PIPELINE", status="COMPLETED")
    db.commit()
    assert count_open(db, SLOT_DIGEST) == 0


def test_count_open_ignores_queued_waiters():
    from services.pipeline.admission import mark_queued

    db = _session()
    _doc(db, "DOC_1")
    _task(db, "TASK_run", "DOC_1", "DIGEST_PIPELINE", status="RUNNING")
    waiter = Task(
        id="TASK_wait",
        document_id="DOC_1",
        task_type="DIGEST_PIPELINE",
        status="PENDING",
        progress=0,
    )
    db.add(waiter)
    mark_queued(waiter)
    db.commit()
    assert count_open(db, SLOT_DIGEST) == 1
