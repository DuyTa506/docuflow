"""submit_extraction_async routes OCR through Temporal when ocr_use_temporal
is on; and _run_extraction(resume=True) must NOT wipe existing extraction
artifacts — kept pages are the resume checkpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, Task
from services.document_service import DocumentService


@pytest.fixture
def db():
    from data.id_generator import IdGenerator

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(session)
    session.add(Document(id="DOC_X", title="t", original_filename="t.pdf", total_pages=3))
    session.commit()
    yield session
    session.close()


@pytest.mark.asyncio
async def test_submit_extraction_async_queues_and_kicks(db):
    svc = DocumentService()
    with (
        patch("services.document_service.settings") as mock_settings,
        patch("services.document_service.task_manager") as mock_tm,
        patch("services.pipeline.job_queue.kick_queue") as mock_kick,
        patch(
            "services.pipeline.temporal_client.start_extraction_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
    ):
        mock_settings.ocr_use_temporal = True
        task_id, reused = await svc.submit_extraction_async(db, "DOC_X")

    mock_tm.submit.assert_not_called()
    mock_start.assert_not_awaited()
    mock_kick.assert_called_once()
    from services.pipeline.admission import is_queued

    task = db.query(Task).filter(Task.id == task_id).first()
    assert task is not None and task.task_type == "EXTRACT"
    assert is_queued(task)
    assert reused is False


@pytest.mark.asyncio
async def test_submit_extraction_async_dedupes_active_task(db):
    db.add(Task(id="EXTRACT_900", document_id="DOC_X", task_type="EXTRACT", status="RUNNING"))
    db.commit()
    svc = DocumentService()
    with (
        patch("services.document_service.settings") as mock_settings,
        patch("services.pipeline.job_queue.kick_queue") as mock_kick,
        patch(
            "services.pipeline.temporal_client.start_extraction_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
    ):
        mock_settings.ocr_use_temporal = True
        task_id, reused = await svc.submit_extraction_async(db, "DOC_X")

    assert task_id == "EXTRACT_900" and reused is True
    mock_start.assert_not_awaited()
    mock_kick.assert_not_called()


@pytest.mark.asyncio
async def test_submit_extraction_when_full_queues_pending_task(db, monkeypatch):
    from config.capacity import CapacityProfile
    from services.pipeline import admission as admission_mod
    from services.pipeline.admission import is_queued

    monkeypatch.setattr(
        admission_mod,
        "capacity_profile",
        lambda: CapacityProfile(
            max_digest_pipelines=20,
            max_extractions=1,
            max_translations=20,
            max_jobs_per_user=20,
            digest_group_a_parallelism=2,
            digest_group_b_parallel=True,
            gpu_docling_slots=1,
            gpu_lease_ttl_seconds=90,
            gpu_lease_wait_seconds=5,
        ),
    )
    db.add(Document(id="DOC_OTHER", title="o", original_filename="o.pdf", total_pages=1))
    db.add(
        Task(
            id="EXTRACT_OLD",
            document_id="DOC_OTHER",
            task_type="EXTRACT",
            status="RUNNING",
            progress=10,
        )
    )
    db.commit()
    svc = DocumentService()
    with (
        patch("services.document_service.settings") as mock_settings,
        patch("services.pipeline.job_queue.kick_queue") as mock_kick,
        patch(
            "services.pipeline.temporal_client.start_extraction_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
    ):
        mock_settings.ocr_use_temporal = True
        task_id, reused = await svc.submit_extraction_async(db, "DOC_X")

    assert reused is False
    mock_start.assert_not_awaited()
    mock_kick.assert_called_once()
    task = db.query(Task).filter(Task.id == task_id).first()
    assert task is not None and task.status == "PENDING"
    assert is_queued(task)
    open_ids = {
        t.id
        for t in db.query(Task).filter(
            Task.task_type == "EXTRACT", Task.status.in_(["PENDING", "RUNNING"])
        )
    }
    assert open_ids == {"EXTRACT_OLD", task_id}


@pytest.mark.asyncio
async def test_resume_keeps_extraction_artifacts():
    """resume=True must skip clear_extraction_artifacts — stored pages are
    the checkpoints a retry resumes from."""
    svc = DocumentService()

    fake_doc = MagicMock()
    fake_doc.file_path = "/nonexistent/file.pdf"
    fake_doc.format = "pdf"
    fake_doc.total_pages = 3

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.first.return_value = fake_doc

    with (
        patch("services.document_service.get_db_manager") as mock_dbm,
        patch("data.repositories.DocumentRepository") as mock_repo_cls,
        patch("services.export_service.export_service"),
        patch("services.object_storage.get_object_storage") as mock_storage,
        patch.object(svc, "_run_extraction_body", new_callable=AsyncMock) as mock_body,
    ):
        mock_dbm.return_value.session.return_value = session
        mock_storage.return_value.resolve_local_or_key.return_value = "/nonexistent/file.pdf"

        await svc._run_extraction("DOC_X", task_id=None, resume=True)
        mock_repo_cls.return_value.clear_extraction_artifacts.assert_not_called()
        assert mock_body.await_args.kwargs["resume"] is True

        await svc._run_extraction("DOC_X", task_id=None, resume=False)
        mock_repo_cls.return_value.clear_extraction_artifacts.assert_called_once()
        assert mock_body.await_args.kwargs["resume"] is False
