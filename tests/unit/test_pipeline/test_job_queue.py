"""When the host is full, submit queues PENDING; dispatch starts the oldest waiter.

Also covers head-of-line skip (per-user cap), Temporal start failure requeue,
and LONG stage types on the digest slot.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.capacity import SLOT_DIGEST, SLOT_TRANSLATE, CapacityProfile
from data.db_models import Base, Document, Task, Translation
from services.pipeline.admission import is_queued, mark_queued
from services.pipeline.job_queue import dispatch_waiting


@pytest.fixture
def db():
    from data.id_generator import IdGenerator

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(session)
    session.add(
        Document(
            id="DOC_Q",
            title="t",
            original_filename="t.pdf",
            total_pages=1,
            source_language="en",
            user_id="USR_A",
        )
    )
    session.commit()
    yield session
    session.close()


def _mgr(db):
    @contextmanager
    def _session():
        yield db

    class _Mgr:
        def session(self):
            return _session()

    return _Mgr()


@pytest.mark.asyncio
async def test_dispatch_starts_oldest_queued_translation(db):
    trans = Translation(id="TRN_Q", document_id="DOC_Q", target_language="vi", status="PENDING")
    db.add(trans)
    task = Task(
        id="TRANSLATE_Q",
        document_id="DOC_Q",
        task_type="TRANSLATE",
        status="PENDING",
        progress=0,
    )
    mark_queued(
        task,
        extra={"translation_id": "TRN_Q", "target_language": "vi", "domain": "general"},
    )
    db.add(task)
    db.commit()

    with (
        patch("data.database.get_db_manager", return_value=_mgr(db)),
        patch(
            "services.pipeline.temporal_client.start_translation_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
    ):
        await dispatch_waiting(SLOT_TRANSLATE)

    mock_start.assert_awaited_once()
    kwargs = mock_start.await_args.kwargs
    assert kwargs["document_id"] == "DOC_Q"
    assert kwargs["translation_id"] == "TRN_Q"
    assert kwargs["parent_task_id"] == "TRANSLATE_Q"
    db.refresh(task)
    assert not is_queued(task)


@pytest.mark.asyncio
async def test_dispatch_skips_user_at_cap_and_starts_other(db, monkeypatch):
    from services.pipeline import admission as admission_mod
    from services.pipeline import job_queue as jq

    monkeypatch.setattr(
        admission_mod,
        "capacity_profile",
        lambda: CapacityProfile(
            max_digest_pipelines=20,
            max_extractions=20,
            max_translations=2,
            max_jobs_per_user=1,
            digest_group_a_parallelism=2,
            digest_group_b_parallel=True,
            docling_slots=1,
            gpu_lease_ttl_seconds=90,
            gpu_lease_wait_seconds=5,
        ),
    )
    monkeypatch.setattr(jq, "capacity_profile", admission_mod.capacity_profile)

    db.add(
        Document(
            id="DOC_B",
            title="b",
            original_filename="b.pdf",
            total_pages=1,
            source_language="en",
            user_id="USR_B",
        )
    )
    # USR_A already running one job → at per-user cap
    db.add(
        Task(
            id="TRANSLATE_RUN",
            document_id="DOC_Q",
            task_type="TRANSLATE",
            status="RUNNING",
            progress=10,
        )
    )
    db.add(Translation(id="TRN_A", document_id="DOC_Q", target_language="vi", status="PENDING"))
    db.add(Translation(id="TRN_B", document_id="DOC_B", target_language="vi", status="PENDING"))
    t_a = Task(id="TRANSLATE_A", document_id="DOC_Q", task_type="TRANSLATE", status="PENDING")
    t_b = Task(id="TRANSLATE_B", document_id="DOC_B", task_type="TRANSLATE", status="PENDING")
    mark_queued(
        t_a,
        extra={
            "fairness_key": "USR_A",
            "translation_id": "TRN_A",
            "target_language": "vi",
        },
    )
    mark_queued(
        t_b,
        extra={
            "fairness_key": "USR_B",
            "translation_id": "TRN_B",
            "target_language": "vi",
        },
    )
    db.add_all([t_a, t_b])
    db.commit()

    with (
        patch("data.database.get_db_manager", return_value=_mgr(db)),
        patch(
            "services.pipeline.temporal_client.start_translation_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
    ):
        await dispatch_waiting(SLOT_TRANSLATE)

    mock_start.assert_awaited_once()
    assert mock_start.await_args.kwargs["parent_task_id"] == "TRANSLATE_B"
    db.refresh(t_a)
    db.refresh(t_b)
    assert is_queued(t_a)
    assert not is_queued(t_b)


@pytest.mark.asyncio
async def test_dispatch_requeues_when_start_fails(db):
    db.add(Translation(id="TRN_Q", document_id="DOC_Q", target_language="vi", status="PENDING"))
    task = Task(id="TRANSLATE_Q", document_id="DOC_Q", task_type="TRANSLATE", status="PENDING")
    mark_queued(
        task,
        extra={"translation_id": "TRN_Q", "target_language": "vi", "fairness_key": "USR_A"},
    )
    db.add(task)
    db.commit()

    with (
        patch("data.database.get_db_manager", return_value=_mgr(db)),
        patch(
            "services.pipeline.temporal_client.start_translation_workflow",
            new_callable=AsyncMock,
            side_effect=RuntimeError("temporal down"),
        ),
    ):
        await dispatch_waiting(SLOT_TRANSLATE)

    db.refresh(task)
    assert task.status == "PENDING"
    assert is_queued(task)


@pytest.mark.asyncio
async def test_dispatch_starts_queued_long_stage(db):
    task = Task(
        id="HIERARCHICAL_SUMMARIZE_1",
        document_id="DOC_Q",
        task_type="HIERARCHICAL_SUMMARIZE",
        status="PENDING",
    )
    mark_queued(task, extra={"fairness_key": "USR_A"})
    db.add(task)
    db.commit()

    with (
        patch("data.database.get_db_manager", return_value=_mgr(db)),
        patch(
            "services.pipeline.temporal_client.start_stage_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
        patch(
            "services.pipeline.temporal_client.start_digest_workflow",
            new_callable=AsyncMock,
        ) as mock_digest,
    ):
        await dispatch_waiting(SLOT_DIGEST)

    mock_digest.assert_not_awaited()
    mock_start.assert_awaited_once()
    assert mock_start.await_args.kwargs["stage"] == "HIERARCHICAL_SUMMARIZE"
    assert mock_start.await_args.kwargs["task_id"] == "HIERARCHICAL_SUMMARIZE_1"
