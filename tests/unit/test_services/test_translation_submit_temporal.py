"""submit_async routes to Temporal when translation_use_temporal is on:
no in-process task_manager coroutine, a TRANSLATE Task row is created for
UI polling, and the workflow is started with the right identifiers.
A COMPLETED prior translation being explicitly re-run drops its MinIO resume
state; a FAILED one keeps it (that's the resume path).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, Task, Translation
from services.translation_service import TranslationService


@pytest.fixture
def db():
    from data.id_generator import IdGenerator

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(session)
    session.add(
        Document(
            id="DOC_S",
            title="t",
            original_filename="t.pdf",
            total_pages=1,
            source_language="en",
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.mark.asyncio
async def test_submit_async_starts_workflow_not_task_manager(db):
    svc = TranslationService()

    with (
        patch("services.translation_service.settings") as mock_settings,
        patch(
            "services.pipeline.temporal_client.start_translation_workflow",
            new_callable=AsyncMock,
        ) as mock_start,
        patch("services.translation_service.task_manager") as mock_tm,
    ):
        mock_settings.translation_use_temporal = True
        task_id, translation_id, reused = await svc.submit_async(db, "DOC_S", "vi", "military")

    mock_tm.submit.assert_not_called()
    mock_start.assert_awaited_once()
    kwargs = mock_start.await_args.kwargs
    assert kwargs["document_id"] == "DOC_S"
    assert kwargs["translation_id"] == translation_id
    assert kwargs["parent_task_id"] == task_id
    assert kwargs["target_language"] == "vi"
    assert kwargs["domain"] == "military"

    task = db.query(Task).filter(Task.id == task_id).first()
    assert task is not None and task.task_type == "TRANSLATE"
    trans = db.query(Translation).filter(Translation.id == translation_id).first()
    assert trans is not None and trans.status == "PENDING"
    assert reused is False


@pytest.mark.asyncio
async def test_submit_async_flag_off_delegates_to_legacy_submit(db):
    svc = TranslationService()
    with (
        patch("services.translation_service.settings") as mock_settings,
        patch.object(svc, "submit", return_value=("T_1", "TRN_1", False)) as mock_submit,
    ):
        mock_settings.translation_use_temporal = False
        out = await svc.submit_async(db, "DOC_S", "vi")
    mock_submit.assert_called_once()
    assert out == ("T_1", "TRN_1", False)


@pytest.mark.asyncio
async def test_retranslating_completed_drops_resume_state_failed_keeps_it(db):
    svc = TranslationService()

    async def run_once(status):
        existing = (
            db.query(Translation)
            .filter(Translation.document_id == "DOC_S", Translation.target_language == "vi")
            .first()
        )
        if existing:
            existing.status = status
        else:
            db.add(
                Translation(id="TRN_R", document_id="DOC_S", target_language="vi", status=status)
            )
        db.commit()
        storage = MagicMock()
        with (
            patch("services.translation_service.settings") as mock_settings,
            patch(
                "services.pipeline.temporal_client.start_translation_workflow",
                new_callable=AsyncMock,
            ),
            patch("services.object_storage.get_object_storage", return_value=storage),
            patch("services.export_service.export_service") as mock_export,
        ):
            mock_export.storage = MagicMock()
            mock_settings.translation_use_temporal = True
            await svc.submit_async(db, "DOC_S", "vi")
        return storage

    storage = await run_once("FAILED")
    storage.delete_prefix.assert_not_called()

    storage = await run_once("COMPLETED")
    storage.delete_prefix.assert_called_once()
    prefix = storage.delete_prefix.call_args.args[0]
    assert prefix.startswith("documents/DOC_S/translations/TRN_R")
