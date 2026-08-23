"""Regression: start_digest_workflow must not pay for the describe()+terminate()
Temporal RPC round-trip unless a prior run is actually known-running. The
unconditional double-RPC was slow enough that the FE's polling could hit a
transient error before start_workflow even returned, surfacing a false
"task failed" toast even though the workflow started successfully."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_session_with_doc(pipeline_state):
    doc = MagicMock()
    doc.pipeline_state = pipeline_state
    db = MagicMock()
    filt = db.query.return_value.filter.return_value
    filt.first.return_value = doc
    filt.all.return_value = []
    session_cm = MagicMock()
    session_cm.__enter__.return_value = db
    session_cm.__exit__.return_value = False
    db_manager = MagicMock()
    db_manager.session.return_value = session_cm
    return db_manager


@pytest.mark.asyncio
async def test_skips_terminate_when_no_prior_run_running():
    from services.pipeline import temporal_client

    db_manager = _mock_session_with_doc(pipeline_state="IDLE")

    with (
        patch.object(temporal_client, "get_db_manager", return_value=db_manager),
        patch.object(temporal_client, "create_parent_task", return_value="TASK_1"),
        patch.object(
            temporal_client, "terminate_running_digest", new=AsyncMock()
        ) as mock_terminate,
        patch.object(temporal_client, "init_pipeline_run"),
        patch.object(temporal_client, "get_temporal_client", new=AsyncMock()) as mock_get_client,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await temporal_client.start_digest_workflow("DOC_TEST")

    mock_terminate.assert_not_called()
    mock_client.start_workflow.assert_awaited_once()


@pytest.mark.asyncio
async def test_calls_terminate_when_prior_run_is_running():
    from services.pipeline import temporal_client

    db_manager = _mock_session_with_doc(pipeline_state="RUNNING")

    with (
        patch.object(temporal_client, "get_db_manager", return_value=db_manager),
        patch.object(temporal_client, "create_parent_task", return_value="TASK_1"),
        patch.object(
            temporal_client, "terminate_running_digest", new=AsyncMock()
        ) as mock_terminate,
        patch.object(temporal_client, "init_pipeline_run"),
        patch.object(temporal_client, "get_temporal_client", new=AsyncMock()) as mock_get_client,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await temporal_client.start_digest_workflow("DOC_TEST")

    mock_terminate.assert_awaited_once_with("DOC_TEST")
    mock_client.start_workflow.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_document_row_skips_terminate():
    from services.pipeline import temporal_client

    db_manager = _mock_session_with_doc(pipeline_state="RUNNING")
    db_manager.session.return_value.__enter__.return_value.query.return_value.filter.return_value.first.return_value = (
        None
    )

    with (
        patch.object(temporal_client, "get_db_manager", return_value=db_manager),
        patch.object(temporal_client, "create_parent_task", return_value="TASK_1"),
        patch.object(
            temporal_client, "terminate_running_digest", new=AsyncMock()
        ) as mock_terminate,
        patch.object(temporal_client, "init_pipeline_run"),
        patch.object(temporal_client, "get_temporal_client", new=AsyncMock()) as mock_get_client,
    ):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await temporal_client.start_digest_workflow("DOC_TEST")

    mock_terminate.assert_not_called()
