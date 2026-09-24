"""A unit test must never reach the developer's real Temporal server.

`TestDeleteDocument::test_success_returns_204` patched `delete_document_cascade`
but not `terminate_document_workflows`. On a machine with the stack up, a plain
`pytest` run therefore connected to localhost:7233 and terminated whatever was
running under `extraction-DOC_001` — reason "Document deleted", on a document
nobody had deleted. It killed a 761-page extraction 19 minutes in, and because
the cascade *was* patched the document row survived, leaving the DB claiming
EXTRACT_IN_PROGRESS against a workflow Temporal had already closed.

`terminate_document_workflows` is best-effort by design and swallows the
failure, so nothing goes red on its own — which is exactly why the leak lived
this long. These tests watch the socket instead of the return value.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


@pytest.fixture
def connect_spy():
    """Spy on the one call that opens a real connection."""
    spy = AsyncMock(side_effect=AssertionError("a test opened a real Temporal connection"))
    with patch("temporalio.client.Client.connect", spy):
        yield spy


class TestTheGuardIsInPlace:
    @pytest.mark.asyncio
    async def test_terminating_a_workflow_from_a_test_is_refused(self, connect_spy):
        """The conftest guard replaces the client factory, so the refusal
        happens before any socket is opened."""
        from services.pipeline.temporal_client import terminate_running_extraction

        with pytest.raises(RuntimeError, match="real Temporal server"):
            await terminate_running_extraction("DOC_001")

        connect_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_best_effort_wrapper_still_opens_no_connection(self, connect_spy):
        """`terminate_document_workflows` swallows the refusal — that is its
        job. What must not happen is the connection itself."""
        from services.pipeline import temporal_client as tc

        session = MagicMock()
        session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []
        manager = MagicMock()
        manager.session.return_value.__enter__ = MagicMock(return_value=session)
        manager.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(tc, "get_db_manager", return_value=manager):
            await tc.terminate_document_workflows("DOC_001")

        connect_spy.assert_not_awaited()


class TestTheDeleteRouteThatLeaked:
    def test_delete_opens_no_connection_even_unpatched(self, connect_spy):
        """The exact shape of the leaking test: cascade patched, terminate not."""
        from fastapi.testclient import TestClient

        from serving.workflow_api import app

        mock_doc = MagicMock(id="DOC_001", user_id="USR_001")
        with (
            patch("serving.routers.documents_router.DocumentRepository"),
            patch("serving.routers.documents_router.delete_document_cascade", return_value=True),
            patch("serving.routers.documents_router.export_service"),
            patch(
                "serving.routers.documents_router.get_authorized_document",
                return_value=mock_doc,
            ),
            patch("api.dependencies.get_current_user", return_value=mock_doc),
        ):
            client = TestClient(app)
            try:
                client.delete("/api/v2/documents/DOC_001")
            except HTTPException:
                pass

        connect_spy.assert_not_awaited()
