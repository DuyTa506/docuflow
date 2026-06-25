import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_db
from serving.workflow_api import app

# Routers import get_authorized_document by name — patch each module separately.
_AUTHORIZED_DOCUMENT_PATCHES = [
    "serving.routers.documents_router.get_authorized_document",
    "serving.routers.translation_router.get_authorized_document",
    "serving.routers.summarization_router.get_authorized_document",
    "serving.routers.keywords_router.get_authorized_document",
    "serving.routers.research_router.get_authorized_document",
    "serving.routers.main_content_router.get_authorized_document",
    "serving.routers.tree_index_router.get_authorized_document",
    "serving.routers.digest_router.get_authorized_document",
    "serving.routers.tasks_router.get_authorized_document",
]


def _make_auth_doc(document_id: str, user_id: str = "USR_001"):
    d = MagicMock()
    d.id = document_id
    d.title = "Test Doc"
    d.original_filename = "test.pdf"
    d.format = "pdf"
    d.file_type = "pdf"
    d.total_pages = 5
    d.processing_status = "EXTRACTED"
    d.source_language = "en"
    d.user_id = user_id
    d.file_path = None
    d.created_at = None
    d.updated_at = None
    return d


def _authorized_document_side_effect(document_id, user, db):
    if document_id == "DOC_999":
        raise HTTPException(status_code=404, detail="Document not found")
    return _make_auth_doc(document_id, user.id)


def _make_user(role="MEMBER", group="TEACHER", uid="USR_001"):
    u = MagicMock()
    u.id = uid
    u.username = "testuser"
    u.full_name = "Test User"
    u.email = "test@example.com"
    u.group = group
    u.role = role
    u.status = "ACTIVE"
    u.created_at = None
    return u


@pytest.fixture
def member_user():
    return _make_user()


@pytest.fixture
def admin_user():
    return _make_user(role="ADMIN", uid="USR_002")


@pytest.fixture
def mock_db():
    db = MagicMock()
    # Default: task query returns empty list (used by list_documents)
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    return db


@pytest.fixture(autouse=True)
def _authorized_document(request):
    """Bypass real DB ownership checks in router unit tests."""
    if request.node.fspath.basename == "test_authorization.py":
        yield
        return
    with ExitStack() as stack:
        for target in _AUTHORIZED_DOCUMENT_PATCHES:
            stack.enter_context(
                patch(target, side_effect=_authorized_document_side_effect)
            )
        yield


@pytest.fixture
def client(mock_db, member_user):
    app.dependency_overrides[get_current_user] = lambda: member_user
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()


@pytest.fixture
def admin_client(mock_db, admin_user):
    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()
