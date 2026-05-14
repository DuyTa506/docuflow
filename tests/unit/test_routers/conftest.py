import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_db
from serving.workflow_api import app


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
