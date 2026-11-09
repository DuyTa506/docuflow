"""Findings a dynamic security scan raises against the exposed API.

Each class pins one: response headers, cross-origin policy, what anonymous
health probes reveal, brute-forcing the login, and webhook targets.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_db
from serving.workflow_api import app

pytest_plugins = ["tests.unit.test_routers.conftest"]


@pytest.fixture
def anon(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSecurityHeaders:
    def test_every_response_carries_the_baseline_headers(self, anon):
        resp = anon.get("/health/live")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert "default-src 'self'" in resp.headers["content-security-policy"]

    def test_the_angular_stylesheet_handler_is_allowed_by_hash_only(self, anon):
        import base64
        import hashlib

        csp = anon.get("/health/live").headers["content-security-policy"]
        digest = base64.b64encode(hashlib.sha256(b"this.media='all'").digest()).decode()
        assert f"script-src-attr 'unsafe-hashes' 'sha256-{digest}'" in csp
        assert "'unsafe-inline'" not in csp.split("style-src")[0]

    def test_api_responses_are_not_cacheable(self, anon):
        resp = anon.get("/api/v2/auth/me")
        assert resp.headers["cache-control"] == "no-store"

    def test_hsts_only_over_https(self, anon):
        assert "strict-transport-security" not in anon.get("/health/live").headers
        https = TestClient(app, base_url="https://testserver")
        assert "max-age=" in https.get("/health/live").headers["strict-transport-security"]


class TestCrossOrigin:
    def test_an_unlisted_origin_gets_no_cors_grant(self, anon):
        resp = anon.get("/health/live", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in resp.headers

    def test_a_preflight_from_an_unlisted_origin_is_refused(self, anon):
        resp = anon.options(
            "/api/v2/auth/login",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in resp.headers


class TestAnonymousHealthProbes:
    def test_capacity_needs_an_admin(self, anon):
        assert anon.get("/health/capacity").status_code == 401

    def test_capacity_is_refused_to_a_member(self, mock_db, member_user):
        app.dependency_overrides[get_current_user] = lambda: member_user
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            assert TestClient(app).get("/health/capacity").status_code == 403
        finally:
            app.dependency_overrides.clear()

    def test_ready_does_not_leak_backend_errors(self, anon):
        with (
            patch(
                "serving.health._check_postgres",
                return_value={"ok": False, "error": "connection to 10.0.0.5:5433 refused"},
            ),
            patch("serving.health._check_minio", return_value={"ok": True}),
            patch("serving.health._check_temporal", return_value={"ok": True}),
        ):
            resp = anon.get("/health/ready")
        assert resp.status_code == 503
        assert "10.0.0.5" not in resp.text
        assert "refused" not in resp.text


class TestLoginBruteForce:
    @pytest.fixture(autouse=True)
    def _fresh_limiter(self):
        from serving import security

        security.login_limiter.reset()
        yield
        security.login_limiter.reset()

    def _post(self, client, user="bob", pw="wrong-password"):
        return client.post("/api/v2/auth/login", json={"username": user, "password": pw})

    def test_repeated_failures_lock_the_account_out(self, mock_db):
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            with patch("serving.routers.auth_router._auth") as auth:
                auth.authenticate.return_value = None
                client = TestClient(app)
                codes = [self._post(client).status_code for _ in range(6)]
                assert codes[:5] == [401] * 5
                assert codes[5] == 429
                # The lock holds even for the correct password.
                auth.authenticate.return_value = MagicMock(status="ACTIVE", id="U", role="MEMBER", group="TEACHER")
                locked = self._post(client, pw="right-password")
                assert locked.status_code == 429
                assert int(locked.headers["retry-after"]) > 0
        finally:
            app.dependency_overrides.clear()

    def test_a_success_clears_the_failures(self, mock_db):
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            with patch("serving.routers.auth_router._auth") as auth:
                client = TestClient(app)
                auth.authenticate.return_value = None
                for _ in range(3):
                    self._post(client)
                auth.authenticate.return_value = MagicMock(status="ACTIVE", id="U", role="MEMBER", group="TEACHER")
                auth.create_access_token.return_value = "tok"
                assert self._post(client, pw="right").status_code == 200
                auth.authenticate.return_value = None
                assert [self._post(client).status_code for _ in range(5)] == [401] * 5
        finally:
            app.dependency_overrides.clear()

    def test_unknown_and_known_users_cost_the_same_hash_work(self, mock_db):
        """No bcrypt on a missing user answers measurably faster than on a real one."""
        from services.auth_service import AuthService

        mock_db.query.return_value.filter.return_value.first.return_value = None
        with patch.object(AuthService, "verify_password", return_value=False) as verify:
            assert AuthService().authenticate(mock_db, "ghost", "whatever") is None
        assert verify.called


class TestPasswordPolicy:
    def test_registration_needs_eight_characters(self, anon):
        resp = anon.post("/api/v2/auth/register", json={"username": "newbie", "password": "abc123"})
        assert resp.status_code == 422

    def test_new_password_needs_eight_characters(self, client):
        resp = client.put(
            "/api/v2/auth/me/password",
            json={"current_password": "whatever1", "new_password": "abc123"},
        )
        assert resp.status_code == 422


class TestWebhookTargets:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:5011/v1/completions",
            "http://localhost:9000/minio",
            "http://169.254.169.254/latest/meta-data",
            "http://0.0.0.0:8022/",
            "http://[::1]:8022/",
            "file:///etc/passwd",
            "gopher://example.com/",
            "http://user:pw@example.com/",
        ],
    )
    def test_internal_or_odd_targets_are_refused(self, url):
        from utils.url_safety import UnsafeUrl, validate_callback_url

        with pytest.raises(UnsafeUrl):
            validate_callback_url(url)

    def test_a_lan_library_system_is_allowed(self):
        from utils.url_safety import validate_callback_url

        with patch("utils.url_safety._resolve", return_value=["203.0.113.7"]):
            url = "https://library.example.org/hook"
            assert validate_callback_url(url) == url
        assert validate_callback_url("http://192.168.1.20:9000/hook")

    def test_a_name_that_resolves_to_loopback_is_refused(self):
        from utils.url_safety import UnsafeUrl, validate_callback_url

        with patch("utils.url_safety._resolve", return_value=["127.0.0.1"]):
            with pytest.raises(UnsafeUrl):
                validate_callback_url("http://rebind.example.org/hook")

    def test_submitting_a_job_with_a_loopback_callback_is_a_400(self, mock_db):
        from api.dependencies import get_db
        from serving.routers.integration_router import get_integration_client

        client_obj = MagicMock(id="ICL_1", service_user_id="USR_1")
        app.dependency_overrides[get_integration_client] = lambda: client_obj
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            resp = TestClient(app).post(
                "/api/v2/integration/jobs",
                files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                data={"callback_url": "http://127.0.0.1:5011/"},
            )
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 400
        assert "callback" in resp.text.lower()

    def test_delivery_refuses_a_stored_unsafe_target(self):
        from services import integration_delivery as d

        job = MagicMock(callback_url="http://127.0.0.1:5011/", status="COMPLETED")
        post = MagicMock()
        with pytest.raises(d.DeliveryError):
            d.deliver_callback(MagicMock(), job, "http://x", post=post)
        post.assert_not_called()


class TestDocumentAccessDoesNotRevealExistence:
    def test_someone_elses_document_looks_like_a_missing_one(self):
        from fastapi import HTTPException

        from api.dependencies import get_authorized_document

        doc = MagicMock(user_id="USR_OWNER")
        user = MagicMock(id="USR_ME", role="MEMBER")
        with patch("data.repositories.DocumentRepository") as repo:
            repo.return_value.get.return_value = doc
            with pytest.raises(HTTPException) as exc:
                get_authorized_document("DOC_1", user, MagicMock())
            repo.return_value.get.return_value = None
            with pytest.raises(HTTPException) as missing:
                get_authorized_document("DOC_1", user, MagicMock())
        assert exc.value.status_code == missing.value.status_code == 404
        assert exc.value.detail == missing.value.detail


class TestTaskWithoutADocument:
    def test_a_member_cannot_read_an_ownerless_task(self, client):
        status = {"task_id": "T1", "document_id": None, "status": "COMPLETED", "result": {"k": "v"}}
        with patch("serving.routers.tasks_router.task_manager") as tm:
            tm.get_status.return_value = status
            resp = client.get("/api/v2/tasks/T1")
        assert resp.status_code == 404


class TestEventStreamUsesTheCurrentAccount:
    def _token(self, role="ADMIN"):
        from services.auth_service import AuthService

        return AuthService.create_access_token({"sub": "USR_9", "role": role})

    def _db_user(self, status="ACTIVE", role="MEMBER"):
        return MagicMock(id="USR_9", status=status, role=role)

    def test_a_deactivated_account_gets_no_stream(self, anon, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = self._db_user("DEACTIVATED")
        resp = anon.get("/api/v2/events", params={"token": self._token()})
        assert resp.status_code in (401, 403)

    def test_a_deleted_account_gets_no_stream(self, anon, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        assert anon.get("/api/v2/events", params={"token": self._token()}).status_code == 401

    def test_the_role_comes_from_the_database_not_the_token(self, anon, mock_db):
        import importlib

        events_router = importlib.import_module("serving.routers.events_router")

        mock_db.query.return_value.filter.return_value.first.return_value = self._db_user(role="MEMBER")
        user = events_router._authenticate(self._token(role="ADMIN"), mock_db)
        assert user.role == "MEMBER"


class TestDeletingAUserChecksBeforeStoppingWork:
    def test_a_refused_deletion_does_not_terminate_workflows(self, admin_client, admin_user):
        with (
            patch("serving.routers.auth_router._auth") as auth,
            patch(
                "services.pipeline.temporal_client.terminate_document_workflows"
            ) as terminate,
        ):
            auth.delete_user.side_effect = ValueError("Cannot delete your own account")
            resp = admin_client.delete(f"/api/v2/auth/users/{admin_user.id}")
        assert resp.status_code == 400
        terminate.assert_not_called()
