from unittest.mock import patch, MagicMock


def _user(role="MEMBER", status="ACTIVE"):
    u = MagicMock()
    u.id = "USR_001"
    u.username = "bob"
    u.full_name = "Bob"
    u.email = "bob@example.com"
    u.group = "TEACHER"
    u.role = role
    u.status = status
    u.created_at = None
    return u


class TestRegister:
    def test_success_returns_201(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.register_user.return_value = _user()
            resp = client.post("/api/v2/auth/register", json={
                "username": "bob",
                "password": "pass123",
                "group": "TEACHER",
                "role": "MEMBER",
            })
        assert resp.status_code == 201
        assert resp.json()["username"] == "bob"

    def test_duplicate_username_returns_400(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.register_user.side_effect = ValueError("Username taken")
            resp = client.post("/api/v2/auth/register", json={
                "username": "bob",
                "password": "pass123",
                "group": "TEACHER",
                "role": "MEMBER",
            })
        assert resp.status_code == 400


class TestLogin:
    def test_success_returns_token(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.authenticate.return_value = _user()
            mock_auth.create_access_token.return_value = "jwt.tok.en"
            resp = client.post("/api/v2/auth/login", json={"username": "bob", "password": "pass"})
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "jwt.tok.en"

    def test_invalid_credentials_returns_401(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.authenticate.return_value = None
            resp = client.post("/api/v2/auth/login", json={"username": "bob", "password": "wrong"})
        assert resp.status_code == 401

    def test_inactive_account_returns_403(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.authenticate.return_value = _user(status="PENDING_APPROVAL")
            resp = client.post("/api/v2/auth/login", json={"username": "bob", "password": "pass"})
        assert resp.status_code == 403


class TestMe:
    def test_returns_current_user(self, client, member_user):
        resp = client.get("/api/v2/auth/me")
        assert resp.status_code == 200
        assert resp.json()["id"] == member_user.id


class TestApproveUser:
    def test_success(self, admin_client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.approve_user.return_value = _user()
            resp = admin_client.post("/api/v2/auth/approve/USR_001")
        assert resp.status_code == 200

    def test_user_not_found_returns_404(self, admin_client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.approve_user.side_effect = ValueError("Not found")
            resp = admin_client.post("/api/v2/auth/approve/USR_999")
        assert resp.status_code == 404

    def test_member_cannot_approve_returns_403(self, client):
        resp = client.post("/api/v2/auth/approve/USR_001")
        assert resp.status_code == 403


class TestListUsers:
    def test_success(self, admin_client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.list_users.return_value = [_user()]
            resp = admin_client.get("/api/v2/auth/users")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_member_cannot_list_returns_403(self, client):
        resp = client.get("/api/v2/auth/users")
        assert resp.status_code == 403


class TestUpdateProfile:
    def test_success_returns_updated_user(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.update_profile.return_value = _user()
            resp = client.patch("/api/v2/auth/me", json={"full_name": "New Name"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "bob"

    def test_email_conflict_returns_400(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.update_profile.side_effect = ValueError("Email already in use")
            resp = client.patch("/api/v2/auth/me", json={"email": "taken@example.com"})
        assert resp.status_code == 400

    def test_unauthenticated_returns_401(self):
        from fastapi.testclient import TestClient
        from serving.workflow_api import app
        bare = TestClient(app, raise_server_exceptions=False)
        resp = bare.patch("/api/v2/auth/me", json={"full_name": "X"})
        assert resp.status_code in (401, 403)


class TestChangePassword:
    def test_success_returns_204(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.change_password.return_value = _user()
            resp = client.put("/api/v2/auth/me/password", json={
                "current_password": "secret123",
                "new_password": "newpass456",
            })
        assert resp.status_code == 204

    def test_wrong_current_password_returns_400(self, client):
        with patch("serving.routers.auth_router._auth") as mock_auth:
            mock_auth.change_password.side_effect = ValueError("Incorrect password")
            resp = client.put("/api/v2/auth/me/password", json={
                "current_password": "wrong",
                "new_password": "newpass456",
            })
        assert resp.status_code == 400

    def test_short_new_password_returns_422(self, client):
        resp = client.put("/api/v2/auth/me/password", json={
            "current_password": "secret123",
            "new_password": "ab",
        })
        assert resp.status_code == 422
