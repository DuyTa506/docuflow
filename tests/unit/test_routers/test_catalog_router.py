"""The CTĐT programme list is data the Academy supplies, not data we ship.

Deployments that have it need a way to load it in; deployments that don't
must still be able to see that §3 is empty *because nothing was loaded*.
"""

from unittest.mock import patch

import pytest

from config.settings import settings

ONE_GROUP = {
    "code": "78602",
    "name": "Quân sự",
    "children": [{"code": "7860201", "name": "Chỉ huy tham mưu Lục quân"}],
}


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path, monkeypatch):
    """Never let a test write over a real deployment's catalog."""
    monkeypatch.setattr(settings, "ctdt_catalog_path", str(tmp_path / "ctdt_catalog.json"))


class TestRead:
    def test_any_user_can_read_the_catalog(self, client):
        resp = client.get("/api/v2/catalog/ctdt")

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "bundled"
        assert body["counts"]["undergraduate"] > 0, "counts đếm ngành, không đếm nhóm ngành"
        codes = {
            child["code"]
            for group in body["catalog"]["undergraduate"]
            for child in group["children"]
        }
        assert "7480201" in codes, "Công nghệ thông tin"

    def test_reports_when_nothing_is_loaded(self, client):
        with patch("serving.routers.catalog_router.catalog_source", return_value="none"):
            with patch("serving.routers.catalog_router.load_catalog", return_value={}):
                body = client.get("/api/v2/catalog/ctdt").json()

        assert body["source"] == "none"
        assert body["has_entries"] is False


class TestUpload:
    def test_admin_upload_replaces_the_catalog(self, admin_client):
        payload = {
            "undergraduate": [ONE_GROUP],
            "_source": "Phòng Đào tạo 2026",
        }

        resp = admin_client.put("/api/v2/catalog/ctdt", json=payload)

        assert resp.status_code == 200
        assert resp.json()["source"] == "uploaded"
        body = admin_client.get("/api/v2/catalog/ctdt").json()
        assert body["counts"]["undergraduate"] == 1
        assert body["catalog"]["undergraduate"][0]["children"][0]["code"] == "7860201"

    def test_member_cannot_upload(self, client):
        resp = client.put("/api/v2/catalog/ctdt", json={"phd": [ONE_GROUP]})

        assert resp.status_code == 403

    def test_malformed_catalog_is_rejected_with_a_reason(self, admin_client):
        resp = admin_client.put("/api/v2/catalog/ctdt", json={"undergraduate": "Ngành X"})

        assert resp.status_code == 400
        assert "undergraduate" in resp.json()["detail"]

    def test_flat_name_list_is_rejected(self, admin_client):
        """Định dạng cũ không còn định danh được ngành — phải báo lỗi, không nhận im."""
        resp = admin_client.put(
            "/api/v2/catalog/ctdt", json={"undergraduate": ["Ngành Khoa học máy tính"]}
        )

        assert resp.status_code == 400

    def test_delete_reverts_to_the_bundled_catalog(self, admin_client):
        admin_client.put("/api/v2/catalog/ctdt", json={"phd": [ONE_GROUP]})

        resp = admin_client.delete("/api/v2/catalog/ctdt")

        assert resp.status_code == 200
        assert resp.json()["source"] == "bundled"

    def test_delete_without_an_upload_is_not_an_error(self, admin_client):
        assert admin_client.delete("/api/v2/catalog/ctdt").status_code == 200
