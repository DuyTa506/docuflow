from unittest.mock import patch


class TestSearch:
    def test_success_returns_document_list_envelope(self, client):
        mock_result = {"items": [], "total": 0, "query": "machine learning"}
        with patch("serving.routers.search_router._svc") as mock_svc:
            mock_svc.search.return_value = mock_result
            resp = client.get("/api/v2/search?q=machine+learning")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "machine learning"
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1
        assert data["limit"] == 20

    def test_pagination_fields_calculated(self, client):
        mock_result = {"items": [], "total": 23, "query": "test"}
        with patch("serving.routers.search_router._svc") as mock_svc:
            mock_svc.search.return_value = mock_result
            resp = client.get("/api/v2/search?q=test&page=2&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 23
        assert data["page"] == 2
        assert data["limit"] == 5
        assert data["total_pages"] == 5  # ceil(23/5)

    def test_with_all_params(self, client):
        mock_result = {"items": [], "total": 0, "query": "deep learning"}
        with patch("serving.routers.search_router._svc") as mock_svc:
            mock_svc.search.return_value = mock_result
            resp = client.get(
                "/api/v2/search?q=deep+learning&search_in=title,content&language=en&page=1&limit=10"
            )
        assert resp.status_code == 200
        _, kwargs = mock_svc.search.call_args
        assert kwargs["search_in"] == ["title", "content"]
        assert kwargs["language"] == "en"
        assert kwargs["limit"] == 10

    def test_missing_query_returns_422(self, client):
        resp = client.get("/api/v2/search")
        assert resp.status_code == 422
