from unittest.mock import patch


class TestSearch:
    def test_success(self, client):
        mock_result = {"results": [], "total": 0, "query": "machine learning"}
        with patch("serving.routers.search_router._svc") as mock_svc:
            mock_svc.search.return_value = mock_result
            resp = client.get("/api/v2/search?q=machine+learning")
        assert resp.status_code == 200
        assert resp.json()["query"] == "machine learning"

    def test_with_all_params(self, client):
        mock_result = {"results": [], "total": 0, "query": "deep learning"}
        with patch("serving.routers.search_router._svc") as mock_svc:
            mock_svc.search.return_value = mock_result
            resp = client.get(
                "/api/v2/search?q=deep+learning&search_in=title,content&language=en&limit=10&offset=0"
            )
        assert resp.status_code == 200
        _, kwargs = mock_svc.search.call_args
        assert kwargs["search_in"] == ["title", "content"]
        assert kwargs["language"] == "en"
        assert kwargs["limit"] == 10

    def test_missing_query_returns_422(self, client):
        resp = client.get("/api/v2/search")
        assert resp.status_code == 422
