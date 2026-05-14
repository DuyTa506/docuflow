from unittest.mock import patch, MagicMock


def _digest():
    d = MagicMock()
    d.document_id = "DOC_001"
    d.title = "Test Document"
    d.source_language = "en"
    d.original_filename = "test.pdf"
    d.abstract = "Abstract text"
    d.main_content = "Main content"
    d.keywords = []
    d.research_directions = []
    d.missing = []
    return d


class TestGetDigest:
    def test_success(self, client):
        mock_digest = _digest()
        with patch("serving.routers.digest_router._digest_svc") as mock_svc:
            mock_svc.assemble.return_value = mock_digest
            resp = client.post("/api/v2/documents/DOC_001/digest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "DOC_001"
        assert data["title"] == "Test Document"
        assert data["missing"] == []

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.digest_router._digest_svc") as mock_svc:
            mock_svc.assemble.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/digest")
        assert resp.status_code == 404


class TestDownloadDigest:
    def test_success_returns_docx(self, client):
        mock_digest = _digest()
        with patch("serving.routers.digest_router._digest_svc") as mock_svc, \
             patch("serving.routers.digest_router._renderer") as mock_renderer:
            mock_svc.assemble.return_value = mock_digest
            mock_renderer.render.return_value = b"PK\x03\x04DOCX"  # minimal docx magic bytes
            resp = client.get("/api/v2/documents/DOC_001/digest/download")
        assert resp.status_code == 200
        assert "application/vnd.openxmlformats" in resp.headers["content-type"]
        assert "attachment" in resp.headers["content-disposition"]
        assert "digest_Test Document.docx" in resp.headers["content-disposition"]

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.digest_router._digest_svc") as mock_svc:
            mock_svc.assemble.side_effect = ValueError("Not found")
            resp = client.get("/api/v2/documents/DOC_999/digest/download")
        assert resp.status_code == 404
