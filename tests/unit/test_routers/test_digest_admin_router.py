"""Nothing could write the digest's admin block, so it was always empty."""

from unittest.mock import MagicMock, patch


def _doc():
    d = MagicMock()
    d.id = "DOC_001"
    d.digest_admin = None
    return d


def test_put_stores_the_block_and_invalidates_the_cached_export(client, mock_db):
    doc = _doc()
    mock_db.query.return_value.filter.return_value.first.return_value = doc

    with patch("serving.routers.digest_router.export_service") as exports:
        resp = client.put(
            "/api/v2/documents/DOC_001/digest/admin",
            json={"reviewer": "Nguyễn Văn A", "entry_date": "28/07/2026"},
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "reviewer": "Nguyễn Văn A",
        "reviewer_approved": "",
        "entry_date": "28/07/2026",
    }
    assert doc.digest_admin["reviewer"] == "Nguyễn Văn A"
    # A stale digest sits in MinIO keyed by document id; without this the
    # download keeps returning the version with the empty block.
    exports.invalidate_digest_export.assert_called_once_with("DOC_001")


def test_get_returns_the_stored_block(client, mock_db):
    doc = _doc()
    doc.digest_admin = {"reviewer": "A", "reviewer_approved": "B", "entry_date": "C"}
    mock_db.query.return_value.filter.return_value.first.return_value = doc

    resp = client.get("/api/v2/documents/DOC_001/digest/admin")

    assert resp.status_code == 200
    assert resp.json()["reviewer_approved"] == "B"


def test_bad_payload_is_rejected_with_a_reason(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    resp = client.put("/api/v2/documents/DOC_001/digest/admin", json={"reviewer": 42})

    assert resp.status_code == 400
    assert "reviewer" in resp.json()["detail"]


def test_unknown_document_is_404(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None

    resp = client.put("/api/v2/documents/DOC_001/digest/admin", json={"reviewer": "A"})

    assert resp.status_code == 404
