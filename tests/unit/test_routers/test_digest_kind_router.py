"""Nhận diện thể loại đọc bìa — nó có thể sai, nên phải sửa được bằng tay."""

from unittest.mock import MagicMock, patch


def _doc(title="Kỷ yếu Hội nghị khoa học 2025", override=None):
    d = MagicMock()
    d.id = "DOC_001"
    d.title = title
    d.digest_doc_kind = override
    return d


def test_get_reports_the_detected_kind_and_why(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    resp = client.get("/api/v2/documents/DOC_001/digest/kind")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_kind"] == "proceedings"
    assert body["doc_kind_source"] == "detected"
    assert body["doc_kind_reason"] == "kỷ yếu"
    assert body["override"] is None


def test_put_overrides_detection_and_invalidates_the_export(client, mock_db):
    doc = _doc()
    mock_db.query.return_value.filter.return_value.first.return_value = doc

    with patch("serving.routers.digest_router.export_service") as exports:
        resp = client.put("/api/v2/documents/DOC_001/digest/kind", json={"doc_kind": "book"})

    assert resp.status_code == 200
    assert resp.json()["doc_kind"] == "book"
    assert resp.json()["doc_kind_source"] == "explicit"
    assert doc.digest_doc_kind == "book"
    exports.invalidate_digest_export.assert_called_once_with("DOC_001")


def test_null_returns_to_auto_detection(client, mock_db):
    doc = _doc(override="book")
    mock_db.query.return_value.filter.return_value.first.return_value = doc

    with patch("serving.routers.digest_router.export_service"):
        resp = client.put("/api/v2/documents/DOC_001/digest/kind", json={"doc_kind": None})

    assert doc.digest_doc_kind is None
    assert resp.json()["doc_kind"] == "proceedings"
    assert resp.json()["doc_kind_source"] == "detected"


def test_unknown_kind_is_rejected_with_the_allowed_values(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    resp = client.put("/api/v2/documents/DOC_001/digest/kind", json={"doc_kind": "magazine"})

    assert resp.status_code == 400
    assert "proceedings" in resp.json()["detail"]


def test_changing_the_kind_says_a_rerun_is_needed(client, mock_db):
    """§2.2 đã được tóm tắt bằng prompt của thể loại cũ — đổi nhãn không viết lại nó."""
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    with patch("serving.routers.digest_router.export_service"):
        resp = client.put("/api/v2/documents/DOC_001/digest/kind", json={"doc_kind": "book"})

    assert "main-content" in resp.json()["rerun_required"]


def test_unknown_document_is_404(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None

    resp = client.get("/api/v2/documents/DOC_001/digest/kind")

    assert resp.status_code == 404
