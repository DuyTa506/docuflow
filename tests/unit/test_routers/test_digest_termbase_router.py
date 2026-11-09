"""The termbase is generated, so a librarian needs a way to correct a term."""

from unittest.mock import MagicMock, patch


def _doc():
    d = MagicMock()
    d.id = "DOC_001"
    d.domain_profile = {"specialty": {"vi": "Đạn đạo học", "en": "Ballistics"}}
    d.termbase = {"Shaped Charges": {"vi": "Đạn lõm", "en": "Shaped charge"}}
    return d


def test_get_lists_terms_as_editable_rows(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    resp = client.get("/api/v2/documents/DOC_001/digest/termbase")

    assert resp.status_code == 200
    body = resp.json()
    assert body["termbase"] == [{"src": "Shaped Charges", "vi": "Đạn lõm", "en": "Shaped charge"}]
    assert body["domain_profile"]["specialty"]["vi"] == "Đạn đạo học"


def test_put_replaces_the_termbase_and_drops_the_stale_export(client, mock_db):
    doc = _doc()
    mock_db.query.return_value.filter.return_value.first.return_value = doc

    with patch("serving.routers.digest_router.export_service") as exports:
        resp = client.put(
            "/api/v2/documents/DOC_001/digest/termbase",
            json={"termbase": [{"src": "Wound Ballistics", "vi": "Đạn đạo vết thương"}]},
        )

    assert resp.status_code == 200
    assert doc.termbase == {
        "Wound Ballistics": {"vi": "Đạn đạo vết thương", "en": "Wound Ballistics"}
    }
    exports.invalidate_digest_export.assert_called_once_with("DOC_001")


def test_put_without_a_list_is_rejected(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _doc()

    resp = client.put("/api/v2/documents/DOC_001/digest/termbase", json={"termbase": "x"})

    assert resp.status_code == 400
