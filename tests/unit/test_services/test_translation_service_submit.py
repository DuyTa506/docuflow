"""TranslationService.submit() must not create a second Translation row when
two requests race for the same (document_id, target_language) — the unique
constraint catches it and submit() must recover by reusing the row that won,
instead of crashing or leaking a duplicate.
"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

import data.repositories as repos_module
import services.export_service as export_service_module
from services.task_manager import task_manager as task_manager_singleton
from services.translation_service import TranslationService


@pytest.fixture(autouse=True)
def _patch_collaborators(monkeypatch):
    fake_doc = MagicMock(source_language="en")
    monkeypatch.setattr(repos_module.DocumentRepository, "get", lambda self, doc_id: fake_doc)
    monkeypatch.setattr(task_manager_singleton, "get_active_task_id", lambda *a, **k: None)
    monkeypatch.setattr(task_manager_singleton, "submit", lambda *a, **k: "TASK_TEST")
    monkeypatch.setattr(
        export_service_module.export_service.storage, "delete", lambda *a, **k: None
    )


def test_submit_recovers_from_integrity_error_race():
    db = MagicMock()
    winner = MagicMock(id="WINNER_TRANS_ID")

    # 1st query = in-flight PENDING/IN_PROGRESS check -> none
    # 2nd query = existing_trans lookup before insert -> none (so submit()
    #             takes the insert branch)
    # 3rd query = re-lookup inside the except-IntegrityError branch -> the row
    #             a concurrent submit() already committed
    db.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [
        None,
        None,
        winner,
    ]
    # 1st commit = the racing insert -> raises (unique constraint hit)
    # 2nd commit = _reset_translation_for_retry()'s commit -> succeeds
    db.commit.side_effect = [IntegrityError("duplicate", None, None), None]

    svc = TranslationService()
    task_id, translation_id, reused = svc.submit(db, "DOC_001", "vi")

    assert task_id == "TASK_TEST"
    assert translation_id == "WINNER_TRANS_ID"
    assert reused is True
    db.rollback.assert_called_once()
    assert winner.status == "PENDING"


def test_submit_creates_single_row_when_no_race():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [
        None,
        None,
    ]
    db.commit.side_effect = [None]
    # Simulate a real session's refresh() populating the server-generated
    # default id after flush (a plain MagicMock db never actually inserts).
    db.refresh.side_effect = lambda obj: setattr(obj, "id", obj.id or "NEW_TRANS_ID")

    svc = TranslationService()
    task_id, translation_id, reused = svc.submit(db, "DOC_001", "vi")

    assert task_id == "TASK_TEST"
    assert translation_id == "NEW_TRANS_ID"
    assert reused is False
    db.rollback.assert_not_called()
