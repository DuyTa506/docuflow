"""Regression test: Translation must have at most one row per
(document_id, target_language) — this is what stops retries/races from
piling up duplicate 'VI-FAILED' rows in the language combobox.
"""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from data.db_models import Translation


def _translation(document_id: str, target_language: str, **kwargs):
    defaults = dict(id=str(uuid.uuid4()), status="PENDING")
    defaults.update(kwargs)
    return Translation(document_id=document_id, target_language=target_language, **defaults)


def test_duplicate_language_pair_rejected(test_db_session):
    doc_id = str(uuid.uuid4())
    test_db_session.add(_translation(doc_id, "vi"))
    test_db_session.commit()

    test_db_session.add(_translation(doc_id, "vi"))
    with pytest.raises(IntegrityError):
        test_db_session.commit()
    test_db_session.rollback()


def test_different_language_pairs_allowed(test_db_session):
    doc_id = str(uuid.uuid4())
    test_db_session.add(_translation(doc_id, "vi"))
    test_db_session.add(_translation(doc_id, "en"))
    test_db_session.commit()

    count = test_db_session.query(Translation).filter(Translation.document_id == doc_id).count()
    assert count == 2


def test_same_language_different_documents_allowed(test_db_session):
    test_db_session.add(_translation(str(uuid.uuid4()), "vi"))
    test_db_session.add(_translation(str(uuid.uuid4()), "vi"))
    test_db_session.commit()
