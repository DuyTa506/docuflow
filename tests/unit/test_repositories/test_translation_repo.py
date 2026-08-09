"""Unit tests for translation list deduplication."""

from unittest.mock import MagicMock

from data.repositories.translation_repo import TranslationRepository


def _row(tid: str, lang: str, status: str = "FAILED"):
    t = MagicMock()
    t.id = tid
    t.target_language = lang
    t.status = status
    t.created_at = None
    return t


class TestTranslationRepositoryList:
    def test_list_returns_one_per_language(self):
        db = MagicMock()
        rows = [
            _row("T1", "vi", "FAILED"),
            _row("T2", "vi", "COMPLETED"),
            _row("T3", "en", "COMPLETED"),
        ]
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = rows
        repo = TranslationRepository(db)
        result = repo.list("DOC_001")
        assert len(result) == 2
        langs = {t.target_language for t in result}
        assert langs == {"vi", "en"}
