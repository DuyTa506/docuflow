from unittest.mock import MagicMock, patch

from services.domain_context import DomainContext, load_domain_context

PROFILE = {"specialty": {"vi": "Đạn đạo học", "en": "Ballistics"}}
TERMBASE = {"Shaped Charges": {"vi": "Đạn lõm", "en": "Shaped charge"}}


class TestClause:
    def test_carries_the_domain_and_only_the_terms_in_the_text(self):
        ctx = DomainContext(PROFILE, TERMBASE)

        clause = ctx.clause("the Shaped Charges chapter")

        assert "Đạn đạo học" in clause and "Shaped Charges → Đạn lõm" in clause

    def test_terms_absent_from_the_text_are_left_out(self):
        clause = DomainContext(PROFILE, TERMBASE).clause("something unrelated")

        assert "Đạn lõm" not in clause and "Đạn đạo học" in clause

    def test_an_empty_context_adds_nothing(self):
        assert DomainContext().clause("anything") == ""

    def test_no_text_means_the_whole_termbase_up_to_the_limit(self):
        assert "Shaped Charges → Đạn lõm" in DomainContext({}, TERMBASE).clause()


class TestLoad:
    def _db(self, row):
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.first.return_value = row
        return MagicMock(session=MagicMock(return_value=session))

    def test_reads_both_columns(self):
        row = MagicMock(domain_profile=PROFILE, termbase=TERMBASE)
        with patch("services.domain_context.get_db_manager", return_value=self._db(row)):
            ctx = load_domain_context("DOC_001")

        assert ctx.profile == PROFILE and ctx.termbase == TERMBASE

    def test_a_database_error_degrades_to_an_empty_context(self):
        with patch("services.domain_context.get_db_manager", side_effect=RuntimeError("down")):
            ctx = load_domain_context("DOC_001")

        assert ctx.clause("x") == ""
