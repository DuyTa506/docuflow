"""«Từ khoá và hướng nghiên cứu tỷ lệ thuận với số trang» — tester's note on a 502-page book."""

import pytest

from utils.digest_format import digest_targets


def test_a_502_page_book_gets_the_testers_thirty_keywords():
    keywords, directions = digest_targets(502)

    assert keywords == 30
    assert directions == 13


def test_a_short_document_gets_fewer_than_a_long_one():
    short, long_ = digest_targets(20), digest_targets(500)

    assert short[0] < long_[0] and short[1] < long_[1]


@pytest.mark.parametrize("pages", [0, None, -5])
def test_unknown_length_falls_back_to_the_form_default(pages):
    assert digest_targets(pages) == (20, 8)


def test_the_counts_are_clamped():
    assert digest_targets(1) == (10, 4)
    assert digest_targets(100000) == (40, 15)


class TestWhereTheTargetsApply:
    def _digest(self, n_keywords, target):
        from services.digest_service import DigestResult, KeywordEntry

        return DigestResult(
            document_id="DOC_001",
            title="Ballistics",
            source_language="en",
            original_filename=None,
            bibliographic={"title_display": "Ballistics", "pages": "502"},
            keywords=[
                KeywordEntry(keyword=f"kw{i}", display=f"Từ khoá {i} (kw{i})", weight=0.9)
                for i in range(n_keywords)
            ],
            keyword_target=target,
        )

    def _rendered_keywords(self, digest):
        import io

        from docx import Document

        from services.digest_renderer import DigestRenderer

        doc = Document(io.BytesIO(DigestRenderer().render(digest)))
        return [p.text for p in doc.paragraphs if p.text.lstrip("- ").startswith("Từ khoá ")]

    def test_the_renderer_prints_as_many_as_the_document_earns(self):
        assert len(self._rendered_keywords(self._digest(40, 30))) == 30

    def test_a_short_document_prints_fewer(self):
        assert len(self._rendered_keywords(self._digest(40, 10))) == 10

    def test_the_quality_report_measures_against_the_documents_own_target(self):
        from types import SimpleNamespace

        from services.pipeline.quality import keyword_shortfall

        keywords = [object()] * 12

        assert keyword_shortfall(SimpleNamespace(keywords=keywords, keyword_target=30)) == (12, 30)
        assert keyword_shortfall(SimpleNamespace(keywords=keywords, keyword_target=10)) is None
        assert keyword_shortfall(SimpleNamespace(keywords=keywords)) == (12, 20)  # older callers


class TestKeywordRunSizesItself:
    @pytest.mark.asyncio
    async def test_the_pipeline_asks_for_the_pages_based_count(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from services.keyword_service import KeywordService

        svc = KeywordService()
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.first.return_value = MagicMock(
            total_pages=502
        )
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            MagicMock(id="KE_1")
        )
        with (
            patch("services.keyword_service.get_db_manager") as dbm,
            patch.object(svc, "_do_extract", new=AsyncMock(return_value={})) as do_extract,
        ):
            dbm.return_value.session.return_value = session
            await svc.run_for_pipeline("DOC_001")

        assert do_extract.await_args.args[1] == 30


class TestResearchDirectionBudget:
    def test_the_count_follows_the_length_up_to_the_ceiling(self, monkeypatch):
        from config.settings import settings
        from services.research_direction_service import direction_budget

        monkeypatch.setattr(settings, "research_directions_max_items", 15)
        monkeypatch.setattr(settings, "research_directions_max_tokens", 2000)

        assert direction_budget(60)[0] == 4
        assert direction_budget(502)[0] == 13

    def test_the_ceiling_still_caps_a_huge_document(self, monkeypatch):
        from config.settings import settings
        from services.research_direction_service import direction_budget

        monkeypatch.setattr(settings, "research_directions_max_items", 8)

        assert direction_budget(5000)[0] == 8

    def test_the_output_room_grows_with_the_count(self, monkeypatch):
        from config.settings import settings
        from services.research_direction_service import direction_budget

        monkeypatch.setattr(settings, "research_directions_max_items", 15)
        monkeypatch.setattr(settings, "research_directions_max_tokens", 2000)

        assert direction_budget(5000)[1] > 2000
        assert direction_budget(60)[1] == 2000

    def test_an_unreadable_page_count_falls_back(self):
        from utils.digest_format import digest_targets

        assert digest_targets("n/a") == (20, 8)
