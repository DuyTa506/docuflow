"""§2.3 asks for 20 keywords and printed 13.

The assembly filters — title/authors, chapter headings, bilingual collapse —
removed 7 of the 20 on N4.11.160, and nothing took their place. Both caps in the
chain were the same number, so every keyword the filters rejected was a slot
lost rather than a slot re-filled.

The filters cannot move earlier: `drop_bibliographic_keywords` needs the
bibliographic record, whose stage runs *in parallel* with keywords, and
`drop_heading_keywords` needs the chapter titles, whose stage runs *after*. So
what is stored becomes a ranked pool, and §2.3 takes the best `target` entries
that survive.

The pool is a ceiling, never a floor: a document with only nine real subjects
still yields nine. Padding a digest to a round number is what this whole
sequence of fixes exists to stop.
"""

from services.digest_service import DIGEST_KEYWORD_TARGET, KeywordEntry
from services.keyword_service import keyword_pool_size


class TestPoolSize:
    def test_the_pool_is_larger_than_what_the_digest_shows(self):
        assert keyword_pool_size(DIGEST_KEYWORD_TARGET) > DIGEST_KEYWORD_TARGET

    def test_the_pool_covers_the_measured_loss(self):
        """7 of 20 were filtered out — the reserve has to clear that."""
        assert keyword_pool_size(20) >= 27

    def test_a_request_for_one_still_asks_for_more_than_one(self):
        assert keyword_pool_size(1) > 1

    def test_zero_and_negative_do_not_produce_a_negative_pool(self):
        assert keyword_pool_size(0) >= 0
        assert keyword_pool_size(-5) >= 0


def _entries(names):
    return [KeywordEntry(keyword=n, display=n, weight=1.0 - i / 100) for i, n in enumerate(names)]


class TestDigestRefill:
    """Exercised through the renderer's context builder, where the filters run."""

    @staticmethod
    def _keywords(pool_names, headings=(), title="", authors=""):
        from unittest.mock import MagicMock

        from services.digest_renderer import DigestRenderer
        from services.digest_service import ChapterEntry, DigestResult
        from utils.digest_format import bibliographic_defaults

        bib = bibliographic_defaults(title=title)
        bib["authors"] = authors
        digest = DigestResult(
            document_id="DOC_001",
            title=title,
            source_language="ru",
            original_filename="x.pdf",
            bibliographic=bib,
            abstract="Tóm tắt.",
            chapters=[
                ChapterEntry(number=i + 1, title_vi=h, title_original=h, content="Nội dung.")
                for i, h in enumerate(headings)
            ],
            keywords=_entries(pool_names),
            usage_scope={
                "undergraduate": [],
                "master": [],
                "phd": [],
                "strong_research_groups": [],
            },
            research_directions=[],
        )
        context = DigestRenderer()._build_context(digest, MagicMock())
        return [k["keyword"] for k in context["keywords"]]

    def test_filtered_slots_are_refilled_from_the_pool(self):
        headings = ["Bộ nhớ ảo và phân trang", "Kiến trúc tập lệnh x86"]
        pool = list(headings) + [f"chủ đề số {i}" for i in range(DIGEST_KEYWORD_TARGET + 5)]

        got = self._keywords(pool, headings=headings)

        assert len(got) == DIGEST_KEYWORD_TARGET
        assert not set(got) & set(headings)

    def test_the_refill_takes_the_next_best_not_an_arbitrary_one(self):
        headings = ["Bộ nhớ ảo và phân trang"]
        pool = headings + [f"chủ đề số {i}" for i in range(DIGEST_KEYWORD_TARGET + 3)]

        got = self._keywords(pool, headings=headings)

        assert got == [f"chủ đề số {i}" for i in range(DIGEST_KEYWORD_TARGET)]

    def test_a_short_pool_is_never_padded(self):
        pool = [f"chủ đề số {i}" for i in range(9)]

        assert len(self._keywords(pool)) == 9

    def test_the_digest_never_exceeds_its_target(self):
        pool = [f"chủ đề số {i}" for i in range(DIGEST_KEYWORD_TARGET * 2)]

        assert len(self._keywords(pool)) == DIGEST_KEYWORD_TARGET
