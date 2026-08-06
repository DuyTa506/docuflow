"""The content candidate tier must work for every script, not only Latin.

Two separate defects were measured on N4.11.160 (Russian book, 816 pages).

**Script coverage.** The old `token_pattern` accepted `a-zA-Z` plus Latin
Extended (Latin-1, Latin-A/B and the Vietnamese block); Cyrillic, Greek, Han and
kana all fell outside it. So:

* Pure Russian/Chinese/Japanese text → `TfidfVectorizer` **raises `ValueError:
  empty vocabulary`**, the whole keywords stage fails and §2.3 comes out empty.
* Russian text with a few Latin part numbers → the vocabulary is just those part
  numbers (Core i7, OMAP4430, FPGA, ASCII), so the content tier is effectively
  empty and chapter titles take the slots: 6 of 20 keywords were the contents.

**Ranking.** What ran was never TF-IDF: with `corpus = [text, ""]` every term
appears in exactly one of two documents, so IDF is a constant and cancels out —
measured as a single distinct IDF value across the whole vocabulary. It was
plain term frequency, which is why the candidate list filled with overlapping
junk n-grams ("algebra describes digital"). YAKE replaces it: single-document by
design, no corpus needed, and it prunes near-duplicate spans itself.

CJK has no spaces, so neither YAKE nor any whitespace tokenizer applies —
character n-grams are used there instead. They are not a real segmenter, but the
spans they produce still appear verbatim in the document, satisfying the
grounding rule, and the LLM filters again afterwards.
"""

import pytest

from services.keyword_service import KeywordService


@pytest.fixture
def service():
    return KeywordService.__new__(KeywordService)


def _kws(candidates):
    return [c["keyword"] for c in candidates]


RUSSIAN = (
    "Цифровой логический уровень определяет вентили и булеву алгебру. "
    "Вентили строятся из транзисторов. Булева алгебра описывает цифровые схемы. "
    "Регистр хранит данные, триггер и защелка образуют память. "
) * 20

CHINESE = "计算机组成与设计 流水线 缓存一致性 虚拟内存 指令集架构 流水线 缓存一致性 " * 20

JAPANESE = "コンピュータの構成と設計 仮想記憶 命令セットアーキテクチャ 仮想記憶 パイプライン " * 20

ENGLISH = (
    "Digital logic level defines gates and boolean algebra. Gates are built from "
    "transistors. Boolean algebra describes digital circuits. "
) * 20


class TestNonLatinScripts:
    def test_russian_yields_cyrillic_terms(self, service):
        out = _kws(service._content_candidates(RUSSIAN, max_candidates=20))

        assert out, "pure Russian text must not come back empty"
        assert any("алгебра" in k or "вентили" in k for k in out), out

    def test_chinese_yields_candidates(self, service):
        out = _kws(service._content_candidates(CHINESE, max_candidates=20))

        assert out, "pure Chinese text must not come back empty"
        assert any("流水线" in k or "缓存" in k for k in out), out

    def test_japanese_yields_candidates(self, service):
        out = _kws(service._content_candidates(JAPANESE, max_candidates=20))

        assert out, "pure Japanese text must not come back empty"
        assert any("仮想記憶" in k or "パイプライン" in k for k in out), out

    def test_english_still_works(self, service):
        """The existing path must not regress."""
        out = _kws(service._content_candidates(ENGLISH, max_candidates=20))

        assert "boolean algebra" in out, out


class TestMixedScript:
    def test_russian_body_with_latin_part_numbers_keeps_both(self, service):
        """Exactly the N4.11.160 failure: Russian body, scattered Latin part numbers.

        The vocabulary used to collapse to those part numbers, leaving the content
        tier empty and letting chapter titles take every slot in §2.3.
        """
        text = RUSSIAN + ("Core i7 OMAP4430 ATmega168 FPGA ASCII " * 20)

        out = _kws(service._content_candidates(text, max_candidates=40))

        assert any("алгебра" in k or "вентили" in k for k in out), out


class TestRankingQuality:
    def test_the_real_term_outranks_its_fragments(self):
        """Frequency gave no ranking signal at all — everything tied.

        Measured on this text under the old ranker, `algebra`, `boolean`,
        `boolean algebra`, `digital` and `gates` ALL scored exactly 0.1815: the
        `corpus = [text, ""]` shape makes IDF a constant, so what was left was
        normalised term frequency, which ties for terms of equal count.

        With YAKE the multi-word term ranks first and is strictly separated from
        its own fragments. Note this does NOT remove overlapping windows like
        `algebra describes digital` — `dedup_lim` was measured and does not do
        that — it only stops them tying for the top slots.
        """
        svc = KeywordService.__new__(KeywordService)

        out = svc._content_candidates(ENGLISH, max_candidates=10)
        names = [c["keyword"] for c in out]
        scores = [c["score"] for c in out]

        assert names[0] == "boolean algebra", names
        assert names.index("boolean algebra") < names.index("algebra")
        assert len(set(scores)) > 1, "a ranker that ties everything is not ranking"

    def test_scores_are_higher_is_better(self):
        """YAKE is lower-is-better; the merged pool downstream is not."""
        svc = KeywordService.__new__(KeywordService)

        out = svc._content_candidates(ENGLISH, max_candidates=10)

        assert out == sorted(out, key=lambda c: c["score"], reverse=True)
        assert all(0.0 < c["score"] <= 1.0 for c in out), out


class TestDegradesInsteadOfRaising:
    """Keywords is a non-critical stage — a failure must degrade to a warning."""

    @pytest.mark.parametrize("text", ["", "   \n\t  ", "!!! ??? ... --- +++"])
    def test_no_usable_text_returns_empty(self, service, text):
        assert service._content_candidates(text) == []
