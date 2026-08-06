"""§2.3 should not copy the table of contents — §2.2 just above prints it.

In N4.11.160, 6 of the 20 keywords were chapter titles, and §2.2 prints each one
verbatim together with a full summary paragraph. An entry that just appeared as
a heading two paragraphs above adds nothing in the keyword list.

Matches **exactly** (after normalisation), never partially: "bộ nhớ ảo" is a real
subject even though chapter 6 discusses it.
"""

import pytest

from utils.digest_format import drop_heading_keywords

HEADINGS = [
    "Giới thiệu",
    "Введение",
    "Mức logic số",
    "Цифровой логический уровень",
    "Mức kiến trúc tập lệnh",
    "Уровень архитектуры набора команд",
    "Kiến trúc máy tính song song",
]


def _names(kws):
    return [k["keyword"] for k in kws]


class TestDropHeadingKeywords:
    def test_a_keyword_matching_a_vietnamese_heading_is_dropped(self):
        kws = [
            {"keyword": "mức kiến trúc tập lệnh", "display": "mức kiến trúc tập lệnh"},
            {"keyword": "bộ nhớ ảo", "display": "bộ nhớ ảo"},
        ]

        assert _names(drop_heading_keywords(kws, HEADINGS)) == ["bộ nhớ ảo"]

    def test_the_original_language_side_counts_too(self):
        """§2.2 prints both halves; matching either one is still a copy.

        `mức logic kỹ thuật số` differs in wording from the Vietnamese title
        `Mức logic số`, but the original-language half matches exactly.
        """
        kws = [
            {
                "keyword": "цифровой логический уровень",
                "display": "mức logic kỹ thuật số (цифровой логический уровень)",
            }
        ]

        assert drop_heading_keywords(kws, HEADINGS) == []

    def test_a_leading_article_does_not_rescue_a_heading(self):
        """`các kiến trúc máy tính song song` vs the title `Kiến trúc máy tính song song`."""
        kws = [
            {
                "keyword": "параллельных компьютерных архитектур",
                "display": "các kiến trúc máy tính song song (параллельных компьютерных архитектур)",
            }
        ]

        assert drop_heading_keywords(kws, HEADINGS) == []

    @pytest.mark.parametrize(
        "keyword",
        [
            "bộ nhớ ảo",
            "vi kiến trúc",
            "hệ thống RISC và CISC",
            "logic",
        ],
    )
    def test_partial_overlap_is_not_enough(self, keyword):
        """`vi kiến trúc` is a real subject even though chapter 4 is `Mức vi kiến trúc`."""
        kws = [{"keyword": keyword, "display": keyword}]

        assert len(drop_heading_keywords(kws, HEADINGS)) == 1

    def test_without_headings_nothing_is_dropped(self):
        kws = [{"keyword": "mức kiến trúc tập lệnh", "display": "mức kiến trúc tập lệnh"}]

        assert len(drop_heading_keywords(kws, [])) == 1
        assert len(drop_heading_keywords(kws, None)) == 1

    def test_empty_list_is_safe(self):
        assert drop_heading_keywords([], HEADINGS) == []

    def test_cjk_headings_match_too(self):
        kws = [
            {"keyword": "流水线", "display": "kỹ thuật đường ống (流水线)"},
            {"keyword": "缓存一致性", "display": "nhất quán bộ nhớ đệm (缓存一致性)"},
        ]

        assert _names(drop_heading_keywords(kws, ["流水线", "Kỹ thuật đường ống"])) == [
            "缓存一致性"
        ]


class TestTranslationVarianceStillMatches:
    """The heading and the keyword are translated by two separate calls.

    Re-running §2.2 on N4.11.160 rendered `Уровень` as "Cấp độ" where the
    keyword pass had said "Mức", so `Mức kiến trúc tập lệnh` — chapter 5's own
    title, printed in full two paragraphs above — slipped back into §2.3. Exact
    matching across two independent translations of the same Russian phrase is
    not a stable test.

    So a keyword is also a repeat when nearly all of its words are the heading's
    words. The floor of three words and the high ratio are what keep short,
    genuine subjects: `bộ nhớ ảo` shares nothing with any chapter title, and
    `số học nhị phân` keeps three of its four words out of `Số nhị phân` — 0.75,
    under the bar — so both survive.
    """

    HEADINGS = [
        "Cấp độ kiến trúc tập lệnh",
        "Уровень архитектуры набора команд",
        "Số nhị phân",
        "Cấp độ hệ điều hành",
        "Kiến trúc máy tính song song",
    ]

    def test_a_differently_worded_translation_of_a_heading_is_dropped(self):
        kws = [{"keyword": "mức kiến trúc tập lệnh", "display": "mức kiến trúc tập lệnh"}]

        assert drop_heading_keywords(kws, self.HEADINGS) == []

    @pytest.mark.parametrize("keyword", ["bộ nhớ ảo", "số học nhị phân", "đa luồng"])
    def test_a_genuine_subject_survives(self, keyword):
        kws = [{"keyword": keyword, "display": keyword}]

        assert len(drop_heading_keywords(kws, self.HEADINGS)) == 1

    def test_a_short_keyword_still_needs_an_exact_match(self):
        """Two words are too few for a ratio to mean anything."""
        kws = [{"keyword": "kiến trúc", "display": "kiến trúc"}]

        assert len(drop_heading_keywords(kws, self.HEADINGS)) == 1
