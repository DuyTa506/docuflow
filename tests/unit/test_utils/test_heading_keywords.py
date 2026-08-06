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
