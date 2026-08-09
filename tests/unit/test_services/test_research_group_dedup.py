"""Strong research groups must not list a group and its own narrow branch.

N4.11.160 returned 8 groups including both `Kiến trúc máy tính` and `Kiến trúc
máy tính hiệu năng cao` — the second ate one of the eight slots without adding
anything.

The old de-duplication compared normalised names only, so two different strings
both got through.

The threshold has to be very cautious: dropping a real group loses information
from an official document, while keeping a redundant pair is only noise.
Vietnamese is especially easy to get wrong because compounds split into separate
syllables — `Kỹ thuật điện` is a syllable prefix of `Kỹ thuật điện tử` yet the
two disciplines are entirely different.
"""

import pytest

from services.usage_scope_service import _clean_research_groups


class TestSpecialisationsAreFolded:
    def test_a_narrower_group_after_a_broader_one_is_dropped(self):
        out = _clean_research_groups(
            [
                "Kiến trúc máy tính",
                "Hệ thống nhúng",
                "Thiết kế vi xử lý",
                "Kiến trúc máy tính hiệu năng cao",
            ]
        )

        assert out == ["Kiến trúc máy tính", "Hệ thống nhúng", "Thiết kế vi xử lý"]

    def test_the_first_listed_wins_regardless_of_order(self):
        """The model orders by descending relevance — the earlier entry is kept."""
        out = _clean_research_groups(["Kiến trúc máy tính hiệu năng cao", "Kiến trúc máy tính"])

        assert out == ["Kiến trúc máy tính hiệu năng cao"]


class TestDistinctDisciplinesSurvive:
    @pytest.mark.parametrize(
        "pair",
        [
            # Entirely different compounds that merely share a leading syllable.
            ("Kỹ thuật điện", "Kỹ thuật điện tử"),
            ("Khoa học máy tính", "Khoa học vật liệu"),
            # A two-syllable qualifier — below the threshold, so both are kept.
            ("Trí tuệ nhân tạo", "Trí tuệ nhân tạo tạo sinh"),
            ("Kỹ thuật máy tính", "Kỹ thuật máy tính lượng tử"),
        ],
    )
    def test_a_short_extension_is_not_treated_as_a_specialisation(self, pair):
        assert _clean_research_groups(list(pair)) == list(pair)

    def test_a_short_base_never_swallows_anything(self):
        """`Điện` swallowing everything that starts with `Điện` would be too much."""
        out = _clean_research_groups(["Điện", "Điện tử công suất cao"])

        assert out == ["Điện", "Điện tử công suất cao"]


class TestWhereTheLineSits:
    """The threshold is a convention, not a truth — record where it sits.

    There is no mechanical way to tell "redundant narrow branch" from "genuinely
    narrower discipline" in a list of free-text names. The threshold leans
    towards **keeping**: dropping a real group loses information from an official
    document, while keeping a redundant pair only costs one of eight slots. The
    semantic judgement is pushed to the prompt.
    """

    def test_three_extra_tokens_fold(self):
        assert _clean_research_groups(
            ["Kiến trúc máy tính", "Kiến trúc máy tính hiệu năng cao"]
        ) == ["Kiến trúc máy tính"]

    def test_two_extra_tokens_do_not(self):
        pair = ["Hệ thống nhúng", "Hệ thống nhúng ô-tô"]

        assert _clean_research_groups(pair) == pair

    def test_unrelated_groups_are_untouched(self):
        items = [
            "Kiến trúc máy tính",
            "Hệ thống nhúng",
            "Thiết kế vi xử lý",
            "Hệ điều hành",
            "Tính toán song song",
            "Hệ thống máy tính đa lõi",
            "Công nghệ vi mạch",
        ]

        assert _clean_research_groups(items) == items


class TestExistingBehaviourIsPreserved:
    def test_exact_duplicates_still_collapse(self):
        assert _clean_research_groups(["Hệ thống nhúng", "  hệ thống  nhúng "]) == [
            "Hệ thống nhúng"
        ]

    def test_non_list_and_non_string_entries(self):
        assert _clean_research_groups(None) == []
        assert _clean_research_groups(["Hệ thống nhúng", 42, None, ""]) == ["Hệ thống nhúng"]

    def test_the_cap_still_holds(self):
        out = _clean_research_groups([f"Nhóm nghiên cứu số {i}" for i in range(20)])

        assert len(out) == 8
