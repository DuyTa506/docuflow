"""`$a = 700, b = 400, c = 300$` printed literal dollar signs into an official document.

`looks_like_math` accepted only LaTeX (`\\`, `^`, `_`, `{}`) or a bare variable
(`$n$`). A run of assignments contains none of those, so it was treated as prose
and the `$` pair was printed — twice in chapter 10 of N4.11.160.

The guard still has to hold: `$100` is a price, not a formula. So the condition
is **an `=` with mathematical symbols on both sides**, not "contains a digit".
"""

import pytest

from utils.math_omml import looks_like_math


class TestAssignments:
    @pytest.mark.parametrize(
        "text",
        [
            "a = 700, b = 400, c = 300",
            "a = 5, b = 210, c = 195",
            "x = 1",
            "n = 32",
            "f = 2, g = 3",
        ],
    )
    def test_variable_assignment_is_math(self, text):
        assert looks_like_math(text) is True


class TestStillRejectsProse:
    @pytest.mark.parametrize(
        "text",
        [
            "100",
            "5",
            "1.200.000 đồng",
            "giá là 100 và 200",
            "và tổng chi phí = một khoản lớn",
            "USD 250 cho mỗi bản",
            "",
        ],
    )
    def test_prose_and_currency_are_not_math(self, text):
        assert looks_like_math(text) is False


class TestUnchangedBehaviour:
    @pytest.mark.parametrize("text", ["n", "2^3 = 8", "2^n", r"\Delta w", r"9 \times 10^{-28}"])
    def test_previous_cases_still_math(self, text):
        assert looks_like_math(text) is True
