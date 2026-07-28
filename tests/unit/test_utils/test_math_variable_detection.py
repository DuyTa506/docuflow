"""`$n$` in ra nguyên `$n$` trong tổng thuật.

Đo trên bản tổng thuật DOC_065 do Gemma sinh: 7 công thức chuyển thành OMML
thật, nhưng còn **10 dấu `$`** in ra chữ — tất cả đều quanh biến một chữ:
`$n$`, `$f$`, `$e$`.

`looks_like_math` đòi fragment phải chứa một trong `\\^_{}` hoặc lệnh LaTeX.
Docstring nói rõ mục đích của nó: *"avoid treating currency like `$5` as a math
fragment"*. Nhưng **tiền luôn bắt đầu bằng chữ số** — một định danh ngắn bắt đầu
bằng chữ cái thì không thể là tiền, nên nới đúng theo mục đích ban đầu.
"""

import pytest

from utils.math_omml import looks_like_math


class TestVariablesAreMath:
    @pytest.mark.parametrize("frag", ["n", "f", "e", "x", "dx", "fs", "Re"])
    def test_short_identifier_is_a_variable(self, frag):
        assert looks_like_math(frag)


class TestCurrencyStillIsNot:
    @pytest.mark.parametrize("frag", ["5", "100", "5.99", "1,000", "20 000", "0"])
    def test_a_number_is_not_math(self, frag):
        """Chính ca mà heuristic sinh ra để chặn."""
        assert not looks_like_math(frag)

    @pytest.mark.parametrize("frag", ["", "   "])
    def test_empty_is_not_math(self, frag):
        assert not looks_like_math(frag)

    def test_a_sentence_is_not_math(self):
        """Hai dấu `$` cách nhau cả câu là dấu tiền tệ đứng đôi, không phải công thức."""
        assert not looks_like_math("50 cho mỗi đơn vị, tổng cộng ")

    def test_a_long_word_is_not_math(self):
        assert not looks_like_math("microarchitecture")


class TestExistingBehaviourKept:
    @pytest.mark.parametrize("frag", [r"\Delta w", "2^n", "x_1", r"f_s \geq 2f_{max}"])
    def test_real_latex_still_matches(self, frag):
        assert looks_like_math(frag)
