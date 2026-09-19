"""Inline LaTeX printed raw in translated PDFs (E2E, Digital Control p.23):
`$T_{sample} = T_S / N$`, `$f_{sample} = N \\cdot f_S$`. PDF text boxes cannot
typeset math, so fragments become readable Unicode text instead: sub/superscript
glyphs where Unicode has them (digits, single letters such as ₛ), else `_x`."""

import pytest

from utils.math_omml import inline_math_to_plain


@pytest.mark.parametrize(
    "src, expected",
    [
        ("chu kỳ $T_{sample} = T_S / N$ giây", "chu kỳ T_sample = T_S / N giây"),
        ("$f_{sample} = N \\cdot f_S$", "f_sample = N · f_S"),
        ("tín hiệu $m_s(t)$ và $x^{2}$", "tín hiệu mₛ(t) và x²"),
        ("\\( \\alpha + \\beta \\)", "α + β"),
        ("$\\frac{a}{b}$", "a/b"),
        ("giá $5 và $10", "giá $5 và $10"),  # currency untouched
    ],
)
def test_inline_math_becomes_plain_text(src, expected):
    assert inline_math_to_plain(src) == expected
