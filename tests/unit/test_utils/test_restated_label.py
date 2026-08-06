"""§2.2 states the unit label twice in 8 of the 12 entries of N4.11.160.

The renderer joins `chapter_heading(...) + " " + content`. The label and chapter
title are already in the heading, yet the model opens the body with that same
label:

    Phụ lục A. Số nhị phân (Двоичные числа). Phụ lục A (Приложение А) trình bày
    về các số nhị phân (двоичные числа)…

The prompt now forbids restating the label, but a prompt is not a guarantee —
this is an official document, so there is a mechanical backstop as well.

Only the *label* becomes a demonstrative ("Chương này"); the content is left
alone. Cutting the whole sentence loses information, and dropping the label
outright would leave the sentence starting in lower case.
"""

import pytest

from utils.digest_format import strip_restated_label


class TestStrips:
    @pytest.mark.parametrize(
        "body,label,expected",
        [
            (
                "Chương 2 trình bày về tổ chức của các hệ thống máy tính.",
                "Chương 2",
                "Chương này trình bày về tổ chức của các hệ thống máy tính.",
            ),
            (
                "Phụ lục A (Приложение А) trình bày về các số nhị phân.",
                "Phụ lục A",
                "Phụ lục này trình bày về các số nhị phân.",
            ),
            (
                "Chương 9 là phần Danh mục tài liệu tham khảo.",
                "Chương 9",
                "Chương này là phần Danh mục tài liệu tham khảo.",
            ),
            (
                "Phần II mô tả kiến trúc.",
                "Phần II",
                "Phần này mô tả kiến trúc.",
            ),
        ],
    )
    def test_leading_label_becomes_a_demonstrative(self, body, label, expected):
        assert strip_restated_label(body, label) == expected

    def test_leading_whitespace_is_tolerated(self):
        assert strip_restated_label("  Chương 4 tập trung vào…", "Chương 4").startswith(
            "Chương này tập trung"
        )


class TestLeavesAlone:
    """Only rewrites when the body restates *the very unit being printed*."""

    def test_a_different_unit_is_a_real_cross_reference(self):
        body = "Chương 8 cũng bàn về xử lý song song."

        assert strip_restated_label(body, "Chương 2") == body

    def test_a_label_in_the_middle_is_untouched(self):
        body = "Nội dung mở rộng ý của Chương 2 về tổ chức hệ thống."

        assert strip_restated_label(body, "Chương 2") == body

    def test_body_that_starts_with_substance_is_unchanged(self):
        body = "Cấp độ logic số nằm ở ranh giới giữa khoa học máy tính và kỹ thuật điện."

        assert strip_restated_label(body, "Chương 3") == body

    def test_empty_inputs_do_not_crash(self):
        assert strip_restated_label("", "Chương 1") == ""
        assert strip_restated_label("Chương 1 nói về…", "") == "Chương 1 nói về…"
        assert strip_restated_label(None, "Chương 1") == ""


class TestRealRegression:
    def test_the_n4_11_160_appendix_a_line(self):
        """A real case that reached the official document."""
        body = (
            "Phụ lục A (Приложение А) trình bày về các số nhị phân (двоичные числа) "
            "và hệ đếm cơ số 16."
        )

        out = strip_restated_label(body, "Phụ lục A")

        assert out.startswith("Phụ lục này trình bày")
        assert "Приложение А" not in out, "phần nhắc lại nhãn gốc cũng phải đi theo"
