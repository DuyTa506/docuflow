"""A §2.2 entry should open with its substance, not by announcing itself.

Two reference digests agree, and they are the standard this output is judged
against — `template/N3.11.2 (Tong thuat).docx`, an approved deliverable, and the
ChatGPT draft supplied alongside it. Both contain **zero** lead-in phrases. The
approved one opens every entry with the verb itself:

    Trình bày các mô hình phân tích tín hiệu và nhiễu…
    Phân tích quy tắc quyết định, tập trung vào các ngưỡng phát hiện…
    Mô tả và đánh giá các bộ phát hiện theo kiến thức…

Ours opened all 12 entries with a subject that carries no information, because
the renderer already prints the heading immediately before it:

    Chương 2. Tổ chức hệ thống máy tính (Организация…). Chương này cung cấp
    cái nhìn tổng quan về tổ chức của các hệ thống máy tính…

Both forms of that subject are removed: the unit's own label, which the model
restates verbatim (8 of 12 entries), and the demonstrative it becomes. The
remainder is capitalised so the sentence still starts as a sentence.

Two things are deliberately left alone. A label naming a *different* unit is a
real cross-reference, and a remainder beginning with the copula "là" — "Chương
này là phần Danh mục tài liệu tham khảo" — would read as "Là phần Danh mục…",
which is not Vietnamese anyone writes.
"""

import pytest

from utils.digest_format import open_with_substance


class TestOwnLabelIsDropped:
    @pytest.mark.parametrize(
        "body,label,expected",
        [
            (
                "Chương 2 trình bày về tổ chức của các hệ thống máy tính.",
                "Chương 2",
                "Trình bày về tổ chức của các hệ thống máy tính.",
            ),
            (
                "Phụ lục A (Приложение А) trình bày về các số nhị phân.",
                "Phụ lục A",
                "Trình bày về các số nhị phân.",
            ),
            (
                "Phần II mô tả kiến trúc.",
                "Phần II",
                "Mô tả kiến trúc.",
            ),
        ],
    )
    def test_the_label_goes_and_the_verb_leads(self, body, label, expected):
        assert open_with_substance(body, label) == expected

    def test_leading_whitespace_is_tolerated(self):
        assert open_with_substance("  Chương 4 tập trung vào ALU.", "Chương 4") == (
            "Tập trung vào ALU."
        )


class TestDemonstrativeSubjectIsDropped:
    """What the model writes when it obeys "do not restate the label"."""

    @pytest.mark.parametrize(
        "body,expected",
        [
            (
                "Chương này cung cấp cái nhìn tổng quan về tổ chức hệ thống.",
                "Cung cấp cái nhìn tổng quan về tổ chức hệ thống.",
            ),
            (
                "Phụ lục này trình bày các nguyên tắc biểu diễn số.",
                "Trình bày các nguyên tắc biểu diễn số.",
            ),
            (
                "Tài liệu này cung cấp thông tin sơ lược về cuốn sách.",
                "Cung cấp thông tin sơ lược về cuốn sách.",
            ),
            (
                "Nội dung này đi sâu vào cấu trúc bộ nhớ.",
                "Đi sâu vào cấu trúc bộ nhớ.",
            ),
        ],
    )
    def test_a_demonstrative_opener_is_removed(self, body, expected):
        assert open_with_substance(body, "Chương 3") == expected


class TestLeavesAlone:
    def test_a_different_unit_is_a_real_cross_reference(self):
        body = "Chương 8 cũng bàn về xử lý song song."

        assert open_with_substance(body, "Chương 2") == body

    def test_a_label_in_the_middle_is_untouched(self):
        body = "Nội dung mở rộng ý của Chương 2 về tổ chức hệ thống."

        assert open_with_substance(body, "Chương 2") == body

    def test_body_that_starts_with_substance_is_unchanged(self):
        body = "Cấp độ logic số nằm ở ranh giới giữa khoa học máy tính và kỹ thuật điện."

        assert open_with_substance(body, "Chương 3") == body

    def test_a_copula_remainder_keeps_its_subject(self):
        """ "Là phần Danh mục tài liệu tham khảo." is not a sentence."""
        body = "Chương này là phần Danh mục tài liệu tham khảo."

        assert open_with_substance(body, "Chương 9") == body

    def test_the_same_guard_applies_to_a_restated_label(self):
        assert open_with_substance("Chương 9 là phần Danh mục.", "Chương 9") == (
            "Chương này là phần Danh mục."
        )

    def test_empty_inputs_do_not_crash(self):
        assert open_with_substance("", "Chương 1") == ""
        assert open_with_substance(None, "Chương 1") == ""
        assert open_with_substance("Chương này trình bày X.", "") == "Trình bày X."


class TestRealRegression:
    def test_the_n4_11_160_appendix_a_line(self):
        body = (
            "Phụ lục A (Приложение А) trình bày về các số nhị phân (двоичные числа) "
            "và hệ đếm cơ số 16."
        )

        out = open_with_substance(body, "Phụ lục A")

        assert out.startswith("Trình bày về các số nhị phân")
        assert "Приложение А" not in out, "phần nhắc lại nhãn gốc cũng phải đi theo"

    def test_the_n4_11_160_chapter_2_line(self):
        body = "Chương này cung cấp cái nhìn tổng quan về tổ chức của các hệ thống máy tính."

        assert open_with_substance(body, "Chương 2").startswith("Cung cấp cái nhìn")


class TestInlineMarkup:
    """The model writes markdown inline, and the renderer strips it downstream.

    Capitalising the first character blindly upper-cases the asterisk of
    `**nhấn mạnh**` and leaves the word lower-case, so the entry opened in
    lower case once the markup was cleaned.
    """

    def test_the_word_is_capitalised_not_the_asterisk(self):
        assert open_with_substance("Chương này **nhấn mạnh** kiến trúc.", "Chương 1") == (
            "**Nhấn mạnh** kiến trúc."
        )

    def test_a_copula_behind_markup_still_keeps_its_subject(self):
        body = "Chương này *là* phần Danh mục."

        assert open_with_substance(body, "Chương 9") == body

    def test_markup_with_nothing_after_it_is_left_alone(self):
        body = "Chương này **"

        assert open_with_substance(body, "Chương 1") == body
