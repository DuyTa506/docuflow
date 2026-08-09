"""An appendix entry that calls itself "chương này" states a falsehood.

Observed on N4.11.160, in both appendix entries that had a real summary:

    Phụ lục B. … Phụ lục này trình bày các nguyên tắc … **Chương này** phân
    tích sự khác biệt … **Chương** cũng thảo luận về việc chuẩn hóa …

The unit kind is known — the renderer already prints "Phụ lục B" from it — so
nothing here is guessed. Only *self-references* are corrected: a word carrying
an ordinal ("Chương 5 đã trình bày") is a genuine cross-reference to a real
chapter and must survive untouched inside an appendix.

The follower list is deliberately closed rather than open. "chương" opens
"chương trình" (program), which is on almost every page of a computer
architecture book; requiring one of a few demonstrative/adverbial markers makes
that collision impossible instead of merely unlikely.
"""

import pytest

from utils.digest_format import correct_unit_kind_words


class TestAppendixCallingItselfAChapter:
    @pytest.mark.parametrize(
        "body,expected",
        [
            ("Chương này phân tích sự khác biệt.", "Phụ lục này phân tích sự khác biệt."),
            ("chương này phân tích.", "phụ lục này phân tích."),
            ("Chương cũng thảo luận về chuẩn hóa.", "Phụ lục cũng thảo luận về chuẩn hóa."),
            ("Chương còn mô tả quy trình.", "Phụ lục còn mô tả quy trình."),
        ],
    )
    def test_a_self_reference_is_corrected(self, body, expected):
        assert correct_unit_kind_words(body, "appendix") == expected

    def test_every_occurrence_is_corrected(self):
        body = "Chương này giải thích. Chương cũng mô tả. Chương này nhấn mạnh."

        assert correct_unit_kind_words(body, "appendix").count("Phụ lục") == 3
        assert "Chương" not in correct_unit_kind_words(body, "appendix")


class TestLeavesAlone:
    def test_a_chapter_calling_itself_a_chapter_is_correct(self):
        body = "Chương này trình bày tổ chức hệ thống."

        assert correct_unit_kind_words(body, "chapter") == body

    def test_a_cross_reference_with_an_ordinal_survives(self):
        """Inside an appendix, "Chương 5" still means chapter 5 of the book."""
        body = "Chương 5 đã trình bày tập lệnh; chương này mở rộng."

        out = correct_unit_kind_words(body, "appendix")

        assert out.startswith("Chương 5 đã trình bày")
        assert out.endswith("phụ lục này mở rộng.")

    @pytest.mark.parametrize(
        "body",
        [
            "Chương trình được viết bằng hợp ngữ.",
            "Nội dung chương trình mô phỏng bộ vi xử lý 8088.",
            "Các chương mục được đánh số liên tục.",
        ],
    )
    def test_a_compound_word_is_never_touched(self, body):
        assert correct_unit_kind_words(body, "appendix") == body

    def test_an_unknown_or_missing_kind_changes_nothing(self):
        body = "Chương này trình bày."

        assert correct_unit_kind_words(body, None) == body
        assert correct_unit_kind_words(body, "") == body

    def test_empty_input_is_safe(self):
        assert correct_unit_kind_words("", "appendix") == ""
        assert correct_unit_kind_words(None, "appendix") == ""


class TestRealRegression:
    def test_the_n4_11_160_appendix_b_line(self):
        body = (
            "Phụ lục này trình bày các nguyên tắc về biểu diễn số dấu phẩy động. "
            "Chương này phân tích sự khác biệt giữa số thực và số dấu phẩy động. "
            "Chương cũng thảo luận về việc chuẩn hóa."
        )

        out = correct_unit_kind_words(body, "appendix")

        assert "Chương" not in out and "chương" not in out
        # One "Phụ lục này" was already correct; the two chapter words join it.
        assert out.count("Phụ lục") == 3 and out.count("Phụ lục này") == 2
