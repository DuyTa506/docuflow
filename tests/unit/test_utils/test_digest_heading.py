"""Ba dạng dòng §2.2 mà mẫu chính thức quy định.

Book:
    "Chương 1. Giới thiệu (Introduction). Giải thích bối cảnh…"

Kỷ yếu — theo Cụm vấn đề:
    "Khoa học máy tính (Computer Science), gồm 5 BBKH. Các nghiên cứu về…"

Kỷ yếu — theo Từng BBKH đơn lẻ:
    "BBKH 1 - Giám sát chất lượng nước… (Real-Time Water Quality Monitoring…). …"

Tách thành hàm thuần để khỏi phải render .docx mới kiểm được cách ghép chuỗi.
"""

import pytest

from utils.digest_format import chapter_heading


class TestBookForm:
    def test_bilingual_title(self):
        assert (
            chapter_heading(1, "Giới thiệu", "Introduction", doc_kind="book")
            == "Chương 1. Giới thiệu (Introduction)."
        )

    def test_no_original_leaves_no_empty_parentheses(self):
        assert chapter_heading(2, "Giới thiệu", "", doc_kind="book") == "Chương 2. Giới thiệu."

    def test_identical_titles_are_not_printed_twice(self):
        assert chapter_heading(3, "Введение", "Введение", doc_kind="book") == "Chương 3. Введение."

    def test_book_is_the_default_kind(self):
        assert chapter_heading(1, "Giới thiệu", "Introduction").startswith("Chương 1.")


class TestProceedingsClusterForm:
    def test_cluster_states_how_many_papers(self):
        assert (
            chapter_heading(
                1, "Khoa học máy tính", "Computer Science", doc_kind="proceedings", paper_count=5
            )
            == "Khoa học máy tính (Computer Science), gồm 5 BBKH."
        )

    def test_cluster_has_no_chuong_prefix(self):
        heading = chapter_heading(2, "Viễn thông", "", doc_kind="proceedings", paper_count=3)

        assert heading == "Viễn thông, gồm 3 BBKH."
        assert "Chương" not in heading


class TestProceedingsSinglePaperForm:
    def test_single_paper_is_numbered_as_bbkh(self):
        assert (
            chapter_heading(
                1,
                "Giám sát chất lượng nước theo thời gian thực",
                "Real-Time Water Quality Monitoring",
                doc_kind="proceedings",
            )
            == "BBKH 1 - Giám sát chất lượng nước theo thời gian thực "
            "(Real-Time Water Quality Monitoring)."
        )

    @pytest.mark.parametrize("count", [None, 0, 1])
    def test_a_cluster_of_one_is_just_a_paper(self, count):
        """`gồm 1 BBKH` là cách nói vô nghĩa — rơi về dạng đơn lẻ."""
        heading = chapter_heading(4, "Bài báo", "", doc_kind="proceedings", paper_count=count)

        assert heading == "BBKH 4 - Bài báo."


class TestDegenerateInput:
    def test_empty_title_still_yields_a_usable_label(self):
        assert chapter_heading(7, "", "", doc_kind="book") == "Chương 7."
        assert chapter_heading(7, "", "", doc_kind="proceedings") == "BBKH 7."
