"""Dòng §2.2 của mẫu là `Chương 1. Giới thiệu (Введение).` — tiêu đề gốc trong
ngoặc **không** kèm "Глава 1.".

Chạy thật trên N4.11.160 cho ra `Chương 1. Глава 1. Введение.`: tiền tố cấu trúc
bị in hai lần, một lần tiếng Việt một lần tiếng Nga. Tách nó ra là bước đầu;
dịch phần còn lại là bước sau.
"""

import pytest

from core.spatial.zone_classifier import split_chapter_heading


class TestStrip:
    @pytest.mark.parametrize(
        "raw,rest",
        [
            ("Глава 1. Введение", "Введение"),
            ("Глава 2. Организация компьютерных систем", "Организация компьютерных систем"),
            ("Приложение А. Двоичные числа", "Двоичные числа"),
            ("Chương 3. Bộ lọc số", "Bộ lọc số"),
            ("Chapter 12 Introduction", "Introduction"),
            ("Phụ lục B. Bảng tra cứu", "Bảng tra cứu"),
        ],
    )
    def test_prefix_is_removed(self, raw, rest):
        assert split_chapter_heading(raw)[1] == rest

    @pytest.mark.parametrize(
        "raw,kind,ordinal",
        [
            ("Глава 4. Уровень микроархитектуры", "chapter", 4),
            ("Приложение А. Двоичные числа", "appendix", 1),
            ("Приложение В. Программирование", "appendix", 3),
        ],
    )
    def test_kind_and_ordinal_come_back_too(self, raw, kind, ordinal):
        matched = split_chapter_heading(raw)[0]

        assert matched is not None
        assert (matched[0], matched[2]) == (kind, ordinal)


class TestLeaveAlone:
    @pytest.mark.parametrize(
        "raw",
        [
            "Введение",
            "Digital Filters for Audio Applications",
            "",
        ],
    )
    def test_a_title_without_a_structural_prefix_is_untouched(self, raw):
        matched, rest = split_chapter_heading(raw)

        assert matched is None
        assert rest == raw.strip()

    def test_a_label_only_title_has_no_name_left(self):
        """«Приложение Б.» chỉ có nhãn, không có tên.

        Giữ lại cả chuỗi thì render ra `Phụ lục B. Phụ lục B. (Приложение Б.)` —
        đã thấy trên N4.11.160. Nhãn được dựng lại từ kind/ordinal trả về, nên
        phần tên rỗng mới là câu trả lời đúng.
        """
        matched, rest = split_chapter_heading("Приложение Б.")

        assert rest == ""
        assert matched is not None and matched[0] == "appendix"
