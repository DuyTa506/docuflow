from utils.digest_format import format_title_display

ORIGINAL = "Ballistics: Theory and Design of Guns and Ammunition"


def test_english_source_shows_the_original_then_the_vietnamese():
    out = format_title_display(
        ORIGINAL, "Đạn đạo học: Lý thuyết và thiết kế súng, đạn dược", ORIGINAL, "en"
    )

    assert out == f"{ORIGINAL} (Đạn đạo học: Lý thuyết và thiết kế súng, đạn dược)"


def test_other_languages_add_the_english_after_the_vietnamese():
    out = format_title_display(
        "Введение в теорию", "Giới thiệu lý thuyết", "Introduction to the theory", "ru"
    )

    assert out == "Введение в теорию (Giới thiệu lý thuyết/Introduction to the theory)"


def test_vietnamese_source_is_the_title_alone():
    assert (
        format_title_display("Giáo dục song ngữ", "Giáo dục song ngữ", "Bilingual education", "vi")
        == "Giáo dục song ngữ"
    )


def test_a_missing_translation_never_prints_empty_brackets():
    assert format_title_display(ORIGINAL, "", "", "en") == ORIGINAL
    assert format_title_display("Введение", "Giới thiệu", "", "ru") == "Введение (Giới thiệu)"


def test_a_translation_equal_to_the_original_is_not_repeated():
    assert format_title_display(ORIGINAL, ORIGINAL, ORIGINAL, "en") == ORIGINAL


class TestKeywordDisplay:
    def test_english_source_is_vietnamese_then_the_term(self):
        from utils.digest_format import format_keyword_display

        assert (
            format_keyword_display("Shaped charge", "Đạn lõm", "Shaped charge", "en")
            == "Đạn lõm (Shaped charge)"
        )

    def test_other_source_adds_the_english_after_the_original(self):
        from utils.digest_format import format_keyword_display

        out = format_keyword_display(
            "Система наведения", "Hệ thống dẫn hướng", "Guidance System", "ru"
        )

        assert out == "Hệ thống dẫn hướng (Система наведения/Guidance System)"

    def test_the_english_is_not_repeated_when_it_equals_the_term(self):
        from utils.digest_format import format_keyword_display

        assert format_keyword_display("V2X", "Kết nối V2X", "V2X", "zh") == "Kết nối V2X (V2X)"

    def test_vietnamese_source_is_the_term_alone(self):
        from utils.digest_format import format_keyword_display

        assert format_keyword_display("dẫn hướng", "dẫn hướng", "guidance", "vi") == "dẫn hướng"

    def test_no_vietnamese_means_no_display_so_the_caller_keeps_the_model_one(self):
        from utils.digest_format import format_keyword_display

        assert format_keyword_display("term", "", "term", "en") == ""
