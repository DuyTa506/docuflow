from utils.lang_detect import detect_source_language, sample_representative_text


def test_detects_vietnamese():
    text = "Đây là một đoạn văn bản tiếng Việt để kiểm tra khả năng nhận diện ngôn ngữ."
    assert detect_source_language(text) == "vi"


def test_detects_english():
    text = "This is a sample paragraph of English text used to test language detection."
    assert detect_source_language(text) == "en"


def test_empty_text_falls_back():
    assert detect_source_language("", fallback="vi") == "vi"
    assert detect_source_language("   ", fallback="en") == "en"
    assert detect_source_language(None, fallback="fr") == "fr"


def test_ambiguous_short_text_falls_back(monkeypatch):
    import utils.lang_detect as mod

    def _raise(*a, **k):
        from langdetect import LangDetectException

        raise LangDetectException(0, "No features in text")

    monkeypatch.setattr("langdetect.detect", _raise)
    assert detect_source_language("123 456 !@#", fallback="vi") == "vi"


def test_normalizes_regional_variant(monkeypatch):
    monkeypatch.setattr("langdetect.detect", lambda text: "zh-cn")
    assert detect_source_language("some text", fallback="en") == "zh"


class TestSampleRepresentativeText:
    def test_picks_first_middle_last_pages(self):
        pages = [f"page {i}" for i in range(10)]
        sample = sample_representative_text(pages, span_chars=20)
        assert "page 0" in sample
        assert "page 5" in sample
        assert "page 9" in sample
        assert "page 1" not in sample

    def test_skips_empty_pages(self):
        pages = ["", "  ", "real content here"]
        sample = sample_representative_text(pages, span_chars=50)
        assert sample == "real content here"

    def test_single_page(self):
        assert sample_representative_text(["only page"], span_chars=50) == "only page"

    def test_no_pages_returns_empty(self):
        assert sample_representative_text([]) == ""

    def test_vietnamese_front_matter_over_non_vietnamese_body_detects_body_language(self):
        """Regression: a Vietnamese cover/stamp page over a Russian body must
        detect Russian, not Vietnamese — the exact DOC_059 bug."""
        vi_cover = "Đây là trang bìa thư viện. Số hiệu đăng ký N4.11.162."
        ru_body_page = (
            "Компьютерное зрение представляет собой область искусственного "
            "интеллекта, занимающуюся анализом и обработкой цифровых "
            "изображений для распознавания образов и объектов."
        )
        pages = [vi_cover] + [ru_body_page] * 20
        sample = sample_representative_text(pages)
        detected = detect_source_language(sample, fallback="en")
        assert detected == "ru"
