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

    monkeypatch.setattr("langdetect.detect_langs", _raise)
    assert detect_source_language("xyzzy plugh qwerty asdfgh zxcvbn", fallback="vi") == "vi"


def test_normalizes_regional_variant(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "langdetect.detect_langs", lambda text: [SimpleNamespace(lang="zh-cn", prob=0.99)]
    )
    assert detect_source_language("some latin text long enough here", fallback="en") == "zh"


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


class TestScriptAwareDetection:
    """Live regression (DOC_009): a Chinese book detected as "vi" and every
    translate request was refused as same-language. Samples are real pages."""

    ZH_COVER = (
        "微电子与集成电路技术丛书\n国家集成电路人才培养基地专家指导委员会组编\n\n\n\n\n"
        "韩郑生 编著\nHan Zhengsheng\n\n赵元富 主审\nZhao Yuanfu\n\n清华大学出版社"
    )
    ZH_BODY = (
        "## 6.1 PDSOI 80C51 微控制器的系统架构\n\n80C51 微控制器是 Intel 公司 MCS-51 "
        "8-bit 系列微控制器（单片机）中的核心电路，该系列中的所有产品都具有 80C51 的基本"
        "结构和软件特征。"
    )
    ZH_FORM = (
        "## 教师反馈表\n\n感谢您购买本书！\n\n<table><tr><td>您需要教辅的教材:</td>"
        '<td colspan="3">抗辐射集成电路概论(韩郑生)</td></tr><tr><td>您的姓名:</td>'
        '<td colspan="3"></td></tr></table>'
    )

    def test_chinese_book_sample_detects_zh(self):
        sample = sample_representative_text([self.ZH_COVER, self.ZH_BODY, self.ZH_FORM])
        assert detect_source_language(sample, fallback="vi") == "zh"

    def test_each_chinese_page_detects_zh(self):
        for page in (self.ZH_COVER, self.ZH_BODY, self.ZH_FORM):
            assert detect_source_language(page, fallback="vi") == "zh"

    def test_russian_with_markup_detects_ru(self):
        text = (
            "## Глава 1\n\n<table><tr><td>Таблица 1.1</td></tr></table>\n"
            "Микропроцессорные системы применяются для управления объектами $x^2$."
        )
        assert detect_source_language(text, fallback="vi") == "ru"

    def test_markup_only_falls_back(self):
        text = "<table><tr><td>1</td><td>2.5</td></tr></table>\n---\n$$x^2 + y^2$$"
        assert detect_source_language(text, fallback="vi") == "vi"

    def test_low_confidence_keeps_fallback(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            "langdetect.detect_langs",
            lambda text: [
                SimpleNamespace(lang="vi", prob=0.55),
                SimpleNamespace(lang="en", prob=0.45),
            ],
        )
        text = "Some Latin-script text long enough to be considered for detection here."
        assert detect_source_language(text, fallback="en") == "en"
