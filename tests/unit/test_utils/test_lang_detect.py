from utils.lang_detect import detect_source_language


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
