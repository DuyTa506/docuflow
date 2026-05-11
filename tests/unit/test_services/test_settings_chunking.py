from config.settings import Settings, lang_name


class TestChunkTokens:
    def test_default_values(self):
        s = Settings(ai_model_context_window=128000, ai_chunk_ratio=0.85)
        assert s.ai_chunk_tokens == int(128000 * 0.85)

    def test_custom_window_and_ratio(self):
        s = Settings(ai_model_context_window=32000, ai_chunk_ratio=0.5)
        assert s.ai_chunk_tokens == 16000

    def test_minimum_one(self):
        s = Settings(ai_model_context_window=0, ai_chunk_ratio=0.0)
        assert s.ai_chunk_tokens == 1

    def test_default_output_langs(self):
        s = Settings()
        assert s.summary_output_lang == "vi"
        assert s.research_output_lang == "vi"


class TestLangName:
    def test_known_codes(self):
        assert lang_name("vi") == "Vietnamese"
        assert lang_name("en") == "English"
        assert lang_name("zh") == "Chinese"

    def test_case_insensitive(self):
        assert lang_name("VI") == "Vietnamese"
        assert lang_name("En") == "English"

    def test_unknown_code_passthrough(self):
        assert lang_name("ko") == "ko"

    def test_empty_string_fallback(self):
        assert lang_name("") == "the source language"
