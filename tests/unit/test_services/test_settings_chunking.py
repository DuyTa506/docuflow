from config.settings import Settings, lang_name, normalize_lang_code


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


class TestInputBudget:
    def test_reserves_output_tokens(self):
        s = Settings(
            ai_model_context_window=16384, ai_chunk_ratio=0.85, ai_output_reserve_tokens=3000
        )
        assert s.ai_input_budget_tokens == s.ai_chunk_tokens - 3000

    def test_floors_at_one_for_tiny_context_window(self):
        s = Settings(
            ai_model_context_window=1000, ai_chunk_ratio=0.5, ai_output_reserve_tokens=3000
        )
        assert s.ai_input_budget_tokens == 1

    def test_default_reserve_is_3000(self):
        s = Settings()
        assert s.ai_output_reserve_tokens == 3000


class TestLangName:
    def test_known_codes(self):
        assert lang_name("vi") == "Vietnamese"
        assert lang_name("en") == "English"
        assert lang_name("zh") == "Chinese"
        assert lang_name("ru") == "Russian"

    def test_case_insensitive(self):
        assert lang_name("VI") == "Vietnamese"
        assert lang_name("En") == "English"

    def test_unknown_code_passthrough(self):
        assert lang_name("ko") == "ko"

    def test_empty_string_fallback(self):
        assert lang_name("") == "the source language"


class TestNormalizeLangCode:
    def test_priority_codes(self):
        assert normalize_lang_code("EN") == "en"
        assert normalize_lang_code("zh-CN") == "zh"
        assert normalize_lang_code("ru-RU") == "ru"

    def test_aliases(self):
        assert normalize_lang_code("english") == "en"
        assert normalize_lang_code("chinese") == "zh"
        assert normalize_lang_code("russian") == "ru"

    def test_auto_defaults_to_en(self):
        assert normalize_lang_code("auto") == "en"
        assert normalize_lang_code("") == "en"
