"""Tests for centralized pipeline output language clauses."""

from config.settings import (
    PIPELINE_OUTPUT_LANG,
    pipeline_keyword_lang_clause,
    pipeline_output_lang_clause,
)


class TestPipelineOutputLangClause:
    def test_default_lang_is_vietnamese(self):
        assert PIPELINE_OUTPUT_LANG == "vi"

    def test_prose_clause_mentions_vietnamese(self):
        clause = pipeline_output_lang_clause()
        assert "Vietnamese" in clause
        assert "OUTPUT LANGUAGE" in clause

    def test_json_clause_mentions_vietnamese(self):
        clause = pipeline_output_lang_clause(json_values=True)
        assert "Vietnamese" in clause
        assert "string values" in clause

    def test_keyword_clause_preserves_source_language(self):
        clause = pipeline_keyword_lang_clause()
        assert "verbatim" in clause
        assert "same language" in clause
        assert "Do NOT translate" in clause
