"""Tests for translation_elements helpers."""

from utils.translation_elements import (
    deserialize_translated_elements,
    flatten_translated_elements,
    serialize_translated_elements,
)


class TestTranslationElements:
    def test_flatten_joins_with_blank_lines(self):
        elements = [
            {"text_content": "Block one"},
            {"text_content": "Block two"},
        ]
        assert flatten_translated_elements(elements) == "Block one\n\nBlock two"

    def test_serialize_roundtrip(self):
        data = [{"label": "text", "text_content": "A", "page_number": 1}]
        raw = serialize_translated_elements(data)
        assert deserialize_translated_elements(raw) == data
