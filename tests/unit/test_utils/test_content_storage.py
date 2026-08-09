"""Tests for text offload helpers."""

from unittest.mock import MagicMock, patch

from utils.content_storage import maybe_offload_text, read_text_field


def test_maybe_offload_text_below_threshold():
    inline, key = maybe_offload_text("DOC_1", field="normalized", content="short")
    assert inline == "short"
    assert key is None


@patch("utils.content_storage.get_object_storage")
@patch("utils.content_storage.settings")
def test_maybe_offload_text_above_threshold(mock_settings, mock_storage):
    mock_settings.text_offload_threshold_chars = 5
    storage = MagicMock()
    mock_storage.return_value = storage

    inline, key = maybe_offload_text("DOC_1", field="normalized", content="x" * 20)
    assert inline is None
    assert key is not None
    storage.put_bytes.assert_called_once()


@patch("utils.content_storage.get_object_storage")
def test_read_text_field_from_minio(mock_storage):
    storage = MagicMock()
    storage.exists.return_value = True
    storage.get_bytes.return_value = b"from minio"
    mock_storage.return_value = storage

    assert read_text_field(inline=None, key="documents/DOC_1/content/normalized.md") == "from minio"
