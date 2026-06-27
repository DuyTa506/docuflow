"""Read/write large text blobs with optional MinIO offload."""

from __future__ import annotations

from typing import Optional

from config.settings import settings
from utils.storage_keys import normalized_content_key, ocr_content_key


def get_object_storage():
    """Lazy accessor for the MinIO storage client.

    ``services.object_storage`` is imported on call rather than at module load:
    ``utils`` is a lower layer than ``services``, and importing the services
    package eagerly here creates a circular import
    (utils.content_storage -> services.object_storage -> services/__init__
    -> services.base_service -> utils.content_storage). Keeping this as a
    module-level name also preserves ``patch("utils.content_storage.get_object_storage")``.
    """
    from services.object_storage import get_object_storage as _get_object_storage

    return _get_object_storage()


def maybe_offload_text(
    doc_id: str,
    *,
    field: str,
    content: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return (db_content, content_key).

    When content exceeds threshold, store in MinIO and return (None, key).
    Otherwise return (content, None).
    """
    if not content:
        return None, None
    threshold = settings.text_offload_threshold_chars
    if len(content) <= threshold:
        return content, None

    storage = get_object_storage()
    key = (
        normalized_content_key(doc_id)
        if field == "normalized"
        else ocr_content_key(doc_id)
    )
    storage.put_bytes(key, content.encode("utf-8"), content_type="text/markdown; charset=utf-8")
    return None, key


def read_text_field(
    *,
    inline: Optional[str],
    key: Optional[str],
) -> str:
    """Read inline DB text or fetch from MinIO when offloaded."""
    if inline:
        return inline
    if key:
        storage = get_object_storage()
        if storage.exists(key):
            return storage.get_bytes(key).decode("utf-8")
    return ""
