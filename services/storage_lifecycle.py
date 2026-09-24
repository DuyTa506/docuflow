"""Document object-lifecycle helpers: prefix delete, orphan scan, user cleanup.

MinIO is the canonical blob store. PostgreSQL only holds keys. Deleting a
document or user without this module leaves page images, exports and
translation caches behind until the disk fills up.
"""

from __future__ import annotations

import logging

from data.db_models import Document
from services.object_storage import get_object_storage
from utils.storage_keys import document_prefix

logger = logging.getLogger(__name__)


def cleanup_document_artifacts(document_id: str) -> None:
    """Best-effort MinIO + export cache removal. Never raises."""
    try:
        from services.export_service import export_service

        export_service.invalidate_document(document_id)
    except Exception:
        logger.warning("Export invalidate failed for %s", document_id, exc_info=True)
    try:
        get_object_storage().delete_prefix(document_prefix(document_id))
    except Exception:
        logger.warning("MinIO prefix delete failed for %s", document_id, exc_info=True)


def list_document_ids(db) -> set[str]:
    return {row[0] for row in db.query(Document.id).all()}


def list_orphan_prefixes(db) -> list[str]:
    """Object prefixes under documents/ whose document row no longer exists."""
    storage = get_object_storage()
    known = list_document_ids(db)
    orphans: list[str] = []
    seen: set[str] = set()
    for key in storage.list_keys("documents/"):
        parts = key.split("/")
        if len(parts) < 2:
            continue
        doc_id = parts[1]
        if not doc_id or doc_id in known or doc_id in seen:
            continue
        seen.add(doc_id)
        orphans.append(document_prefix(doc_id))
    return orphans


def delete_orphan_prefixes(db, *, apply: bool = False) -> list[str]:
    orphans = list_orphan_prefixes(db)
    if apply:
        storage = get_object_storage()
        for prefix in orphans:
            storage.delete_prefix(prefix)
            logger.info("Deleted orphan prefix %s", prefix)
    return orphans
