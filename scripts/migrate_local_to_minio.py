#!/usr/bin/env python3
"""
Migrate legacy local file paths to MinIO object keys.

- documents.file_path: ./uploads/... or uploads/<doc_id>/... → documents/{id}/original/{filename}
- translations.translated_file_path: ./uploads/translations/... → documents/{doc_id}/translations/{id}.{ext}

Skips rows already using MinIO keys (documents/...). Missing local files are reported and left unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from data.database import get_db_manager  # noqa: E402
from data.db_models import Document, Translation  # noqa: E402
from services.object_storage import get_object_storage  # noqa: E402
from utils.storage_keys import original_key, translation_file_key  # noqa: E402


def _resolve_local_path(path: str) -> str | None:
    if not path or path.startswith("documents/"):
        return None
    candidates = [
        path,
        os.path.join(ROOT, path),
        os.path.normpath(path),
        os.path.normpath(os.path.join(ROOT, path.lstrip("./"))),
        os.path.join(settings.upload_dir, os.path.basename(path)),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def migrate_documents(db, storage, *, dry_run: bool) -> tuple[int, int, int]:
    uploaded = skipped = missing = 0
    docs = db.query(Document).all()
    for doc in docs:
        path = doc.file_path
        if not path:
            continue
        if storage.is_object_key(path):
            skipped += 1
            continue
        local = _resolve_local_path(path)
        if not local:
            print(f"[MISSING] {doc.id}: {path}")
            missing += 1
            continue
        filename = os.path.basename(local)
        key = original_key(doc.id, filename)
        if storage.exists(key):
            print(f"[EXISTS]  {doc.id}: {key}")
        else:
            print(f"[UPLOAD]  {doc.id}: {local} → {key}")
            if not dry_run:
                storage.put_file(key, local)
        if not dry_run:
            doc.file_path = key
        uploaded += 1
    return uploaded, skipped, missing


def migrate_translations(db, storage, *, dry_run: bool) -> tuple[int, int, int]:
    uploaded = skipped = missing = 0
    rows = db.query(Translation).filter(Translation.translated_file_path.isnot(None)).all()
    for row in rows:
        path = row.translated_file_path
        if storage.is_object_key(path):
            skipped += 1
            continue
        local = _resolve_local_path(path)
        if not local:
            print(f"[MISSING] translation {row.id}: {path}")
            missing += 1
            continue
        ext = Path(local).suffix.lstrip(".") or "bin"
        key = translation_file_key(row.document_id, row.id, ext)
        if storage.exists(key):
            print(f"[EXISTS]  translation {row.id}: {key}")
        else:
            print(f"[UPLOAD]  translation {row.id}: {local} → {key}")
            if not dry_run:
                storage.put_file(key, local)
        if not dry_run:
            row.translated_file_path = key
        uploaded += 1
    return uploaded, skipped, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate local upload paths to MinIO")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not upload or update DB")
    args = parser.parse_args()

    storage = get_object_storage()
    storage.ensure_bucket()
    print(f"MinIO bucket: {storage.bucket}")

    db_manager = get_db_manager()
    with db_manager.session() as db:
        doc_up, doc_skip, doc_miss = migrate_documents(db, storage, dry_run=args.dry_run)
        tr_up, tr_skip, tr_miss = migrate_translations(db, storage, dry_run=args.dry_run)
        if not args.dry_run:
            db.commit()

    print(
        f"\nDocuments: uploaded/updated={doc_up}, already_minio={doc_skip}, missing={doc_miss}"
    )
    print(
        f"Translations: uploaded/updated={tr_up}, already_minio={tr_skip}, missing={tr_miss}"
    )
    if args.dry_run:
        print("(dry-run — no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
