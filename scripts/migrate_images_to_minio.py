#!/usr/bin/env python3
"""
Migrate page and crop base64 blobs from SQLite/Postgres to MinIO.

1. pages.image_base64 → image_key
2. layout_elements.crop_image_base64 → crop_image_key

Clears legacy base64 columns after successful upload and verification.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import joinedload  # noqa: E402

from data.database import get_db_manager  # noqa: E402
from data.db_models import LayoutElement, Page  # noqa: E402
from services.object_storage import get_object_storage  # noqa: E402
from utils.storage_keys import layout_crop_key, page_image_key  # noqa: E402


def migrate_pages(db, storage, *, dry_run: bool) -> tuple[int, int, int]:
    migrated = skipped = failed = 0
    pages = db.query(Page).filter(Page.image_base64.isnot(None), Page.image_base64 != "").all()
    for page in pages:
        if page.image_key and storage.exists(page.image_key):
            skipped += 1
            continue
        try:
            data = base64.b64decode(page.image_base64)
            key = page_image_key(page.document_id, page.page_number)
            print(f"[PAGE] {page.document_id} p{page.page_number} → {key} ({len(data)} bytes)")
            if not dry_run:
                storage.put_bytes(key, data, content_type="image/jpeg")
                if not storage.exists(key):
                    raise RuntimeError(f"upload verify failed: {key}")
                page.image_key = key
                page.image_base64 = None
            migrated += 1
        except Exception as exc:
            print(f"[FAIL] page {page.id}: {exc}")
            failed += 1
    return migrated, skipped, failed


def migrate_crops(db, storage, *, dry_run: bool) -> tuple[int, int, int]:
    migrated = skipped = failed = 0
    elements = (
        db.query(LayoutElement)
        .options(joinedload(LayoutElement.page))
        .filter(
            LayoutElement.crop_image_base64.isnot(None),
            LayoutElement.crop_image_base64 != "",
        )
        .all()
    )
    for elem in elements:
        if elem.crop_image_key and storage.exists(elem.crop_image_key):
            skipped += 1
            continue
        page = elem.page
        if not page:
            failed += 1
            continue
        try:
            data = base64.b64decode(elem.crop_image_base64)
            seq = elem.sequence_order if elem.sequence_order is not None else 0
            key = layout_crop_key(page.document_id, page.page_number, seq)
            print(f"[CROP] {page.document_id} p{page.page_number} #{seq} → {key}")
            if not dry_run:
                storage.put_bytes(key, data, content_type="image/jpeg")
                if not storage.exists(key):
                    raise RuntimeError(f"upload verify failed: {key}")
                elem.crop_image_key = key
                elem.crop_image_base64 = None
            migrated += 1
        except Exception as exc:
            print(f"[FAIL] element {elem.id}: {exc}")
            failed += 1
    return migrated, skipped, failed


def vacuum_sqlite(db_manager) -> None:
    if not db_manager.is_sqlite:
        return
    from sqlalchemy import text

    with db_manager.engine.connect() as conn:
        conn.execute(text("VACUUM"))
        conn.commit()
    print("SQLite VACUUM completed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate page/crop base64 to MinIO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vacuum", action="store_true", help="Run SQLite VACUUM after migrate")
    args = parser.parse_args()

    storage = get_object_storage()
    storage.ensure_bucket()

    db_manager = get_db_manager()
    with db_manager.session() as db:
        p_m, p_s, p_f = migrate_pages(db, storage, dry_run=args.dry_run)
        c_m, c_s, c_f = migrate_crops(db, storage, dry_run=args.dry_run)
        if not args.dry_run:
            db.commit()

    print(
        f"\nPages: migrated={p_m}, skipped={p_s}, failed={p_f}\n"
        f"Crops: migrated={c_m}, skipped={c_s}, failed={c_f}"
    )
    if args.dry_run:
        print("(dry-run — no changes written)")
    elif args.vacuum:
        vacuum_sqlite(db_manager)
    return 0 if (p_f + c_f) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
