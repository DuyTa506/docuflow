#!/usr/bin/env python3
"""
Deduplicate translation rows: keep one row per (document_id, target_language).

Priority: COMPLETED > PENDING_REVIEW > IN_PROGRESS > PENDING > FAILED, then newest.
Also removes orphaned MinIO objects for deleted translation IDs.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.database import get_db_manager  # noqa: E402
from data.db_models import Translation  # noqa: E402
from services.object_storage import get_object_storage  # noqa: E402
from utils.storage_keys import translation_file_key  # noqa: E402

_STATUS_RANK = {
    "COMPLETED": 0,
    "PENDING_REVIEW": 1,
    "IN_PROGRESS": 2,
    "PENDING": 3,
    "FAILED": 4,
}


def _pick_winner(rows: list[Translation]) -> Translation:
    return sorted(
        rows,
        key=lambda t: (
            _STATUS_RANK.get(t.status, 99),
            -(t.created_at.timestamp() if t.created_at else 0),
        ),
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate translation rows")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not delete")
    args = parser.parse_args()

    db_manager = get_db_manager()
    storage = get_object_storage()

    with db_manager.session() as db:
        all_rows = db.query(Translation).order_by(Translation.created_at.desc()).all()
        groups: dict[tuple[str, str], list[Translation]] = defaultdict(list)
        for row in all_rows:
            groups[(row.document_id, row.target_language)].append(row)

        removed = 0
        for (_doc_id, _lang), rows in groups.items():
            if len(rows) <= 1:
                continue
            keeper = _pick_winner(rows)
            for row in rows:
                if row.id == keeper.id:
                    continue
                print(
                    f"Remove duplicate {row.id} ({row.document_id} → {row.target_language}, {row.status})"
                )
                if not args.dry_run:
                    for ext in ("docx", "pdf"):
                        storage.delete(translation_file_key(row.document_id, row.id, ext))
                    db.delete(row)
                    removed += 1

        if not args.dry_run:
            db.commit()

    print(f"Done. Removed {removed} duplicate translation row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
