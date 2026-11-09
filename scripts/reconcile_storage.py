#!/usr/bin/env python3
"""Find (and optionally delete) MinIO prefixes with no matching document row."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.database import get_db_manager
from services.storage_lifecycle import delete_orphan_prefixes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete orphan prefixes. Default is a dry run.",
    )
    args = parser.parse_args()
    with get_db_manager().session() as db:
        orphans = delete_orphan_prefixes(db, apply=args.apply)
    if not orphans:
        print("No orphan document prefixes.")
        return 0
    action = "Deleted" if args.apply else "Would delete"
    print(f"{action} {len(orphans)} prefix(es):")
    for prefix in orphans:
        print(f"  {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
