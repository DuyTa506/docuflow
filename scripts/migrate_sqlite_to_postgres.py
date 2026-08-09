#!/usr/bin/env python3
"""
Migrate data from legacy SQLite document_store.db to PostgreSQL.

Run after Phase A image migration for a smaller source DB.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, inspect  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from data.database import DEFAULT_DB_PATH, DatabaseManager  # noqa: E402
from data.db_models import Base  # noqa: E402

TABLE_ORDER = [
    "id_sequences",
    "users",
    "documents",
    "pages",
    "layout_elements",
    "digitized_texts",
    "translations",
    "summaries",
    "main_contents",
    "keywords",
    "document_keywords",
    "keyword_extractions",
    "research_directions",
    "document_research_directions",
    "research_extractions",
    "tree_indices",
    "tree_nodes",
    "tasks",
]

SKIP_COLUMNS = {
    "pages": {"image_base64"},
    "layout_elements": {"crop_image_base64"},
}


def _sqlite_url(path: str) -> str:
    return f"sqlite:///{path}"


def copy_table(src_sess, dst_sess, table_name: str, *, dry_run: bool) -> int:
    mapper = Base.metadata.tables[table_name]
    rows = src_sess.execute(mapper.select()).mappings().all()
    skip_cols = SKIP_COLUMNS.get(table_name, set())
    count = 0
    for row in rows:
        payload = dict(row)
        for col in skip_cols:
            if payload.get(f"{col.replace('_base64', '_key')}") and payload.get(col):
                payload[col] = None
        if dry_run:
            count += 1
            continue
        dst_sess.execute(mapper.insert().values(**payload))
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default=DEFAULT_DB_PATH,
        help="Path to source SQLite DB",
    )
    parser.add_argument(
        "--postgres-url",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://docuflow:docuflow@localhost:5433/docuflow",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path(args.sqlite_path).is_file():
        print(f"SQLite file not found: {args.sqlite_path}")
        return 1

    src_engine = create_engine(_sqlite_url(args.sqlite_path))
    dst_manager = DatabaseManager(args.postgres_url)
    dst_manager.create_tables()
    dst_sess_maker = sessionmaker(bind=dst_manager.engine)

    src_sess = sessionmaker(bind=src_engine)()
    dst_sess = dst_sess_maker()

    try:
        totals: dict[str, int] = {}
        for table in TABLE_ORDER:
            if table not in inspect(src_engine).get_table_names():
                continue
            n = copy_table(src_sess, dst_sess, table, dry_run=args.dry_run)
            totals[table] = n
            print(f"{table}: {n} rows")
        if not args.dry_run:
            dst_sess.commit()
    except Exception:
        dst_sess.rollback()
        raise
    finally:
        src_sess.close()
        dst_sess.close()

    print(f"\nMigration complete. dry_run={args.dry_run}")
    print(totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
