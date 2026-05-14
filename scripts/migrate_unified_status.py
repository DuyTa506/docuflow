#!/usr/bin/env python3
"""
Migration: add unified job-status columns and extraction job-tracker tables.

Idempotent — safe to run multiple times.

Schema changes:
  1. ALTER TABLE summaries        ADD COLUMN status TEXT NOT NULL DEFAULT 'COMPLETED'
  2. ALTER TABLE main_contents    ADD COLUMN status TEXT NOT NULL DEFAULT 'COMPLETED'
  3. CREATE TABLE keyword_extractions (...)
  4. CREATE TABLE research_extractions (...)
  5. UPDATE translations SET status='COMPLETED' WHERE status='PENDING_REVIEW'

Existing summaries/main_contents rows are marked COMPLETED because if they
exist they have content, i.e. the past run did succeed.
"""
import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from data.database import DatabaseManager, DEFAULT_DB_PATH


KEYWORD_EXTRACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS keyword_extractions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'PENDING',
    max_keywords INTEGER NOT NULL DEFAULT 20,
    total_keywords INTEGER,
    error TEXT,
    created_at DATETIME,
    updated_at DATETIME
)
"""

RESEARCH_EXTRACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS research_extractions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'PENDING',
    total_directions INTEGER,
    error TEXT,
    created_at DATETIME,
    updated_at DATETIME
)
"""


def column_exists(conn, table: str, column: str) -> bool:
    insp = inspect(conn)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def table_exists(conn, table: str) -> bool:
    insp = inspect(conn)
    return table in insp.get_table_names()


def main():
    parser = argparse.ArgumentParser(description="Migrate DocuFlow DB to unified job-status schema")
    parser.add_argument(
        "--database-url",
        default=None,
        help=f"Database URL (default: sqlite:///{DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    dbm = DatabaseManager(args.database_url)
    print(f"Migrating: {dbm.database_url}")

    with dbm.engine.begin() as conn:
        # 1. summaries.status
        if not column_exists(conn, "summaries", "status"):
            conn.execute(text(
                "ALTER TABLE summaries ADD COLUMN status TEXT NOT NULL DEFAULT 'COMPLETED'"
            ))
            print("  + Added column summaries.status")
        else:
            print("  - summaries.status already exists")

        # 2. main_contents.status
        if not column_exists(conn, "main_contents", "status"):
            conn.execute(text(
                "ALTER TABLE main_contents ADD COLUMN status TEXT NOT NULL DEFAULT 'COMPLETED'"
            ))
            print("  + Added column main_contents.status")
        else:
            print("  - main_contents.status already exists")

        # 3. keyword_extractions
        if not table_exists(conn, "keyword_extractions"):
            conn.execute(text(KEYWORD_EXTRACTIONS_DDL))
            print("  + Created table keyword_extractions")
        else:
            print("  - keyword_extractions already exists")

        # 4. research_extractions
        if not table_exists(conn, "research_extractions"):
            conn.execute(text(RESEARCH_EXTRACTIONS_DDL))
            print("  + Created table research_extractions")
        else:
            print("  - research_extractions already exists")

        # 5. Normalize legacy translation status
        result = conn.execute(text(
            "UPDATE translations SET status='COMPLETED' WHERE status='PENDING_REVIEW'"
        ))
        if result.rowcount:
            print(f"  + Renamed {result.rowcount} translations PENDING_REVIEW -> COMPLETED")

    print("Migration complete.")


if __name__ == "__main__":
    main()
