#!/usr/bin/env python3
"""Add pipeline mirror columns to documents for Temporal digest workflow UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from data.database import DatabaseManager

_COLUMNS = [
    ("pipeline_workflow_id", "VARCHAR"),
    ("pipeline_state", "VARCHAR DEFAULT 'IDLE'"),
    ("pipeline_stage", "VARCHAR"),
    ("pipeline_progress", "INTEGER DEFAULT 0"),
    ("pipeline_message", "VARCHAR"),
    ("quality_report", "JSON"),
]


def column_exists(conn, table: str, column: str) -> bool:
    insp = inspect(conn)
    return column in {c["name"] for c in insp.get_columns(table)}


def main():
    dbm = DatabaseManager()
    with dbm.engine.connect() as conn:
        for name, col_type in _COLUMNS:
            if not column_exists(conn, "documents", name):
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {col_type}"))
                print(f"  + documents.{name}")
            else:
                print(f"  - documents.{name} exists")
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
