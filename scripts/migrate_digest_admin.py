#!/usr/bin/env python3
"""Add documents.digest_admin — the "Thông tin quản trị CSDL" block.

Three librarian-entered fields (reviewer, reviewer_approved, entry_date). They
were accepted by the renderer but had nowhere to live, so the block rendered
empty on every export.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from data.database import DatabaseManager


def column_exists(conn, table: str, column: str) -> bool:
    insp = inspect(conn)
    return column in {c["name"] for c in insp.get_columns(table)}


def main():
    dbm = DatabaseManager()
    with dbm.engine.connect() as conn:
        if not column_exists(conn, "documents", "digest_admin"):
            conn.execute(text("ALTER TABLE documents ADD COLUMN digest_admin JSON"))
            print("  + documents.digest_admin")
        else:
            print("  - documents.digest_admin exists")

        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
