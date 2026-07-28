#!/usr/bin/env python3
"""Add documents.digest_doc_kind — "book" | "proceedings" | NULL (auto).

The digest template defines two modes (§2.1 and §2.2 both branch on it), but
there was nowhere to record which one a document is. Detection reads the title
and front matter only, so it can be wrong; this column is how a librarian
overrides it.
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
        if not column_exists(conn, "documents", "digest_doc_kind"):
            conn.execute(text("ALTER TABLE documents ADD COLUMN digest_doc_kind VARCHAR"))
            print("  + documents.digest_doc_kind")
        else:
            print("  - documents.digest_doc_kind exists")

        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
