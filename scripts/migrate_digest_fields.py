#!/usr/bin/env python3
"""Add bibliographic_metadata and usage_scope JSON columns to documents."""
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
        if not column_exists(conn, "documents", "bibliographic_metadata"):
            conn.execute(text(
                "ALTER TABLE documents ADD COLUMN bibliographic_metadata JSON"
            ))
            print("  + documents.bibliographic_metadata")
        else:
            print("  - documents.bibliographic_metadata exists")

        if not column_exists(conn, "documents", "usage_scope"):
            conn.execute(text(
                "ALTER TABLE documents ADD COLUMN usage_scope JSON"
            ))
            print("  + documents.usage_scope")
        else:
            print("  - documents.usage_scope exists")

        if not column_exists(conn, "document_keywords", "display"):
            conn.execute(text(
                "ALTER TABLE document_keywords ADD COLUMN display TEXT"
            ))
            print("  + document_keywords.display")
        else:
            print("  - document_keywords.display exists")

        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
