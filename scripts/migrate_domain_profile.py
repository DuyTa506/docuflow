#!/usr/bin/env python3
"""Add documents.domain_profile and documents.termbase (DOMAIN_PROFILE stage).

The profile names the document's field / discipline / specialty; the termbase
pins each recurring term to one Vietnamese rendering for every digest stage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from data.database import DatabaseManager

COLUMNS = ("domain_profile", "termbase")


def main():
    dbm = DatabaseManager()
    with dbm.engine.connect() as conn:
        existing = {c["name"] for c in inspect(conn).get_columns("documents")}
        for column in COLUMNS:
            if column in existing:
                print(f"  - documents.{column} exists")
                continue
            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {column} JSON"))
            print(f"  + documents.{column}")
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
