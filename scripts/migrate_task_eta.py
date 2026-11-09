"""Apply the additive task ETA schema and SSE trigger idempotently."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import init_database  # noqa: E402


def main() -> None:
    # DatabaseManager.create_tables creates the observation/profile tables,
    # adds columns to existing task tables, creates indexes, and installs the
    # PostgreSQL NOTIFY trigger. Every operation is safe to repeat.
    init_database()
    print("Task ETA schema and notification trigger installed")


if __name__ == "__main__":
    main()
