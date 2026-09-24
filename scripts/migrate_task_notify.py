"""Install the pg_notify trigger powering the SSE /api/v2/events stream.

Every INSERT/UPDATE on tasks emits a NOTIFY on channel 'task_events' with a
JSON payload — a DB-level trigger catches ALL writers (uvicorn API and the
Temporal worker are separate processes; app-level hooks would miss one).

Idempotent: re-running replaces the function/trigger.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import get_db_manager, init_database  # noqa: E402
from data.task_notify import install_task_notify  # noqa: E402


def main() -> None:
    init_database()
    db_manager = get_db_manager()
    with db_manager.session() as db:
        if db.bind.dialect.name != "postgresql":
            print("Skipped: task notify trigger requires PostgreSQL")
            return
        install_task_notify(db)
        db.commit()
    print("task_events trigger installed")


if __name__ == "__main__":
    main()
