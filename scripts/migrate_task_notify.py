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

from sqlalchemy import text

from data.database import get_db_manager, init_database

TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION notify_task_change() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('task_events', json_build_object(
    'task_id', NEW.id,
    'document_id', NEW.document_id,
    'task_type', NEW.task_type,
    'status', NEW.status,
    'progress', NEW.progress,
    'message', LEFT(COALESCE(NEW.message, ''), 500),
    'updated_at', to_char(NEW.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
  )::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS task_notify ON tasks;
CREATE TRIGGER task_notify
AFTER INSERT OR UPDATE ON tasks
FOR EACH ROW EXECUTE FUNCTION notify_task_change();
"""


def main() -> None:
    init_database()
    db_manager = get_db_manager()
    with db_manager.session() as db:
        if db.bind.dialect.name != "postgresql":
            print("Skipped: task notify trigger requires PostgreSQL")
            return
        db.execute(text(TRIGGER_SQL))
        db.commit()
    print("task_events trigger installed")


if __name__ == "__main__":
    main()
