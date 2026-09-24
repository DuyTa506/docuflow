"""PostgreSQL NOTIFY trigger used by the authenticated task SSE stream."""

from sqlalchemy import text

# PostgreSQL NOTIFY payloads are limited to 8000 bytes. Build a public,
# allow-listed payload and degrade to the task identity/status if structured
# progress ever grows unexpectedly large.
TASK_NOTIFY_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION notify_task_change() RETURNS trigger AS $$
DECLARE
  payload jsonb;
BEGIN
  payload := jsonb_strip_nulls(jsonb_build_object(
    'task_id', NEW.id,
    'document_id', NEW.document_id,
    'task_type', NEW.task_type,
    'status', NEW.status,
    'progress', NEW.progress,
    'message', LEFT(COALESCE(NEW.message, ''), 500),
    'started_at', CASE WHEN NEW.started_at IS NULL THEN NULL
      ELSE to_char(NEW.started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END,
    'completed_at', CASE WHEN NEW.completed_at IS NULL THEN NULL
      ELSE to_char(NEW.completed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END,
    'progress_meta', NEW.progress_meta,
    'eta', NEW.eta,
    'created_at', CASE WHEN NEW.created_at IS NULL THEN NULL
      ELSE to_char(NEW.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END,
    'updated_at', CASE WHEN NEW.updated_at IS NULL THEN NULL
      ELSE to_char(NEW.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END
  ));

  IF octet_length(payload::text) > 7600 THEN
    payload := jsonb_strip_nulls(jsonb_build_object(
      'task_id', NEW.id,
      'document_id', NEW.document_id,
      'task_type', NEW.task_type,
      'status', NEW.status,
      'progress', NEW.progress,
      'message', LEFT(COALESCE(NEW.message, ''), 300),
      'started_at', CASE WHEN NEW.started_at IS NULL THEN NULL
        ELSE to_char(NEW.started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END,
      'completed_at', CASE WHEN NEW.completed_at IS NULL THEN NULL
        ELSE to_char(NEW.completed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END,
      'progress_meta', jsonb_strip_nulls(jsonb_build_object(
        'version', NEW.progress_meta->'version',
        'pipeline', NEW.progress_meta->'pipeline',
        'phase', NEW.progress_meta->'phase',
        'mode', NEW.progress_meta->'mode',
        'stage', NEW.progress_meta->'stage',
        'unit_kind', NEW.progress_meta->'unit_kind',
        'units_done', NEW.progress_meta->'units_done',
        'units_total', NEW.progress_meta->'units_total',
        'attempt', NEW.progress_meta->'attempt'
      )),
      'eta', NEW.eta,
      'updated_at', CASE WHEN NEW.updated_at IS NULL THEN NULL
        ELSE to_char(NEW.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END
    ));
  END IF;

  PERFORM pg_notify('task_events', payload::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS task_notify ON tasks;
CREATE TRIGGER task_notify
AFTER INSERT OR UPDATE ON tasks
FOR EACH ROW EXECUTE FUNCTION notify_task_change();
"""


def install_task_notify(conn) -> None:
    """Install or replace the trigger in an existing transaction."""

    conn.execute(text(TASK_NOTIFY_TRIGGER_SQL))
