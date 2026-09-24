from sqlalchemy import inspect, text

from data.database import DatabaseManager
from data.task_notify import TASK_NOTIFY_TRIGGER_SQL


def test_fresh_database_has_eta_tables_columns_and_indexes(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path / 'fresh.db'}")
    manager.create_tables()
    inspector = inspect(manager.engine)
    task_columns = {column["name"] for column in inspector.get_columns("tasks")}
    assert {
        "started_at",
        "completed_at",
        "progress_meta",
        "eta",
        "eta_estimator_state",
    } <= task_columns
    assert {"task_eta_observations", "task_eta_profiles"} <= set(inspector.get_table_names())
    profile_indexes = {index["name"] for index in inspector.get_indexes("task_eta_profiles")}
    assert "ix_task_eta_profiles_lookup" in profile_indexes


def test_existing_task_table_is_migrated_idempotently(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path / 'existing.db'}")
    with manager.engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE tasks (
                    id VARCHAR PRIMARY KEY,
                    document_id VARCHAR,
                    task_type VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message VARCHAR,
                    result JSON,
                    error TEXT,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
    manager.create_tables()
    manager.create_tables()
    columns = {column["name"] for column in inspect(manager.engine).get_columns("tasks")}
    assert "progress_meta" in columns
    assert "eta_estimator_state" in columns


def test_notify_trigger_contains_structured_contract_and_size_guard():
    assert "'progress_meta', NEW.progress_meta" in TASK_NOTIFY_TRIGGER_SQL
    assert "'eta', NEW.eta" in TASK_NOTIFY_TRIGGER_SQL
    assert "octet_length(payload::text) > 7600" in TASK_NOTIFY_TRIGGER_SQL
    assert "DROP TRIGGER IF EXISTS task_notify" in TASK_NOTIFY_TRIGGER_SQL
