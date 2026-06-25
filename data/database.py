"""
Database connection and session management.

Provides utilities for creating database engine, sessions, and table initialization.

Session factory for FastAPI dependency injection:
    Use ``api.dependencies.get_db`` — the canonical FastAPI dependency.
    Do NOT use the removed ``data.database.get_db`` (deleted — was duplicate).
"""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .db_models import Base


# Default database path (can be overridden via environment variable)
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "document_store.db",
)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL

        # Create engine
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            self.engine = create_engine(self.database_url, echo=False)

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        self._migrate_schema()
        print(f"Database tables created at: {self.database_url}")

    def _migrate_schema(self):
        """Add columns to existing SQLite DBs without dropping data."""
        if not self.database_url.startswith("sqlite"):
            return
        from sqlalchemy import inspect, text

        insp = inspect(self.engine)
        if "translations" not in insp.get_table_names():
            return
        existing = {c["name"] for c in insp.get_columns("translations")}
        additions = {
            "translated_file_path": "VARCHAR",
            "translated_elements": "TEXT",
            "translation_mode": "VARCHAR",
        }
        with self.engine.begin() as conn:
            for col, col_type in additions.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE translations ADD COLUMN {col} {col_type}"))

        if "digitized_texts" in insp.get_table_names():
            dt_cols = {c["name"] for c in insp.get_columns("digitized_texts")}
            if "text_overridden" not in dt_cols:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE digitized_texts ADD COLUMN text_overridden BOOLEAN DEFAULT 0"
                        )
                    )

        if "pages" in insp.get_table_names():
            page_cols = {c["name"] for c in insp.get_columns("pages")}
            if "page_type" not in page_cols:
                with self.engine.begin() as conn:
                    conn.execute(
                        text("ALTER TABLE pages ADD COLUMN page_type VARCHAR")
                    )

        self._ensure_indexes(                    )

    def _ensure_indexes(self):
        """Create composite indexes idempotently (SQLite)."""
        if not self.database_url.startswith("sqlite"):
            return
        from sqlalchemy import inspect, text

        indexes = {
            "ix_documents_user_created": "CREATE INDEX IF NOT EXISTS ix_documents_user_created ON documents (user_id, created_at)",
            "ix_pages_document_page_number": "CREATE INDEX IF NOT EXISTS ix_pages_document_page_number ON pages (document_id, page_number)",
            "ix_layout_elements_page_sequence": "CREATE INDEX IF NOT EXISTS ix_layout_elements_page_sequence ON layout_elements (page_id, sequence_order)",
            "ix_layout_elements_page_label": "CREATE INDEX IF NOT EXISTS ix_layout_elements_page_label ON layout_elements (page_id, label)",
            "ix_tasks_document_type_status_created": "CREATE INDEX IF NOT EXISTS ix_tasks_document_type_status_created ON tasks (document_id, task_type, status, created_at)",
        }
        insp = inspect(self.engine)
        existing = set()
        for table in insp.get_table_names():
            for idx in insp.get_indexes(table):
                existing.add(idx.get("name"))

        with self.engine.begin() as conn:
            for name, ddl in indexes.items():
                if name not in existing:
                    conn.execute(text(ddl))

    def drop_tables(self):
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(bind=self.engine)
        print(f"Database tables dropped from: {self.database_url}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions (auto-commit / rollback)."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def seed_sequences(self):
        """Seed the id_sequences table after tables are created."""
        from .id_generator import IdGenerator

        with self.session() as session:
            IdGenerator.seed_sequences(session)
        print("ID sequences seeded.")


# ── Global singleton ───────────────────────────────────────────────

_db_manager = None


def get_db_manager(database_url: str = None) -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def init_database(database_url: str = None):
    """Initialize database — create tables + seed sequences."""
    db_manager = get_db_manager(database_url)
    db_manager.create_tables()
    db_manager.seed_sequences()


def get_session() -> Session:
    """Get a new database session from global manager."""
    return get_db_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager using global manager."""
    with get_db_manager().session() as session:
        yield session
# NOTE: The FastAPI get_db() dependency lives in api/dependencies.py.
# A duplicate async def get_db() that previously existed here has been removed
# (Phase 8 cleanup). Use api.dependencies.get_db for all FastAPI route injection.
