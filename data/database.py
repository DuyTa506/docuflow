"""
Database connection and session management.

Provides utilities for creating database engine, sessions, and table initialization.
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
    'document_store.db'
)

DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DEFAULT_DB_PATH}')


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL. If None, uses DATABASE_URL env var
                         or default SQLite path.
        """
        self.database_url = database_url or DATABASE_URL
        
        # Create engine
        # For SQLite, enable foreign key constraints and use check_same_thread=False for async
        if self.database_url.startswith('sqlite'):
            self.engine = create_engine(
                self.database_url,
                connect_args={'check_same_thread': False},
                echo=False  # Set to True for SQL query logging
            )
        else:
            self.engine = create_engine(self.database_url, echo=False)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        print(f"✓ Database tables created at: {self.database_url}")
    
    def drop_tables(self):
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(bind=self.engine)
        print(f"✗ Database tables dropped from: {self.database_url}")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Remember to close the session when done:
            session = db_manager.get_session()
            try:
                # ... use session ...
            finally:
                session.close()
        
        Or use the session context manager instead.
        """
        return self.SessionLocal()
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Automatically commits on success and rolls back on exception.
        
        Usage:
            with db_manager.session() as session:
                # ... use session ...
                # Automatically commits when exiting normally
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database manager instance
_db_manager = None


def get_db_manager(database_url: str = None) -> DatabaseManager:
    """
    Get or create global database manager instance.
    
    Args:
        database_url: Optional database URL. Only used on first call.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def init_database(database_url: str = None):
    """
    Initialize database by creating all tables.
    
    Args:
        database_url: Optional database URL
    """
    db_manager = get_db_manager(database_url)
    db_manager.create_tables()


def get_session() -> Session:
    """
    Get a new database session from global manager.
    
    Returns:
        SQLAlchemy Session
    """
    return get_db_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions using global manager.
    
    Usage:
        from serving.database import session_scope
        
        with session_scope() as session:
            doc = session.query(Document).first()
            # ... use session ...
    """
    with get_db_manager().session() as session:
        yield session


# Async support for FastAPI dependency injection
async def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints.
    
    Usage:
        @app.get("/documents/{doc_id}")
        async def get_document(doc_id: str, db: Session = Depends(get_db)):
            doc = db.query(Document).filter(Document.id == doc_id).first()
            return doc
    """
    with session_scope() as session:
        yield session
