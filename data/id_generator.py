"""
Prefixed ID generator for DocuFlow.

Generates IDs like DOC_001, USR_001, TASK_001 using a DB-backed sequence table.
Uses SELECT FOR UPDATE semantics to ensure uniqueness under concurrency.
"""
from sqlalchemy.orm import Session

from data.db_models import IdSequence


# Mapping: table_name → prefix
SEQUENCE_SEEDS = {
    "users":     "USR",
    "documents": "DOC",
    "tasks":     "TASK",
}


class IdGenerator:
    """Thread-safe prefixed ID generator backed by the id_sequences table."""

    @staticmethod
    def next_id(session: Session, table_name: str) -> str:
        """
        Atomically increment the sequence for *table_name* and return the next
        prefixed ID string (e.g. ``DOC_042``).

        For SQLite (no real row-level locking) we rely on the Python GIL +
        serialised transactions; for PostgreSQL this uses ``SELECT ... FOR UPDATE``.
        """
        seq = (
            session.query(IdSequence)
            .filter(IdSequence.table_name == table_name)
            .with_for_update()
            .first()
        )
        if seq is None:
            raise ValueError(
                f"No ID sequence configured for table '{table_name}'. "
                f"Run IdGenerator.seed_sequences() first."
            )

        seq.current_value += 1
        session.flush()
        return f"{seq.prefix}_{seq.current_value:03d}"

    @staticmethod
    def seed_sequences(session: Session) -> None:
        """
        Insert initial seed rows into ``id_sequences`` for every table listed in
        :data:`SEQUENCE_SEEDS`.  Existing rows are left untouched (idempotent).
        """
        for table_name, prefix in SEQUENCE_SEEDS.items():
            existing = (
                session.query(IdSequence)
                .filter(IdSequence.table_name == table_name)
                .first()
            )
            if existing is None:
                session.add(
                    IdSequence(
                        table_name=table_name,
                        prefix=prefix,
                        current_value=0,
                    )
                )
        session.commit()
