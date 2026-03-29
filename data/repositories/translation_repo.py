"""
Translation repository — queries for Translation model.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import Translation


class TranslationRepository:
    """CRUD + query methods for Translation. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, translation_id: str, document_id: str) -> Optional[Translation]:
        """Return a single translation by ID scoped to a document."""
        return (
            self.db.query(Translation)
            .filter(
                Translation.id == translation_id,
                Translation.document_id == document_id,
            )
            .first()
        )

    def list(self, document_id: str) -> List[Translation]:
        """Return all translations for a document."""
        return (
            self.db.query(Translation)
            .filter(Translation.document_id == document_id)
            .all()
        )

    def get_latest(self, document_id: str, lang: str) -> Optional[Translation]:
        """Return the most recent translation for a given language."""
        return (
            self.db.query(Translation)
            .filter(
                Translation.document_id == document_id,
                Translation.target_language == lang,
            )
            .order_by(Translation.created_at.desc())
            .first()
        )

    def update(self, translation_id: str, content: str) -> None:
        """Update translated_content and set status to PENDING_REVIEW."""
        t = self.db.query(Translation).filter(Translation.id == translation_id).first()
        if t:
            t.translated_content = content
            t.status = "PENDING_REVIEW"
            self.db.flush()
