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

    def update(self, translation_id: str, document_id: str, content: str) -> Optional[Translation]:
        """Overwrite translated_content with user-uploaded correction."""
        t = (
            self.db.query(Translation)
            .filter(
                Translation.id == translation_id,
                Translation.document_id == document_id,
            )
            .first()
        )
        if t:
            t.translated_content = content
            self.db.commit()
            self.db.refresh(t)
        return t

    def save_result(
        self,
        translation_id: str,
        document_id: str,
        *,
        translated_content: str | None = None,
        translated_file_path: str | None = None,
        translated_elements: str | None = None,
        translation_mode: str | None = None,
        status: str = "COMPLETED",
    ) -> Optional[Translation]:
        """Persist structured translation output."""
        t = (
            self.db.query(Translation)
            .filter(
                Translation.id == translation_id,
                Translation.document_id == document_id,
            )
            .first()
        )
        if not t:
            return None
        if translated_content is not None:
            t.translated_content = translated_content
        if translated_file_path is not None:
            t.translated_file_path = translated_file_path
        if translated_elements is not None:
            t.translated_elements = translated_elements
        if translation_mode is not None:
            t.translation_mode = translation_mode
        t.status = status
        self.db.commit()
        self.db.refresh(t)
        return t
