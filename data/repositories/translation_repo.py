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
        """Return one translation per target_language (most recent)."""
        rows = (
            self.db.query(Translation)
            .filter(Translation.document_id == document_id)
            .order_by(Translation.target_language, Translation.created_at.desc())
            .all()
        )
        seen: set[str] = set()
        result: List[Translation] = []
        for row in rows:
            if row.target_language in seen:
                continue
            seen.add(row.target_language)
            result.append(row)
        return result

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
            t.translated_elements = None
            t.translated_file_path = None
            t.translation_mode = None
            t.status = "PENDING_REVIEW"
            self.db.commit()
            self.db.refresh(t)
        return t
