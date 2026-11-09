"""
MainContent repository — queries for the MainContent model.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import MainContent


class MainContentRepository:
    """CRUD + query methods for MainContent. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, main_content_id: str, document_id: str) -> Optional[MainContent]:
        return (
            self.db.query(MainContent)
            .filter(
                MainContent.id == main_content_id,
                MainContent.document_id == document_id,
            )
            .first()
        )

    def list(self, document_id: str) -> List[MainContent]:
        """Return all main content extractions for a document, newest first."""
        return (
            self.db.query(MainContent)
            .filter(MainContent.document_id == document_id)
            .order_by(MainContent.created_at.desc())
            .all()
        )
