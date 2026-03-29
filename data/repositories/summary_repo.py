"""
Summary repository — queries for Summary model.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import Summary


class SummaryRepository:
    """CRUD + query methods for Summary. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get_latest(self, document_id: str, summary_type: str) -> Optional[Summary]:
        """Return the most recent summary of a given type for a document."""
        return (
            self.db.query(Summary)
            .filter(
                Summary.document_id == document_id,
                Summary.summary_type == summary_type,
            )
            .order_by(Summary.created_at.desc())
            .first()
        )

    def list(self, document_id: str) -> List[Summary]:
        """Return all summaries for a document."""
        return (
            self.db.query(Summary)
            .filter(Summary.document_id == document_id)
            .all()
        )
