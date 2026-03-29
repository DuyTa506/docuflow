"""
Research repository — queries for ResearchDirection and DocumentResearchDirection.
"""
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from data.db_models import ResearchDirection, DocumentResearchDirection


class ResearchRepository:
    """CRUD + query methods for research directions. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get_directions(
        self, document_id: str
    ) -> List[Tuple[DocumentResearchDirection, ResearchDirection]]:
        """Return (DocumentResearchDirection, ResearchDirection) pairs for a document."""
        return (
            self.db.query(DocumentResearchDirection, ResearchDirection)
            .join(
                ResearchDirection,
                DocumentResearchDirection.direction_id == ResearchDirection.id,
            )
            .filter(DocumentResearchDirection.document_id == document_id)
            .order_by(DocumentResearchDirection.confidence.desc())
            .all()
        )

    def get_catalog(self) -> List[ResearchDirection]:
        """Return all predefined research directions, alphabetically."""
        return (
            self.db.query(ResearchDirection)
            .order_by(ResearchDirection.direction_name)
            .all()
        )

    def add_catalog(self, name: str) -> ResearchDirection:
        """Add a new predefined direction. Caller must check for duplicates."""
        rd = ResearchDirection(direction_name=name, is_predefined=True)
        self.db.add(rd)
        self.db.flush()
        self.db.refresh(rd)
        return rd

    def find_catalog_by_name(self, name: str) -> Optional[ResearchDirection]:
        """Return existing direction by name or None."""
        return (
            self.db.query(ResearchDirection)
            .filter(ResearchDirection.direction_name == name)
            .first()
        )
