"""
Research repository — queries for ResearchDirection, DocumentResearchDirection,
and ResearchExtraction.
"""
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from data.db_models import (
    ResearchDirection,
    DocumentResearchDirection,
    ResearchExtraction,
)


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

    # Backward-compat aliases used by some routers
    def get_by_name(self, name: str) -> Optional[ResearchDirection]:
        return self.find_catalog_by_name(name)

    # ── Extraction job tracking ─────────────────────────────────────

    def list_extractions(self, document_id: str) -> List[ResearchExtraction]:
        """Return all research-extraction jobs for a document, newest first."""
        return (
            self.db.query(ResearchExtraction)
            .filter(ResearchExtraction.document_id == document_id)
            .order_by(ResearchExtraction.created_at.desc())
            .all()
        )

    def get_extraction(
        self, extraction_id: str, document_id: str
    ) -> Optional[ResearchExtraction]:
        return (
            self.db.query(ResearchExtraction)
            .filter(
                ResearchExtraction.id == extraction_id,
                ResearchExtraction.document_id == document_id,
            )
            .first()
        )

    def get_latest_extraction(self, document_id: str) -> Optional[ResearchExtraction]:
        return (
            self.db.query(ResearchExtraction)
            .filter(ResearchExtraction.document_id == document_id)
            .order_by(ResearchExtraction.created_at.desc())
            .first()
        )
