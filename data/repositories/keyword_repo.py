"""
Keyword repository — queries for Keyword, DocumentKeyword, and KeywordExtraction.
"""

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from data.db_models import DocumentKeyword, Keyword, KeywordExtraction


class KeywordRepository:
    """CRUD + query methods for keywords. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get_for_document(self, document_id: str) -> List[Tuple[DocumentKeyword, Keyword]]:
        """Return (DocumentKeyword, Keyword) pairs ordered by weight desc."""
        return (
            self.db.query(DocumentKeyword, Keyword)
            .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
            .filter(DocumentKeyword.document_id == document_id)
            .order_by(DocumentKeyword.weight.desc())
            .all()
        )

    # ── Extraction job tracking ─────────────────────────────────────

    def list_extractions(self, document_id: str) -> List[KeywordExtraction]:
        """Return all keyword-extraction jobs for a document, newest first."""
        return (
            self.db.query(KeywordExtraction)
            .filter(KeywordExtraction.document_id == document_id)
            .order_by(KeywordExtraction.created_at.desc())
            .all()
        )

    def get_extraction(self, extraction_id: str, document_id: str) -> Optional[KeywordExtraction]:
        return (
            self.db.query(KeywordExtraction)
            .filter(
                KeywordExtraction.id == extraction_id,
                KeywordExtraction.document_id == document_id,
            )
            .first()
        )

    def get_latest_extraction(self, document_id: str) -> Optional[KeywordExtraction]:
        return (
            self.db.query(KeywordExtraction)
            .filter(KeywordExtraction.document_id == document_id)
            .order_by(KeywordExtraction.created_at.desc())
            .first()
        )
