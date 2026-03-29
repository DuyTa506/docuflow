"""
Document repository — all DB queries for Document, Page, LayoutElement, DigitizedText.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import Document, Page, LayoutElement, DigitizedText


class DocumentRepository:
    """CRUD + query methods for Document aggregate. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    # ── Document ────────────────────────────────────────────────────

    def get(self, document_id: str) -> Optional[Document]:
        """Return document by ID or None."""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list(self, limit: int = 50, offset: int = 0) -> List[Document]:
        """Paginated list of ALL documents, newest first (admin use)."""
        return (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def list_for_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Document]:
        """Paginated list of documents belonging to a specific user, newest first."""
        return (
            self.db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def list_for_librarians(
        self, limit: int = 50, offset: int = 0
    ) -> List[Document]:
        """
        Return all documents uploaded by LIBRARIAN users.
        Used by admins to review librarian uploads.
        """
        from data.db_models import User
        return (
            self.db.query(Document)
            .join(User, Document.user_id == User.id)
            .filter(User.role == "LIBRARIAN")
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    # ── DigitizedText ───────────────────────────────────────────────

    def get_digitized_text(self, document_id: str) -> Optional[DigitizedText]:
        """Return the (first) DigitizedText row for a document, or None."""
        return (
            self.db.query(DigitizedText)
            .filter(DigitizedText.document_id == document_id)
            .first()
        )

    # ── Pages ───────────────────────────────────────────────────────

    def get_pages(self, document_id: str) -> List[Page]:
        """Return all pages for a document ordered by page number."""
        return (
            self.db.query(Page)
            .filter(Page.document_id == document_id)
            .order_by(Page.page_number)
            .all()
        )

    # ── Layout elements ─────────────────────────────────────────────

    def get_elements(
        self, document_id: str, label: Optional[str] = None
    ) -> List[LayoutElement]:
        """Return layout elements for a document, optionally filtered by label."""
        query = (
            self.db.query(LayoutElement)
            .join(Page)
            .filter(Page.document_id == document_id)
        )
        if label:
            query = query.filter(LayoutElement.label == label)
        return query.order_by(Page.page_number, LayoutElement.sequence_order).all()
