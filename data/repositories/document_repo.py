"""
Document repository — all DB queries for Document, Page, LayoutElement, DigitizedText.
"""
from typing import List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from data.db_models import (
    Document,
    Page,
    LayoutElement,
    DigitizedText,
    TreeIndex,
    TreeNode,
    Translation,
    Summary,
    MainContent,
    DocumentKeyword,
    KeywordExtraction,
    DocumentResearchDirection,
    ResearchExtraction,
    Task,
)


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

    def count(self) -> int:
        """Total count of ALL documents (admin use)."""
        return self.db.query(Document).count()

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

    def count_for_user(self, user_id: str) -> int:
        """Total count of documents belonging to a specific user."""
        return self.db.query(Document).filter(Document.user_id == user_id).count()

    def delete_cascade(self, document_id: str) -> bool:
        """Bulk-delete a document and all related rows (fast path for large OCR docs)."""
        self._delete_related_rows(document_id)
        result = self.db.execute(delete(Document).where(Document.id == document_id))
        self.db.commit()
        return result.rowcount > 0

    def clear_extraction_artifacts(self, document_id: str) -> None:
        """Remove pages, layout, digitized text, and tree data without deleting the document."""
        self._delete_extraction_artifacts(document_id)
        self.db.commit()

    def _delete_extraction_artifacts(self, document_id: str) -> None:
        page_ids = select(Page.id).where(Page.document_id == document_id)
        tree_index_ids = select(TreeIndex.id).where(TreeIndex.document_id == document_id)

        self.db.execute(delete(LayoutElement).where(LayoutElement.page_id.in_(page_ids)))
        self.db.execute(delete(Page).where(Page.document_id == document_id))
        self.db.execute(delete(TreeNode).where(TreeNode.tree_index_id.in_(tree_index_ids)))
        self.db.execute(delete(TreeIndex).where(TreeIndex.document_id == document_id))
        self.db.execute(delete(DigitizedText).where(DigitizedText.document_id == document_id))

    def _delete_related_rows(self, document_id: str) -> None:
        self._delete_extraction_artifacts(document_id)
        self.db.execute(delete(Translation).where(Translation.document_id == document_id))
        self.db.execute(delete(Summary).where(Summary.document_id == document_id))
        self.db.execute(delete(MainContent).where(MainContent.document_id == document_id))
        self.db.execute(delete(DocumentKeyword).where(DocumentKeyword.document_id == document_id))
        self.db.execute(delete(KeywordExtraction).where(KeywordExtraction.document_id == document_id))
        self.db.execute(
            delete(DocumentResearchDirection).where(
                DocumentResearchDirection.document_id == document_id
            )
        )
        self.db.execute(
            delete(ResearchExtraction).where(ResearchExtraction.document_id == document_id)
        )
        self.db.execute(delete(Task).where(Task.document_id == document_id))

    # ── DigitizedText ───────────────────────────────────────────────

    def get_digitized_text(self, document_id: str) -> Optional[DigitizedText]:
        """Return the (first) DigitizedText row for a document, or None."""
        return (
            self.db.query(DigitizedText)
            .filter(DigitizedText.document_id == document_id)
            .first()
        )

    def update_digitized_text(self, document_id: str, content: str) -> Optional[DigitizedText]:
        """Overwrite normalized_content with user-uploaded correction."""
        dt = self.get_digitized_text(document_id)
        if dt:
            dt.normalized_content = content
            dt.text_overridden = True
            self.db.commit()
            self.db.refresh(dt)
        return dt

    # ── Pages ───────────────────────────────────────────────────────

    def get_pages(self, document_id: str, limit: Optional[int] = None) -> List[Page]:
        """Return pages for a document ordered by page number.

        Pass `limit` to fetch only the first N pages (preview mode) instead of
        loading the whole document.
        """
        query = (
            self.db.query(Page)
            .filter(Page.document_id == document_id)
            .order_by(Page.page_number)
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def count_pages(self, document_id: str) -> int:
        """Cheap page count for export routing."""
        return int(
            self.db.query(func.count(Page.id))
            .filter(Page.document_id == document_id)
            .scalar()
            or 0
        )

    def count_scanned_pages(self, document_id: str) -> Optional[int]:
        """Return scanned-page count when extraction stored page_type; else None."""
        rows = (
            self.db.query(Page.page_type)
            .filter(Page.document_id == document_id)
            .all()
        )
        if not rows:
            return None
        types = [r[0] for r in rows]
        if all(t is None for t in types):
            return None
        return sum(1 for t in types if t == "scanned")

    # ── Layout elements ─────────────────────────────────────────────

    def count_elements(
        self, document_id: str, label: Optional[str] = None
    ) -> int:
        """Return layout element count for a document (cheap pre-check)."""
        query = (
            self.db.query(func.count(LayoutElement.id))
            .join(Page)
            .filter(Page.document_id == document_id)
        )
        if label:
            query = query.filter(LayoutElement.label == label)
        return int(query.scalar() or 0)

    def get_elements(
        self, document_id: str, label: Optional[str] = None
    ) -> List[LayoutElement]:
        """Return layout elements for a document, optionally filtered by label."""
        query = (
            self.db.query(LayoutElement)
            .join(Page)
            .filter(Page.document_id == document_id)
            .options(joinedload(LayoutElement.page))
        )
        if label:
            query = query.filter(LayoutElement.label == label)
        return query.order_by(Page.page_number, LayoutElement.sequence_order).all()


def delete_document_cascade(document_id: str) -> bool:
    """Thread-safe document delete using a fresh DB session."""
    from data.database import get_db_manager

    db_manager = get_db_manager()
    with db_manager.session() as db:
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            return False
        repo.delete_cascade(document_id)
        return True
