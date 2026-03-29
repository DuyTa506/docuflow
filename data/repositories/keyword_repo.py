"""
Keyword repository — queries for Keyword and DocumentKeyword models.
"""
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from data.db_models import Keyword, DocumentKeyword


class KeywordRepository:
    """CRUD + query methods for keywords. No business logic."""

    def __init__(self, db: Session):
        self.db = db

    def get_for_document(
        self, document_id: str
    ) -> List[Tuple[DocumentKeyword, Keyword]]:
        """Return (DocumentKeyword, Keyword) pairs ordered by weight desc."""
        return (
            self.db.query(DocumentKeyword, Keyword)
            .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
            .filter(DocumentKeyword.document_id == document_id)
            .order_by(DocumentKeyword.weight.desc())
            .all()
        )

    def find_or_create(self, name: str) -> Keyword:
        """Return existing Keyword by name or create a new one."""
        kw = (
            self.db.query(Keyword)
            .filter(Keyword.keyword_name == name)
            .first()
        )
        if not kw:
            kw = Keyword(keyword_name=name)
            self.db.add(kw)
            self.db.flush()
        return kw
