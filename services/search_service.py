"""
Search service.

SQL LIKE/ILIKE search across documents, text content, keywords, and translations.
Returns the same item shape as GET /api/v2/documents (DocumentListItem).
"""
from typing import List, Optional, Set, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from data.db_models import (
    Document, DigitizedText, Keyword, DocumentKeyword, Translation, Task,
)


class SearchService:
    """Full-text search across the document library."""

    def search(
        self,
        db: Session,
        query: str,
        search_in: Optional[List[str]] = None,
        language: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> dict:
        """
        Search across specified fields.

        *search_in* can include: ``title``, ``content``, ``keywords``, ``translations``.
        Defaults to all four.

        Returns ``{"items": [...], "total": int, "query": str}`` where each item
        matches ``DocumentListItem`` (+ optional ``snippet``, ``match_field``).
        """
        if not query or not query.strip():
            return {"items": [], "total": 0, "query": query}

        if search_in is None:
            search_in = ["title", "content", "keywords", "translations"]

        pattern = f"%{query}%"
        matches: List[Tuple[str, str, str]] = []  # (doc_id, snippet, match_field)
        seen_ids: Set[str] = set()
        visible_ids = self._visible_doc_ids(db, user_id, is_admin)

        def _allow(doc_id: str) -> bool:
            return visible_ids is None or doc_id in visible_ids

        # 1. Title search
        if "title" in search_in:
            q = db.query(Document).filter(Document.title.ilike(pattern))
            if visible_ids is not None:
                q = q.filter(Document.user_id == user_id)
            for d in q.all():
                if d.id not in seen_ids and _allow(d.id):
                    matches.append((d.id, d.title, "title"))
                    seen_ids.add(d.id)

        # 2. Content search (normalized text / OCR text)
        if "content" in search_in:
            dts = (
                db.query(DigitizedText)
                .filter(
                    or_(
                        DigitizedText.normalized_content.ilike(pattern),
                        DigitizedText.ocr_content.ilike(pattern),
                    )
                )
                .all()
            )
            for dt in dts:
                if dt.document_id in seen_ids or not _allow(dt.document_id):
                    continue
                doc = db.query(Document).filter(Document.id == dt.document_id).first()
                if not doc:
                    continue
                text = dt.normalized_content or dt.ocr_content or ""
                snippet = self._extract_snippet(text, query)
                matches.append((dt.document_id, snippet, "content"))
                seen_ids.add(dt.document_id)

        # 3. Keyword search
        if "keywords" in search_in:
            kw_q = (
                db.query(DocumentKeyword, Keyword, Document)
                .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
                .join(Document, DocumentKeyword.document_id == Document.id)
                .filter(Keyword.keyword_name.ilike(pattern))
            )
            if visible_ids is not None:
                kw_q = kw_q.filter(Document.user_id == user_id)
            for assoc, kw, doc in kw_q.all():
                if doc.id in seen_ids:
                    continue
                matches.append((
                    doc.id,
                    f"Keyword: {kw.keyword_name} (weight: {assoc.weight:.2f})",
                    "keywords",
                ))
                seen_ids.add(doc.id)

        # 4. Translation search
        if "translations" in search_in:
            trans = (
                db.query(Translation)
                .filter(Translation.translated_content.ilike(pattern))
                .all()
            )
            if language:
                trans = [t for t in trans if t.target_language == language]
            for t in trans:
                if t.document_id in seen_ids or not _allow(t.document_id):
                    continue
                doc = db.query(Document).filter(Document.id == t.document_id).first()
                if not doc:
                    continue
                snippet = self._extract_snippet(t.translated_content or "", query)
                matches.append((t.document_id, snippet, "translations"))
                seen_ids.add(t.document_id)

        total = len(matches)
        page_matches = matches[offset: offset + limit]
        items = self._build_list_items(db, page_matches)

        return {"items": items, "total": total, "query": query}

    @staticmethod
    def _visible_doc_ids(
        db: Session, user_id: Optional[str], is_admin: bool
    ) -> Optional[Set[str]]:
        """Return None when all documents are visible (admin), else allowed doc IDs."""
        if is_admin or not user_id:
            return None
        rows = db.query(Document.id).filter(Document.user_id == user_id).all()
        return {row[0] for row in rows}

    @staticmethod
    def _build_list_items(
        db: Session, matches: List[Tuple[str, str, str]]
    ) -> List[dict]:
        """Hydrate match tuples into DocumentListItem-compatible dicts."""
        if not matches:
            return []

        doc_ids = [doc_id for doc_id, _, _ in matches]
        docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
        doc_map = {d.id: d for d in docs}

        task_summary_map: dict[str, dict[str, str]] = {doc_id: {} for doc_id in doc_ids}
        tasks = (
            db.query(Task)
            .filter(Task.document_id.in_(doc_ids))
            .order_by(Task.created_at.asc())
            .all()
        )
        for t in tasks:
            task_summary_map[t.document_id][t.task_type] = t.status

        items: List[dict] = []
        for doc_id, snippet, match_field in matches:
            d = doc_map.get(doc_id)
            if not d:
                continue
            items.append({
                "id": d.id,
                "title": d.title,
                "original_filename": d.original_filename,
                "format": d.format,
                "total_pages": d.total_pages,
                "processing_status": d.processing_status,
                "source_language": d.source_language,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "task_summary": task_summary_map.get(d.id) or None,
                "snippet": snippet,
                "match_field": match_field,
            })
        return items

    @staticmethod
    def _extract_snippet(text: str, query: str, context: int = 100) -> str:
        """Extract a short snippet around the first occurrence of *query*."""
        lower_text = text.lower()
        idx = lower_text.find(query.lower())
        if idx == -1:
            return text[:200] + ("..." if len(text) > 200 else "")
        start = max(0, idx - context)
        end = min(len(text), idx + len(query) + context)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet
