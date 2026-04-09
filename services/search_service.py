"""
Search service.

SQL LIKE/ILIKE search across documents, text content, keywords, and translations.
"""
from typing import List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from data.db_models import (
    Document, DigitizedText, Keyword, DocumentKeyword, Translation,
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
    ) -> dict:
        """
        Search across specified fields.

        *search_in* can include: ``title``, ``content``, ``keywords``, ``translations``.
        Defaults to all four.
        """
        if not query or not query.strip():
            return {"results": [], "total": 0, "query": query}

        if search_in is None:
            search_in = ["title", "content", "keywords", "translations"]

        pattern = f"%{query}%"
        results = []
        seen_ids = set()

        # 1. Title search
        if "title" in search_in:
            docs = (
                db.query(Document)
                .filter(Document.title.ilike(pattern))
                .limit(limit)
                .all()
            )
            for d in docs:
                if d.id not in seen_ids:
                    results.append({
                        "document_id": d.id,
                        "title": d.title,
                        "snippet": d.title,
                        "match_field": "title",
                    })
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
                .limit(limit)
                .all()
            )
            for dt in dts:
                if dt.document_id not in seen_ids:
                    doc = db.query(Document).filter(Document.id == dt.document_id).first()
                    text = dt.normalized_content or dt.ocr_content or ""
                    # Extract snippet around match
                    snippet = self._extract_snippet(text, query)
                    results.append({
                        "document_id": dt.document_id,
                        "title": doc.title if doc else "Unknown",
                        "snippet": snippet,
                        "match_field": "content",
                    })
                    seen_ids.add(dt.document_id)

        # 3. Keyword search
        if "keywords" in search_in:
            keyword_matches = (
                db.query(DocumentKeyword, Keyword, Document)
                .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
                .join(Document, DocumentKeyword.document_id == Document.id)
                .filter(Keyword.keyword_name.ilike(pattern))
                .limit(limit)
                .all()
            )
            for assoc, kw, doc in keyword_matches:
                if doc.id not in seen_ids:
                    results.append({
                        "document_id": doc.id,
                        "title": doc.title,
                        "snippet": f"Keyword: {kw.keyword_name} (weight: {assoc.weight:.2f})",
                        "match_field": "keywords",
                    })
                    seen_ids.add(doc.id)

        # 4. Translation search
        if "translations" in search_in:
            trans = (
                db.query(Translation)
                .filter(Translation.translated_content.ilike(pattern))
                .limit(limit)
                .all()
            )
            if language:
                trans = [t for t in trans if t.target_language == language]
            for t in trans:
                if t.document_id not in seen_ids:
                    doc = db.query(Document).filter(Document.id == t.document_id).first()
                    snippet = self._extract_snippet(t.translated_content or "", query)
                    results.append({
                        "document_id": t.document_id,
                        "title": doc.title if doc else "Unknown",
                        "snippet": snippet,
                        "match_field": "translations",
                    })
                    seen_ids.add(t.document_id)

        # Apply pagination
        total = len(results)
        results = results[offset: offset + limit]

        return {"results": results, "total": total, "query": query}

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
