"""
Document Digest service.

Assembles a structured digest (tổng thuật) from existing per-service results
already stored in the DB.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import (
    Document,
    DocumentKeyword,
    DocumentResearchDirection,
    Keyword,
    MainContent,
    ResearchDirection,
    Summary,
)
from utils.digest_admin import normalize_digest_admin
from utils.digest_format import (
    bibliographic_defaults,
    is_chapter_schema,
    usage_scope_defaults,
)
from utils.doc_kind import BOOK, normalize_doc_kind


@dataclass
class ChapterEntry:
    number: int
    title_vi: str
    title_original: str
    content: str
    # Kỷ yếu: số BBKH gộp trong mục này. None/0/1 → mục là một BBKH đơn lẻ.
    paper_count: Optional[int] = None
    # Loại đơn vị mà tiêu đề gốc tự khai (chapter/appendix/part) — để một phụ lục
    # không bị in thành "Chương N".
    heading_kind: Optional[str] = None
    heading_ordinal: Optional[int] = None


@dataclass
class KeywordEntry:
    keyword: str
    weight: float
    display: str = ""


@dataclass
class ResearchDirectionEntry:
    direction_name: str
    confidence: float


@dataclass
class DigestResult:
    document_id: str
    title: str
    source_language: str
    original_filename: Optional[str]

    bibliographic: dict = field(default_factory=dict)
    abstract: Optional[str] = None
    # "book" | "proceedings" — quyết định dạng dòng §2.2 (utils/doc_kind.py).
    doc_kind: str = BOOK
    chapters: List[ChapterEntry] = field(default_factory=list)
    keywords: List[KeywordEntry] = field(default_factory=list)
    usage_scope: dict = field(default_factory=dict)
    research_directions: List[ResearchDirectionEntry] = field(default_factory=list)

    reviewer: str = ""
    reviewer_approved: str = ""
    entry_date: str = ""

    missing: List[str] = field(default_factory=list)

    # Legacy field for API backward compat
    @property
    def main_content(self) -> Optional[dict]:
        if not self.chapters:
            return None
        return {
            "chapters": [
                {
                    "number": c.number,
                    "title_vi": c.title_vi,
                    "title_original": c.title_original,
                    "content": c.content,
                }
                for c in self.chapters
            ]
        }


class DigestService:
    """Assemble DigestResult from DB — no LLM calls."""

    def assemble(self, db: Session, document_id: str) -> DigestResult:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError(f"Document '{document_id}' not found")

        missing: List[str] = []

        bib = dict(
            doc.bibliographic_metadata
            or bibliographic_defaults(title=doc.title, pages=doc.total_pages)
        )
        if not bib.get("title_display"):
            bib["title_display"] = doc.title or ""
        if not any(bib.get(k) for k in ("authors", "publisher", "isbn")):
            missing.append("bibliographic")

        summary_row = (
            db.query(Summary)
            .filter(Summary.document_id == document_id, Summary.status == "COMPLETED")
            .order_by(Summary.created_at.desc())
            .first()
        )
        abstract = summary_row.content if summary_row else None
        if abstract is None:
            missing.append("abstract")

        chapters: List[ChapterEntry] = []
        mc_row = (
            db.query(MainContent)
            .filter(MainContent.document_id == document_id, MainContent.status == "COMPLETED")
            .order_by(MainContent.created_at.desc())
            .first()
        )
        if mc_row and is_chapter_schema(mc_row.details):
            for ch in mc_row.details.get("chapters", []):
                chapters.append(
                    ChapterEntry(
                        number=int(ch.get("number", len(chapters) + 1)),
                        title_vi=str(ch.get("title_vi", "")),
                        title_original=str(ch.get("title_original", "")),
                        content=str(ch.get("content", "")),
                        paper_count=ch.get("paper_count"),
                        heading_kind=ch.get("heading_kind"),
                        heading_ordinal=ch.get("heading_ordinal"),
                    )
                )
        if not chapters:
            missing.append("main_content")

        # Ưu tiên giá trị đặt tay; nếu không có thì dùng thể loại mà lượt chạy
        # main_content đã chốt, để dòng §2.2 khớp với prompt đã sinh ra nó.
        doc_kind = normalize_doc_kind(getattr(doc, "digest_doc_kind", None))
        if not doc_kind and mc_row and isinstance(mc_row.details, dict):
            doc_kind = normalize_doc_kind(mc_row.details.get("doc_kind"))

        kw_rows = (
            db.query(DocumentKeyword, Keyword)
            .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
            .filter(DocumentKeyword.document_id == document_id)
            .order_by(DocumentKeyword.weight.desc())
            .limit(20)
            .all()
        )
        keywords = []
        for dk, kw in kw_rows:
            # `display` carries the "Tiếng Việt (original)" bilingual form when
            # the model produced one; there is no way to synthesise it here, so
            # the bare term is the honest fallback. Coverage is reported as a
            # quality warning rather than papered over.
            keywords.append(
                KeywordEntry(
                    keyword=kw.keyword_name,
                    weight=dk.weight,
                    display=(dk.display or "").strip() or kw.keyword_name,
                )
            )
        if not keywords:
            missing.append("keywords")

        usage = dict(doc.usage_scope or usage_scope_defaults())
        if not any(
            usage.get(k) for k in ("undergraduate", "master", "phd", "strong_research_groups")
        ):
            missing.append("usage_scope")

        rd_rows = (
            db.query(DocumentResearchDirection, ResearchDirection)
            .join(
                ResearchDirection,
                DocumentResearchDirection.direction_id == ResearchDirection.id,
            )
            .filter(DocumentResearchDirection.document_id == document_id)
            .order_by(DocumentResearchDirection.confidence.desc())
            .limit(20)
            .all()
        )
        research_directions = [
            ResearchDirectionEntry(
                direction_name=rd.direction_name,
                confidence=drd.confidence,
            )
            for drd, rd in rd_rows
        ]
        if not research_directions:
            missing.append("research_directions")

        admin = normalize_digest_admin(getattr(doc, "digest_admin", None) or None)

        return DigestResult(
            document_id=document_id,
            title=doc.title,
            source_language=doc.source_language or "en",
            original_filename=doc.original_filename,
            bibliographic=bib,
            abstract=abstract,
            doc_kind=doc_kind or BOOK,
            chapters=chapters,
            keywords=keywords,
            usage_scope=usage,
            research_directions=research_directions,
            reviewer=admin["reviewer"],
            reviewer_approved=admin["reviewer_approved"],
            entry_date=admin["entry_date"],
            missing=missing,
        )
