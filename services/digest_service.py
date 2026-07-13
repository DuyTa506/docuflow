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
from utils.digest_format import (
    bibliographic_defaults,
    format_keyword_bilingual,
    is_chapter_schema,
    usage_scope_defaults,
)


@dataclass
class ChapterEntry:
    number: int
    title_vi: str
    title_original: str
    content: str


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
                    )
                )
        if not chapters:
            missing.append("main_content")

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
            display = dk.display or ""
            keywords.append(
                KeywordEntry(
                    keyword=kw.keyword_name,
                    weight=dk.weight,
                    display=format_keyword_bilingual(
                        doc.source_language or "en",
                        kw.keyword_name,
                        display if display else None,
                    ),
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

        return DigestResult(
            document_id=document_id,
            title=doc.title,
            source_language=doc.source_language or "en",
            original_filename=doc.original_filename,
            bibliographic=bib,
            abstract=abstract,
            chapters=chapters,
            keywords=keywords,
            usage_scope=usage,
            research_directions=research_directions,
            missing=missing,
        )
