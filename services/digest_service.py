"""
Document Digest service.

Assembles a structured digest (tổng thuật) from existing per-service results
already stored in the DB:

  Section 1  — document metadata (Document table)
  Section 2.1 — abstract/summary        → summaries table (type="short")
  Section 2.2 — chapter breakdown       → main_contents table
  Section 2.3 — keywords (20 terms)     → document_keywords + keywords tables
  Section 3   — scope / usage           → document_research_directions + research_directions

The service does NOT re-run any AI; it reads what is already in the DB and
assembles a DigestResult dataclass.  A separate DigestRenderer turns that into
a .docx file.

Prerequisite order (must be run before generating a digest):
  1. extract / ocr           → DigitizedText exists
  2. (optional) translate    → if source language ≠ vi, run translation first
  3. summarize  (short)      → Summary type="short"
  4. main_content            → MainContent record
  5. keywords                → DocumentKeyword rows
  6. research_directions     → DocumentResearchDirection rows
"""
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from data.db_models import (
    Document,
    Summary,
    MainContent,
    Keyword,
    DocumentKeyword,
    ResearchDirection,
    DocumentResearchDirection,
)


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class KeywordEntry:
    keyword: str
    weight: float


@dataclass
class ResearchDirectionEntry:
    direction_name: str
    confidence: float


@dataclass
class DigestResult:
    """
    Fully assembled digest ready for rendering.

    All text fields are in the document's own language (or Vietnamese if a
    translation was run).  The renderer writes these verbatim into the .docx.
    """
    # ── Section 1: bibliographic metadata ──────────────────────────
    document_id: str
    title: str
    source_language: str
    original_filename: Optional[str]

    # ── Section 2.1: abstract ───────────────────────────────────────
    abstract: Optional[str] = None          # Summary type="short"

    # ── Section 2.2: chapter / main content breakdown ──────────────
    # Raw JSON from MainContent.details:
    # {"key_points": [...], "methods": [...], "results": [...], "conclusions": [...]}
    main_content: Optional[dict] = None

    # ── Section 2.3: keywords ───────────────────────────────────────
    keywords: List[KeywordEntry] = field(default_factory=list)

    # ── Section 3: scope / research directions ──────────────────────
    research_directions: List[ResearchDirectionEntry] = field(default_factory=list)

    # ── Flags ───────────────────────────────────────────────────────
    missing: List[str] = field(default_factory=list)
    """Names of sections that were absent from the DB (not yet processed)."""


# ── Service ───────────────────────────────────────────────────────────

class DigestService:
    """
    Assemble a DigestResult from DB records for a given document.

    Usage:
        svc = DigestService()
        result = svc.assemble(db, document_id)   # sync, fast — no LLM calls
        # then pass result to DigestRenderer
    """

    def assemble(self, db: Session, document_id: str) -> DigestResult:
        """
        Read all relevant DB rows and return a DigestResult.

        Raises ValueError if the document does not exist.
        Partial results are allowed — missing sections are listed in
        DigestResult.missing so the caller can decide whether to abort or
        render a partial digest.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError(f"Document '{document_id}' not found")

        missing: List[str] = []

        # ── 2.1 Abstract ────────────────────────────────────────────
        summary_row = (
            db.query(Summary)
            .filter(
                Summary.document_id == document_id,
                Summary.summary_type == "short",
            )
            .order_by(Summary.created_at.desc())
            .first()
        )
        abstract = summary_row.content if summary_row else None
        if abstract is None:
            missing.append("abstract (run summarize with type='short')")

        # ── 2.2 Main content ────────────────────────────────────────
        mc_row = (
            db.query(MainContent)
            .filter(MainContent.document_id == document_id)
            .order_by(MainContent.created_at.desc())
            .first()
        )
        main_content = mc_row.details if mc_row else None
        if main_content is None:
            missing.append("main_content (run main_content service)")

        # ── 2.3 Keywords ────────────────────────────────────────────
        kw_rows = (
            db.query(DocumentKeyword, Keyword)
            .join(Keyword, DocumentKeyword.keyword_id == Keyword.id)
            .filter(DocumentKeyword.document_id == document_id)
            .order_by(DocumentKeyword.weight.desc())
            .all()
        )
        keywords = [
            KeywordEntry(keyword=kw.keyword_name, weight=dk.weight)
            for dk, kw in kw_rows
        ]
        if not keywords:
            missing.append("keywords (run keyword service)")

        # ── 3. Research directions ───────────────────────────────────
        rd_rows = (
            db.query(DocumentResearchDirection, ResearchDirection)
            .join(
                ResearchDirection,
                DocumentResearchDirection.direction_id == ResearchDirection.id,
            )
            .filter(DocumentResearchDirection.document_id == document_id)
            .order_by(DocumentResearchDirection.confidence.desc())
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
            missing.append("research_directions (run research_direction service)")

        return DigestResult(
            document_id=document_id,
            title=doc.title,
            source_language=doc.source_language or "en",
            original_filename=doc.original_filename,
            abstract=abstract,
            main_content=main_content,
            keywords=keywords,
            research_directions=research_directions,
            missing=missing,
        )
