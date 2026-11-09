"""Score a document's digest against a reviewer's reference answers (read-only).

python scripts/eval_digest_quality.py DOC_018 --gold path/to/gold.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import services  # noqa: F401,E402  (resolves the utils.tree_payload <-> services import cycle)
from data.database import get_db_manager  # noqa: E402
from data.db_models import (  # noqa: E402
    Document,
    DocumentKeyword,
    DocumentResearchDirection,
    Keyword,
    MainContent,
    Summary,
)
from utils.digest_eval import chapter_coverage, inconsistent_terms, keyword_overlap  # noqa: E402


def evaluate(document_id: str, gold: dict) -> dict:
    with get_db_manager().get_session() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        mc = (
            db.query(MainContent)
            .filter(MainContent.document_id == document_id, MainContent.status == "COMPLETED")
            .order_by(MainContent.created_at.desc())
            .first()
        )
        summary = (
            db.query(Summary)
            .filter(Summary.document_id == document_id, Summary.status == "COMPLETED")
            .order_by(Summary.created_at.desc())
            .first()
        )
        kw_rows = (
            db.query(Keyword.keyword_name, DocumentKeyword.display)
            .join(DocumentKeyword, DocumentKeyword.keyword_id == Keyword.id)
            .filter(DocumentKeyword.document_id == document_id)
            .all()
        )
        directions = (
            db.query(DocumentResearchDirection)
            .filter(DocumentResearchDirection.document_id == document_id)
            .count()
        )
        bib = dict(doc.bibliographic_metadata or {})
        pages = doc.total_pages
        chapters = list((mc.details or {}).get("chapters") or []) if mc else []
        summary_text = summary.content if summary else ""

    displays = [d or k for k, d in kw_rows]
    texts = [summary_text, *(c.get("content", "") for c in chapters), *displays]
    gold_terms = [en for _, en in gold.get("keywords", [])]
    title = bib.get("title_display") or ""
    return {
        "document": document_id,
        "pages": pages,
        "title": title,
        "title_ok": all(
            s.casefold() in title.casefold() for s in gold.get("title_must_contain", [])
        ),
        "chapters": chapter_coverage(
            [c.get("heading_ordinal") for c in chapters if c.get("heading_kind") == "chapter"],
            gold.get("chapters", 0),
        ),
        "keywords": {"count": len(displays), **keyword_overlap(displays, gold.get("keywords", []))},
        "research_directions": directions,
        "inconsistent_terms": inconsistent_terms(texts, gold_terms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(args.document_id, gold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
