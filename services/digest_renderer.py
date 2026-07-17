"""
Digest DOCX renderer — docxtpl template matching Mau Tong thuat Book.
"""

import io
from pathlib import Path
from typing import Optional

from docxtpl import DocxTemplate

from services.digest_service import DigestResult
from utils.digest_format import join_catalog_items

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "template" / "docuflow_digest_template.docx"
)


class DigestRenderer:
    """Render DigestResult to .docx via docxtpl."""

    def __init__(self, template_path: Optional[Path] = None):
        self.template_path = template_path or _TEMPLATE_PATH

    def _build_context(
        self,
        digest: DigestResult,
        reviewer: str = "",
        reviewer_approved: str = "",
        entry_date: str = "",
    ) -> dict:
        usage = digest.usage_scope or {}
        return {
            "bib": digest.bibliographic or {},
            "abstract": digest.abstract or "[Chưa có — chạy summarize trước]",
            "chapters": [
                {
                    "number": c.number,
                    "title_vi": c.title_vi,
                    "title_original": c.title_original,
                    "content": c.content,
                }
                for c in digest.chapters
            ],
            "keywords": [
                {"display": k.display or k.keyword, "keyword": k.keyword, "weight": k.weight}
                for k in digest.keywords
            ],
            "usage": {
                "undergraduate_text": join_catalog_items(usage.get("undergraduate", [])),
                "master_text": join_catalog_items(usage.get("master", [])),
                "phd_text": join_catalog_items(usage.get("phd", [])),
                "strong_research_groups_text": join_catalog_items(
                    usage.get("strong_research_groups", [])
                ),
            },
            "research_directions": [
                {"direction_name": rd.direction_name, "confidence": rd.confidence}
                for rd in digest.research_directions
            ],
            "reviewer": reviewer or digest.reviewer,
            "reviewer_approved": reviewer_approved or digest.reviewer_approved,
            "entry_date": entry_date or digest.entry_date,
        }

    def render(
        self,
        digest: DigestResult,
        reviewer: str = "",
        reviewer_approved: str = "",
        entry_date: str = "",
    ) -> bytes:
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Digest template not found: {self.template_path}")

        tpl = DocxTemplate(str(self.template_path))
        context = self._build_context(
            digest,
            reviewer=reviewer,
            reviewer_approved=reviewer_approved,
            entry_date=entry_date,
        )
        # autoescape: LLM-generated content can contain XML-special chars —
        # a literal `<a>` in one chapter summary swallowed every later
        # section of the rendered document (DOC_066, chapter 45/106).
        tpl.render(context, autoescape=True)
        buf = io.BytesIO()
        tpl.save(buf)
        return buf.getvalue()
