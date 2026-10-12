"""DOMAIN_PROFILE — name the document's specialty and pin its recurring terms.

Runs once after the tree is built. Every later digest stage reads the result
(``Document.domain_profile`` / ``Document.termbase``), so the title, keywords,
summary and chapters speak one specialist Vietnamese instead of each translating
word by word on its own.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from core.spatial.zone_classifier import split_chapter_heading
from data.database import get_db_manager
from services.base_service import BaseTaskService
from utils.domain_profile import domain_clause, parse_profile, parse_termbase

logger = logging.getLogger(__name__)

EXCERPT_CHARS = 6000
OUTLINE_MAX_TITLES = 150
TERMBASE_MAX_TERMS = 150
_NUMBERING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")


def outline_titles(tree: Any, depth: int = 2) -> List[str]:
    """Heading titles of the top ``depth`` levels, without «Chapter N» / «1.2» labels."""
    titles: List[str] = []
    seen = set()

    def walk(node: Dict[str, Any], level: int) -> None:
        for child in node.get("children") or []:
            if not isinstance(child, dict):
                continue
            raw = str(child.get("title") or "").strip()
            title = _NUMBERING_RE.sub("", split_chapter_heading(raw)[1] or raw).strip()
            if title and title.casefold() not in seen:
                seen.add(title.casefold())
                titles.append(title)
            if level < depth:
                walk(child, level + 1)

    if isinstance(tree, dict):
        walk(tree, 1)
    return titles[:OUTLINE_MAX_TITLES]


class DomainProfileService(BaseTaskService):
    async def run_for_pipeline(
        self, document_id: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        from api.dependencies import get_llm_client

        llm = get_llm_client()
        self._progress(task_id, 10, "Đang đọc mục lục và phần đầu tài liệu")
        inputs = self._load_inputs(document_id)

        self._progress(task_id, 30, "Đang xác định lĩnh vực, ngành, chuyên ngành")
        profile = await self._profile(llm, inputs)

        self._progress(task_id, 60, "Đang lập bảng thuật ngữ")
        termbase = await self._termbase(llm, inputs, profile)

        self._progress(task_id, 90, "Đang lưu hồ sơ lĩnh vực")
        self._save(document_id, profile, termbase)
        self._progress(task_id, 100, "Hoàn tất")
        return {"profile": profile, "terms": len(termbase)}

    # ── LLM calls ────────────────────────────────────────────────────

    async def _profile(self, llm, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from utils.doc_kind import resolve_doc_kind_async

        prompt = (
            "You are a senior librarian classifying a document for a military technical "
            "academy's catalogue.\n\n"
            "TASK: From the title, contents and opening text, state the document's FIELD "
            "(lĩnh vực), DISCIPLINE (ngành) and SPECIALTY (chuyên ngành), each in Vietnamese "
            "(the standard name used in Vietnamese textbooks) and in English.\n"
            "If the document is a volume of independent papers covering several areas, also "
            'list those areas in "clusters".\n\n'
            "RULES: Judge only from the text below, which may be in any language. Give the "
            "specialty its established Vietnamese name, not a literal translation "
            "(e.g. Ballistics → Đạn đạo học).\n\n"
            "OUTPUT: JSON only —\n"
            '{"field": {"vi": "", "en": ""}, "discipline": {"vi": "", "en": ""}, '
            '"specialty": {"vi": "", "en": ""}, "clusters": [{"vi": "", "en": ""}], '
            '"summary_en": "one sentence"}\n\n'
            f"TITLE: {inputs['title']}\n\n"
            f"CONTENTS:\n{chr(10).join(inputs['outline'])}\n\n"
            f"OPENING TEXT:\n{inputs['excerpt'][:EXCERPT_CHARS]}\n\nJSON:"
        )
        profile: Dict[str, Any] = {}
        try:
            profile = parse_profile(llm.extract_json(await llm.chat_completion(prompt)))
        except Exception as exc:
            logger.warning("domain profile: model answer unusable (%s)", exc)

        kind = await resolve_doc_kind_async(
            llm, inputs.get("explicit_kind"), title=inputs["title"], text=inputs["excerpt"]
        )
        return {**profile, "doc_kind": kind["doc_kind"]}

    async def _termbase(
        self, llm, inputs: Dict[str, Any], profile: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        if not inputs["outline"]:
            return {}
        prompt = (
            "You are a terminologist for Vietnamese technical publishing.\n\n"
            f"{domain_clause(profile)}\n"
            "TASK: From the headings below pick the specialist terms (not generic words) "
            f"that recur in the document, at most {TERMBASE_MAX_TERMS}. For each give the "
            "standard Vietnamese term used by specialists in this field and the English "
            "term.\n\n"
            "RULES: One Vietnamese term per source term. Keep the source term exactly as "
            "written in the headings. Do not invent terms that are not in the headings.\n\n"
            'OUTPUT: JSON array only — [{"src": "", "vi": "", "en": ""}]\n\n'
            f"HEADINGS:\n{chr(10).join(inputs['outline'])}\n\nJSON:"
        )
        try:
            rows = llm.extract_json(await llm.chat_completion(prompt, max_tokens=4000))
        except Exception as exc:
            logger.warning("domain profile: termbase answer unusable (%s)", exc)
            return {}
        return parse_termbase(rows)

    # ── Database seams ───────────────────────────────────────────────

    def _load_inputs(self, document_id: str) -> Dict[str, Any]:
        from data.db_models import Document, TreeIndex
        from services.bibliographic_service import front_matter_excerpt
        from utils.tree_payload import get_tree_payload

        with get_db_manager().session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError("Document not found")
            tree_row = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )
            outline = outline_titles(get_tree_payload(db, tree_row)) if tree_row else []
            inputs = {
                "title": doc.title or "",
                "source_language": doc.source_language or "",
                "outline": outline,
                "explicit_kind": doc.digest_doc_kind,
            }
        inputs["excerpt"] = front_matter_excerpt(self._read_text(document_id), EXCERPT_CHARS)
        return inputs

    def _save(self, document_id: str, profile: Dict[str, Any], termbase: Dict[str, Any]) -> None:
        from data.db_models import Document

        with get_db_manager().session() as db:
            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.domain_profile = profile
                row.termbase = termbase
