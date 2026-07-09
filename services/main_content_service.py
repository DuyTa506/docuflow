"""
Main content extraction service — chapter breakdown for digest §2.2.

Tree-first: walk TreeIndex chapter nodes and LLM-summarize each section.
Fallback: detect markdown headings in OCR text.
"""
import re
from typing import List, Dict, Optional

from config.settings import pipeline_output_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import MainContent
from services.base_service import BaseTaskService
from services.task_manager import task_manager


def _collect_chapter_nodes(node: dict, depth: int = 0, max_depth: int = 2) -> List[dict]:
    """Collect top-level chapter nodes from tree."""
    chapters = []
    children = node.get("children") or node.get("child_nodes") or []
    if depth == 0 and children:
        for i, child in enumerate(children, start=1):
            chapters.append({"node": child, "number": i})
        return chapters
    if depth < max_depth and children:
        for child in children:
            title = (child.get("title") or "").strip()
            if title:
                chapters.append({"node": child, "number": len(chapters) + 1})
    return chapters


def _gather_node_text(node: dict, max_chars: int = 4000, max_nodes: int = 300) -> str:
    """Concatenate a node's own text with descendant text in document
    reading order (depth-first pre-order) until max_chars is reached.

    Handles heading-only nodes whose real body content lives one or more
    levels down in children, which a node's own `content`/`text` field
    alone would miss. Depth-first (not breadth-first) matters whenever
    grandchild-level text is needed: it keeps each child's own
    descendants adjacent to it, instead of interleaving every child's
    text before any grandchild's.

    `max_nodes` bounds traversal independent of `max_chars`: a subtree of
    mostly heading-only/empty nodes (confirmed real on large books — one
    761-page document had a 258-node chapter subtree) would otherwise be
    walked in full before the char budget ever kicks in, since empty nodes
    never add to `total_len`. This is called once per top-level chapter by
    the digest pipeline's main-content stage, so an unbounded worst case
    here is an unbounded worst case for every digest run.
    """
    def _own_text(n: dict) -> str:
        return (
            n.get("content") or n.get("text") or n.get("text_content") or n.get("summary") or ""
        ).strip()

    parts: list[str] = []
    total_len = 0
    visited = 0
    stack = [node]

    while stack and total_len < max_chars and visited < max_nodes:
        current = stack.pop()
        visited += 1
        text = _own_text(current)
        if text:
            parts.append(text)
            total_len += len(text)
        children = current.get("children") or current.get("child_nodes") or []
        for child in reversed(children):  # reversed + LIFO pop = left-to-right pre-order
            stack.append(child)

    return "\n\n".join(parts)


def _parse_markdown_chapters(text: str) -> List[dict]:
    """Fallback: split on markdown # / ## headings."""
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chapters = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = m.group(2).strip()
        chapters.append({
            "number": i + 1,
            "title_vi": title,
            "title_original": title,
            "content": body[:3000],
        })
    return chapters


class MainContentService(BaseTaskService):
    """Extract per-chapter main content (background task)."""

    def submit(self, db, document_id: str) -> tuple:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        existing_task = task_manager.get_active_task_id(db, document_id, "MAIN_CONTENT")
        if existing_task:
            mc = (
                db.query(MainContent)
                .filter(MainContent.document_id == document_id)
                .order_by(MainContent.created_at.desc())
                .first()
            )
            return existing_task, (mc.id if mc else None), True

        mc = MainContent(document_id=document_id, status="PENDING")
        db.add(mc)
        db.commit()
        db.refresh(mc)
        main_content_id = mc.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="MAIN_CONTENT",
            coro_factory=lambda tid: self._extract(document_id, main_content_id, tid),
        )
        return task_id, main_content_id, False

    async def run_for_pipeline(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        with db_manager.session() as db:
            mc = (
                db.query(MainContent)
                .filter(MainContent.document_id == document_id)
                .order_by(MainContent.created_at.desc())
                .first()
            )
            if not mc:
                mc = MainContent(document_id=document_id, status="PENDING")
                db.add(mc)
                db.commit()
                db.refresh(mc)
            main_content_id = mc.id
        return await self._extract(document_id, main_content_id, task_id)

    async def _summarize_chapter(self, llm, node: dict, number: int) -> tuple[dict, bool, bool]:
        """Returns (chapter_dict, degraded, raw_passthrough).

        degraded=True: the LLM call failed and the raw-text fallback excerpt
        was used instead (a runtime failure).
        raw_passthrough=True: the LLM was never called because the node's
        (own + descendant) content was too short to summarize meaningfully —
        not a failure, but still means this chapter's digest entry is raw
        source text rather than a real summary, which the caller should be
        able to surface as a quality signal distinct from `degraded`.
        """
        title = (node.get("title") or f"Chương {number}").strip()
        content = _gather_node_text(node)
        lang_clause = pipeline_output_lang_clause()
        degraded = False
        raw_passthrough = False

        if len(content.strip()) < 80:
            body = content.strip() or title
            raw_passthrough = True
        else:
            prompt = (
                "You are a document analyst.\n\n"
                "TASK: Write a detailed summary of this book chapter for a library digest.\n"
                "Format: 2-5 sentences in Vietnamese covering main topics, methods, and findings.\n"
                "Preserve technical terms; add English/Russian originals in parentheses when helpful.\n\n"
                f"{lang_clause}"
                f"Chapter title: {title}\n\n"
                f"Chapter text:\n{content[:4000]}\n\n"
                # Repeated immediately before the generation point: with a long
                # non-Vietnamese source block in context, smaller local models
                # (confirmed live: qwen3.5-9b on a Russian-source chapter)
                # otherwise mirror the source language instead of following an
                # instruction stated only once, further back in the prompt.
                f"{lang_clause}"
                "Summary (in Vietnamese):"
            )
            try:
                body = (await llm.chat_completion(prompt)).strip()
            except Exception:
                body = content[:500] + ("..." if len(content) > 500 else "")
                degraded = True

        # Split bilingual title: "Vi (Original)" heuristic
        title_vi = title
        title_original = title
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", title)
        if m:
            title_vi = m.group(1).strip()
            title_original = m.group(2).strip()

        return {
            "number": number,
            "title_vi": title_vi,
            "title_original": title_original,
            "content": body,
        }, degraded, raw_passthrough

    async def _extract(
        self,
        document_id: str,
        main_content_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()

        def _set_status(status: str):
            if not main_content_id:
                return
            with db_manager.session() as db:
                mc = db.query(MainContent).filter(MainContent.id == main_content_id).first()
                if mc:
                    mc.status = status

        _set_status("IN_PROGRESS")

        try:
            from api.dependencies import get_llm_client
            llm = get_llm_client()

            tree_data = None
            with db_manager.session() as db:
                from data.db_models import TreeIndex
                from utils.tree_payload import get_tree_payload

                tree_index = (
                    db.query(TreeIndex)
                    .filter(TreeIndex.document_id == document_id)
                    .order_by(TreeIndex.created_at.desc())
                    .first()
                )
                if tree_index:
                    tree_data = get_tree_payload(db, tree_index)

            chapters: List[dict] = []
            degraded_chapters = 0
            raw_passthrough_chapters = 0

            if tree_data:
                self._progress(task_id, 15, "Walking tree for chapters")
                nodes = _collect_chapter_nodes(tree_data)
                total = len(nodes)
                for i, item in enumerate(nodes):
                    ch, degraded, raw_passthrough = await self._summarize_chapter(
                        llm, item["node"], item["number"]
                    )
                    chapters.append(ch)
                    if degraded:
                        degraded_chapters += 1
                    if raw_passthrough:
                        raw_passthrough_chapters += 1
                    pct = int(15 + ((i + 1) / max(total, 1)) * 75)
                    self._progress(task_id, pct, f"Chapter {i + 1}/{total}")

            if not chapters:
                self._progress(task_id, 20, "Fallback: markdown headings")
                text = self._read_text(document_id)
                chapters = _parse_markdown_chapters(text)
                if not chapters:
                    excerpt = BaseEnricher(llm).truncate_to_tokens(text, 2000)
                    chapters = [{
                        "number": 1,
                        "title_vi": "Tài liệu",
                        "title_original": "Document",
                        "content": excerpt[:1500],
                    }]

            details = {
                "chapters": chapters,
                "degraded_chapters": degraded_chapters,
                "raw_passthrough_chapters": raw_passthrough_chapters,
            }

            with db_manager.session() as db:
                if main_content_id:
                    mc = db.query(MainContent).filter(MainContent.id == main_content_id).first()
                    if mc:
                        mc.details = details
                        mc.status = "COMPLETED"
                else:
                    db.add(MainContent(
                        document_id=document_id,
                        details=details,
                        status="COMPLETED",
                    ))

            self._progress(task_id, 100, "Done")

            from services.export_service import export_service

            export_service.mark_digest_dirty(document_id)

            return details
        except Exception:
            _set_status("FAILED")
            raise
