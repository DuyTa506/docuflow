"""
Main content extraction service — chapter breakdown for digest §2.2.

Tree-first: walk TreeIndex chapter nodes and LLM-summarize each section.
Fallback: detect markdown headings in OCR text.
"""

import re
from typing import Dict, List, Optional

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


# Shared with the stratified sampling utility (utils/doc_sampling.py) —
# kept importable under the old name for existing callers/tests.
from utils.doc_sampling import gather_node_text as _gather_node_text

# Node-classification gate (§2.2 noise control): trees mirror every heading,
# so publisher front matter can become dozens of sibling nodes of real
# chapters, each rendering as a title echo or "nothing to summarize" filler.
GATE_BATCH_SIZE = 30
GATE_AUX_LABELS = frozenset({"front_matter", "toc_fragment"})
GATE_LABELS = GATE_AUX_LABELS | {"substantive"}
# A `toc_fragment` verdict is only accepted when the node's gathered content
# is actually this thin — a fat chapter mislabeled by the LLM stays
# substantive no matter what the model says.
TOC_FRAGMENT_MAX_CHARS = 300


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
        chapters.append(
            {
                "number": i + 1,
                "title_vi": title,
                "title_original": title,
                "content": body[:3000],
            }
        )
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

    async def _classify_nodes(self, llm, nodes: List[dict]) -> tuple[Dict[int, str], bool]:
        """Label every chapter node substantive / front_matter / toc_fragment
        in batched LLM calls (title + content length + short excerpt only).

        Fail-open: any batch that errors or returns unparseable JSON leaves
        its nodes substantive (today's behavior) and flags gate_degraded.
        """
        labels: Dict[int, str] = {item["number"]: "substantive" for item in nodes}
        content_chars: Dict[int, int] = {}
        degraded = False

        for start in range(0, len(nodes), GATE_BATCH_SIZE):
            batch = nodes[start : start + GATE_BATCH_SIZE]
            lines = []
            for item in batch:
                node = item["node"]
                title = (node.get("title") or "").strip() or f"Section {item['number']}"
                text = _gather_node_text(node, max_chars=600)
                content_chars[item["number"]] = len(text)
                excerpt = " ".join(text.split())[:150]
                lines.append(f"{item['number']} | {title} | chars={len(text)} | {excerpt}")

            prompt = (
                "You are a document structure analyst.\n\n"
                "TASK: Classify each section listed below into exactly one label:\n"
                '- "substantive": real chapter/section content — topics, methods, '
                "findings, discussion.\n"
                '- "front_matter": publisher/administrative material — copyright page, '
                "acknowledgements, author biographies, subscription or support offers, "
                "errata/feedback instructions, dedication.\n"
                '- "toc_fragment": a bare heading with no real body text of its own.\n\n'
                "RULES:\n"
                "- Judge ONLY from the title, content length, and excerpt given.\n"
                '- If unsure, use "substantive".\n'
                "- Documents of any genre or language may appear; do not assume a "
                "specific layout.\n\n"
                "SECTIONS (number | title | content chars | excerpt):\n"
                + "\n".join(lines)
                + "\n\nOUTPUT: JSON array only, one entry per section: "
                '[{"number": 1, "label": "front_matter"}, ...]'
            )
            try:
                response = await llm.chat_completion(prompt)
            except Exception:
                degraded = True
                continue
            parsed = self._extract_json(llm, response, list_key="sections")
            if not parsed:
                degraded = True
                continue
            valid_numbers = {item["number"] for item in batch}
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                try:
                    num = int(row.get("number"))
                except (TypeError, ValueError):
                    continue
                label = str(row.get("label") or "").strip()
                if num not in valid_numbers or label not in GATE_LABELS:
                    continue
                if label == "toc_fragment" and content_chars.get(num, 0) > TOC_FRAGMENT_MAX_CHARS:
                    continue
                labels[num] = label

        return labels, degraded

    async def _summarize_with_gate(
        self,
        llm,
        nodes: List[dict],
        task_id: Optional[str],
    ) -> tuple[List[dict], int, int, int, bool]:
        """Classify nodes once, collapse *consecutive* auxiliary nodes into a
        single digest entry listing their titles, then summarize the
        substantive chapters as before (with their final numbering).

        Returns (chapters, degraded_count, raw_passthrough_count,
        auxiliary_sections, gate_degraded).
        """
        if len(nodes) < 2:
            labels: Dict[int, str] = {}
            gate_degraded = False
        else:
            self._progress(task_id, 12, "Classifying sections")
            labels, gate_degraded = await self._classify_nodes(llm, nodes)

        # Ordered plan: substantive items keep their own slot; a run of
        # consecutive auxiliary nodes shares one slot.
        plan: List[tuple] = []  # ("chapter", item) | ("aux", [titles])
        for item in nodes:
            if labels.get(item["number"]) in GATE_AUX_LABELS:
                title = (item["node"].get("title") or "").strip() or f"Mục {item['number']}"
                if plan and plan[-1][0] == "aux":
                    plan[-1][1].append(title)
                else:
                    plan.append(("aux", [title]))
            else:
                plan.append(("chapter", item))

        to_summarize = []
        for final_number, (kind, payload) in enumerate(plan, start=1):
            if kind == "chapter":
                to_summarize.append({"node": payload["node"], "number": final_number})

        summarized, degraded_count, raw_count = await self._summarize_chapters(
            llm, to_summarize, task_id
        )

        chapters: List[dict] = []
        auxiliary_sections = 0
        summarized_iter = iter(summarized)
        for final_number, (kind, payload) in enumerate(plan, start=1):
            if kind == "chapter":
                chapters.append(next(summarized_iter))
            else:
                auxiliary_sections += len(payload)
                chapters.append(
                    {
                        "number": final_number,
                        "title_vi": "Các mục phụ trợ",
                        "title_original": "Auxiliary sections",
                        "content": (
                            "Gồm các mục: "
                            + "; ".join(payload)
                            + ". Đây là các phần phụ trợ của tài liệu (bản quyền, "
                            "giới thiệu, mục lục, đề mục không có nội dung riêng), "
                            "không chứa nội dung chuyên môn để tóm tắt."
                        ),
                    }
                )
        return chapters, degraded_count, raw_count, auxiliary_sections, gate_degraded

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

        if len(content.strip()) < 150:
            body = content.strip() or title
            raw_passthrough = True
        else:
            prompt = (
                "You are a document analyst.\n\n"
                "TASK: Write a detailed summary of this book chapter for a library digest.\n"
                "Format: 2-5 sentences in Vietnamese covering main topics, methods, and findings.\n"
                "Preserve technical terms; add English/Russian originals in parentheses when helpful.\n\n"
                "CONSTRAINTS:\n"
                "- Every claim MUST be directly supported by the chapter text below.\n"
                "- Do NOT add external knowledge about the book, author, or subject — rely only "
                "on what this excerpt actually says.\n"
                "- Preserve numbers, names, dates, and technical terms verbatim.\n"
                "- If this excerpt is front matter (title page, author listing, table of "
                "contents) rather than real chapter content, say so briefly instead of "
                "inventing topics, methods, or findings it doesn't contain.\n\n"
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

        return (
            {
                "number": number,
                "title_vi": title_vi,
                "title_original": title_original,
                "content": body,
            },
            degraded,
            raw_passthrough,
        )

    async def _summarize_chapters(
        self,
        llm,
        nodes: List[dict],
        task_id: Optional[str],
        summarize=None,
    ) -> tuple[List[dict], int, int]:
        """Summarise chapters with bounded concurrency, preserving document
        order. Returns (chapters, degraded_count, raw_passthrough_count).
        This is the pipeline's longest stage — one-at-a-time chapter calls
        left the LLM idle between chapters on large books."""
        from services.translators._parallel import run_parallel

        summarize = summarize or self._summarize_chapter
        total = len(nodes)

        async def worker(_idx: int, item: dict):
            return await summarize(llm, item["node"], item["number"])

        def on_progress(pct: int, msg: str) -> None:
            self._progress(task_id, int(15 + (pct / 95) * 75), msg)

        results = await run_parallel(
            nodes,
            worker,
            parallelism=settings.ai_max_concurrent_requests,
            on_progress=on_progress,
            progress_label="Chapter",
        )
        chapters = [r[0] for r in results]
        degraded_chapters = sum(1 for r in results if r[1])
        raw_passthrough_chapters = sum(1 for r in results if r[2])
        return chapters, degraded_chapters, raw_passthrough_chapters

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
            auxiliary_sections = 0
            gate_degraded = False

            if tree_data:
                self._progress(task_id, 15, "Walking tree for chapters")
                nodes = _collect_chapter_nodes(tree_data)
                (
                    chapters,
                    degraded_chapters,
                    raw_passthrough_chapters,
                    auxiliary_sections,
                    gate_degraded,
                ) = await self._summarize_with_gate(llm, nodes, task_id)

            if not chapters:
                self._progress(task_id, 20, "Fallback: markdown headings")
                text = self._read_text(document_id)
                chapters = _parse_markdown_chapters(text)
                if not chapters:
                    excerpt = BaseEnricher(llm).truncate_to_tokens(text, 2000)
                    chapters = [
                        {
                            "number": 1,
                            "title_vi": "Tài liệu",
                            "title_original": "Document",
                            "content": excerpt[:1500],
                        }
                    ]

            details = {
                "chapters": chapters,
                "degraded_chapters": degraded_chapters,
                "raw_passthrough_chapters": raw_passthrough_chapters,
                "auxiliary_sections": auxiliary_sections,
                "gate_degraded": gate_degraded,
            }

            with db_manager.session() as db:
                if main_content_id:
                    mc = db.query(MainContent).filter(MainContent.id == main_content_id).first()
                    if mc:
                        mc.details = details
                        mc.status = "COMPLETED"
                else:
                    db.add(
                        MainContent(
                            document_id=document_id,
                            details=details,
                            status="COMPLETED",
                        )
                    )

            self._progress(task_id, 100, "Done")

            from services.export_service import export_service

            export_service.mark_digest_dirty(document_id)

            return details
        except Exception:
            _set_status("FAILED")
            raise
