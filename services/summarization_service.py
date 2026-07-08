"""
Summarization service.

Always runs hierarchical summarization:
- Primary path : tree-based (bottom-up walk of the document's TreeIndex)
- Fallback path: chunk-based map-reduce when no tree index exists yet

Output language is always Vietnamese (see config.settings.pipeline_output_lang_clause).
"""
import asyncio
import logging
from typing import Optional

from config.settings import pipeline_output_lang_clause, settings
from data.database import get_db_manager
from data.db_models import Summary
from services.base_service import BaseTaskService
from services.task_manager import task_manager

logger = logging.getLogger(__name__)


class SummarizationService(BaseTaskService):
    """Document summarization (background task, always hierarchical)."""

    def submit(self, db, document_id: str, summary_type: str = "short") -> str:
        """
        Submit a summarization task.  `summary_type` is accepted for API
        backwards-compatibility but is ignored — the service always runs the
        hierarchical pipeline (tree-based if a TreeIndex exists, else chunk-based).

        Creates a Summary record with status=PENDING immediately so that
        clients GET-ing /summaries see the job appear at once.
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        existing_task = task_manager.get_active_task_id(
            db, document_id, "HIERARCHICAL_SUMMARIZE"
        )
        if existing_task:
            summary = (
                db.query(Summary)
                .filter(Summary.document_id == document_id)
                .order_by(Summary.created_at.desc())
                .first()
            )
            return existing_task, (summary.id if summary else None), True

        summary = Summary(
            document_id=document_id,
            summary_type="hierarchical",
            status="PENDING",
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        summary_id = summary.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="HIERARCHICAL_SUMMARIZE",
            coro_factory=lambda tid: self._summarize(document_id, summary_id, tid),
        )
        return task_id, summary_id, False

    async def run_for_pipeline(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        with db_manager.session() as db:
            s = (
                db.query(Summary)
                .filter(Summary.document_id == document_id)
                .order_by(Summary.created_at.desc())
                .first()
            )
            if not s:
                s = Summary(
                    document_id=document_id,
                    summary_type="hierarchical",
                    status="PENDING",
                )
                db.add(s)
                db.commit()
                db.refresh(s)
            summary_id = s.id
        return await self._summarize(document_id, summary_id, task_id)

    async def _summarize(
        self,
        document_id: str,
        summary_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()

        def _set_status(status: str):
            if not summary_id:
                return
            with db_manager.session() as db:
                s = db.query(Summary).filter(Summary.id == summary_id).first()
                if s:
                    s.status = status

        _set_status("IN_PROGRESS")

        try:
            await self._wait_for_digitized_text(document_id, task_id=task_id)

            from api.dependencies import get_llm_client
            llm = get_llm_client()

            # ── Try tree-based first ─────────────────────────────────────
            with db_manager.session() as db:
                from data.db_models import TreeIndex
                tree_index = (
                    db.query(TreeIndex)
                    .filter(TreeIndex.document_id == document_id)
                    .order_by(TreeIndex.created_at.desc())
                    .first()
                )
                has_tree = tree_index is not None

            if has_tree:
                summary_text, meta = await self._hierarchical_tree_summarize(
                    document_id, llm, task_id
                )
            else:
                # ── Fallback: chunk-based map-reduce ─────────────────────
                text = self._read_text(document_id)
                if not text or not text.strip():
                    raise ValueError(
                        "No text content found for this document. "
                        "Run Extract first before summarizing."
                    )
                self._progress(task_id, 5, "No tree index — using chunk-based summarization")
                summary_text = await self._chunk_summarize(llm, text, task_id)
                meta = {"summary_type": "chunk_based", "length": len(summary_text)}

            self._progress(task_id, 98, "Saving summary")

            with db_manager.session() as db:
                if summary_id:
                    s = db.query(Summary).filter(Summary.id == summary_id).first()
                    if s:
                        s.content = summary_text.strip() if summary_text else ""
                        s.status = "IN_PROGRESS"
                else:
                    db.add(Summary(
                        document_id=document_id,
                        summary_type="hierarchical",
                        content=(summary_text or "").strip(),
                        status="IN_PROGRESS",
                    ))
                    db.flush()
                    summary_id = (
                        db.query(Summary)
                        .filter(Summary.document_id == document_id)
                        .order_by(Summary.created_at.desc())
                        .first()
                        .id
                    )

            self._progress(task_id, 99, "Preparing download export…")

            from services.export_service import export_service

            export_service.mark_digest_dirty(document_id)
            if summary_id:
                await export_service.cache_summary_export(document_id, summary_id)

            with db_manager.session() as db:
                if summary_id:
                    s = db.query(Summary).filter(Summary.id == summary_id).first()
                    if s:
                        s.status = "COMPLETED"

            self._progress(task_id, 100, "Done")

            return meta
        except Exception:
            _set_status("FAILED")
            raise

    # ── Tree-based (primary) ─────────────────────────────────────────

    async def _hierarchical_tree_summarize(
        self, document_id: str, llm, task_id: str = None
    ) -> tuple:
        """
        Walk the document's TreeIndex BOTTOM-UP and generate LLM summaries at
        every node.  Parent nodes synthesise from their own text plus their
        children's summaries.  Output is always Vietnamese.
        """
        db_manager = get_db_manager()

        with db_manager.session() as db:
            from data.db_models import TreeIndex
            tree_index = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )
            tree_data = dict(tree_index.tree_data)
            tree_index_id = tree_index.id

        total_nodes = _count_nodes(tree_data)
        processed = [0]
        degraded = [0]

        async def summarise_node(node: dict) -> str:
            children = node.get("children", [])

            # Siblings are independent — only this node's own synthesis needs
            # to wait for all of them, so summarise them concurrently instead
            # of one at a time.
            child_results = await asyncio.gather(
                *(summarise_node(child) for child in children)
            )
            child_summaries = [
                f"- {child.get('title', 'Section')}: {child_summary}"
                for child, child_summary in zip(children, child_results)
            ]

            own_content = (
                node.get("content")
                or node.get("text")
                or node.get("text_content")
                or ""
            )

            # Reuse existing node summary if already populated (e.g. from build step)
            existing_summary = node.get("summary")
            if existing_summary and isinstance(existing_summary, str) and existing_summary.strip():
                processed[0] += 1
                return existing_summary

            lang_clause = pipeline_output_lang_clause()
            if child_summaries:
                synthesis_input = ""
                if own_content.strip():
                    synthesis_input += f"Section text:\n{own_content[:2000]}\n\n"
                synthesis_input += (
                    "Sub-section summaries:\n" + "\n".join(child_summaries)
                )
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    "TASK: Synthesise a document-level abstract (10-15 sentences) for this section "
                    "that preserves all key facts, arguments, and domain-specific terms.\n\n"
                    "CONSTRAINTS:\n"
                    "- Every factual claim MUST be directly supported in the source text below.\n"
                    "- Do NOT add external knowledge, interpretation, or inference.\n"
                    "- Preserve all numbers, names, dates, and domain terms verbatim.\n"
                    "- If the source lacks sufficient information, write a short summary anyway "
                    "— but never fabricate.\n\n"
                    f"{lang_clause}\n\n"
                    f"{synthesis_input}\n\nSummary:"
                )
            elif own_content.strip():
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    "TASK: Summarise this section in 1-3 sentences, preserving all key "
                    "facts, findings, and domain-specific terms.\n\n"
                    "CONSTRAINTS:\n"
                    "- Every claim MUST be directly supported in the source text below.\n"
                    "- Do NOT add external knowledge or interpretation.\n"
                    "- Preserve numbers, names, dates, and technical terms verbatim.\n\n"
                    f"{lang_clause}\n\n"
                    f"{own_content[:2000]}\n\nSummary:"
                )
            else:
                node["summary"] = node.get("title", "")
                processed[0] += 1
                return node["summary"]

            try:
                summary = await llm.chat_completion(prompt)
                summary = summary.strip()
            except Exception as exc:
                summary = own_content[:200] + ("..." if len(own_content) > 200 else "")
                degraded[0] += 1
                logger.warning(
                    "Node summarization failed for document %s, node %r: %s",
                    document_id,
                    node.get("title"),
                    exc,
                )

            node["summary"] = summary
            processed[0] += 1

            if total_nodes > 0:
                pct = min(90, int((processed[0] / total_nodes) * 85) + 5)
                self._progress(
                    task_id, pct, f"Summarised {processed[0]}/{total_nodes} nodes"
                )
            return summary

        self._progress(task_id, 5, "Starting tree summarisation")
        document_summary = await summarise_node(tree_data)

        # Persist node-level summaries back into the tree
        with db_manager.session() as db:
            from data.db_models import TreeIndex
            ti = db.query(TreeIndex).filter(TreeIndex.id == tree_index_id).first()
            if ti:
                ti.tree_data = tree_data

        self._progress(task_id, 92, "Tree summary done")

        return document_summary, {
            "summary_type": "hierarchical",
            "nodes_summarised": processed[0],
            "degraded_nodes": degraded[0],
            "length": len(document_summary),
        }

    # ── Chunk-based fallback ─────────────────────────────────────────

    async def _chunk_summarize(
        self, llm, text: str, task_id: str = None
    ) -> str:
        """
        Map-reduce over chunks sized to the model's context window.
        Used when no tree index exists.
        """
        from core.pageindex.enrichment.base import BaseEnricher
        from services.translators._parallel import run_parallel

        enricher = BaseEnricher(llm)
        chunks = enricher.chunk_text(text, max_tokens=settings.ai_chunk_tokens)
        lang_clause = pipeline_output_lang_clause()

        async def _summarise_chunk(_idx: int, chunk: str) -> str:
            prompt = (
                "You are a document analysis assistant.\n\n"
                "TASK: Read the following section and write a thorough summary.\n\n"
                "WHAT TO INCLUDE:\n"
                "- Preserve all key facts, arguments, data, findings, and domain-specific "
                "terms exactly as they appear\n"
                "- Capture the logical structure (claims → evidence → conclusion where present)\n\n"
                "WHAT TO AVOID:\n"
                "- Do NOT add information not present in the section\n"
                "- Do NOT interpret or infer beyond what is stated\n"
                "- Do NOT omit key quantitative data (numbers, percentages, comparisons)\n\n"
                f"{lang_clause}\n\n"
                f"Section:\n{chunk}\n\nSection Summary:"
            )
            s = await llm.chat_completion(prompt)
            return s.strip()

        async def _on_progress(pct: int, msg: str) -> None:
            self._progress(task_id, 10 + int(pct * 0.45), msg)

        chunk_summaries = await run_parallel(
            chunks,
            _summarise_chunk,
            parallelism=settings.ai_max_concurrent_requests,
            on_progress=_on_progress,
            progress_label="Summarized section",
        )

        entries = [
            f"### Section {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
        ]
        combined = await self._collapse_to_budget(
            enricher, llm, entries, lang_clause, task_id
        )
        final_prompt = (
            "You are a document analysis assistant.\n\n"
            "TASK: Below are section-by-section summaries of a document. "
            "Write a comprehensive summary in markdown for the library digest abstract (§2.1).\n\n"
            "LENGTH: Write 10 to 15 complete sentences.\n\n"
            "WHAT TO INCLUDE:\n"
            "- Preserve all key facts, arguments, results, and conclusions from every section\n"
            "- Group related ideas — use headings for major themes if the content warrants it\n"
            "- Keep numbers, dates, names, and domain terms verbatim\n\n"
            "WHAT TO AVOID:\n"
            "- Do NOT add information not present in the section summaries\n"
            "- Do NOT invent data, statistics, or references\n"
            "- Do NOT merge distinct sections into one paragraph if they cover different topics\n\n"
            f"{lang_clause}\n\n"
            f"Section Summaries:\n{combined}\n\nComprehensive Summary:"
        )
        self._progress(task_id, 88, "Synthesizing final summary")
        return await llm.chat_completion(final_prompt)

    async def _collapse_to_budget(
        self,
        enricher,
        llm,
        entries: list,
        lang_clause: str,
        task_id: str = None,
        max_rounds: int = 5,
    ) -> str:
        """Recursively merge section summaries in budget-sized batches until
        the combined text fits the model's input budget.

        A flat concatenation of every chunk-summary can itself overflow the
        budget once there are enough chunks — exactly the large-document
        case this fallback exists for. Each round packs entries into
        budget-sized batches and asks the LLM to merge each batch into one
        shorter entry, converging toward a single combined block.
        """
        from services.translators._parallel import run_parallel

        budget = settings.ai_input_budget_tokens
        combined = "\n\n".join(entries)
        round_num = 0

        while (
            enricher.count_tokens(combined) > budget
            and len(entries) > 1
            and round_num < max_rounds
        ):
            round_num += 1
            batches = self._pack_into_batches(enricher, entries, budget)

            async def _collapse_batch(_idx: int, batch_entries: list) -> str:
                batch_text = "\n\n".join(batch_entries)
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    "TASK: Merge the following section summaries into a single, more "
                    "concise summary that preserves all key facts, arguments, data, "
                    "and domain terms.\n\n"
                    "WHAT TO AVOID:\n"
                    "- Do NOT add information not present in the summaries below\n"
                    "- Do NOT drop distinct topics — condense wording, not content\n\n"
                    f"{lang_clause}\n\n"
                    f"{batch_text}\n\nMerged Summary:"
                )
                s = await llm.chat_completion(prompt)
                return s.strip()

            entries = await run_parallel(
                batches,
                _collapse_batch,
                parallelism=settings.ai_max_concurrent_requests,
            )
            combined = "\n\n".join(entries)
            self._progress(
                task_id,
                min(85, 55 + round_num * 5),
                f"Merging summaries (round {round_num})",
            )

        if enricher.count_tokens(combined) > budget:
            combined = enricher.truncate_to_tokens(combined, budget)

        return combined

    @staticmethod
    def _pack_into_batches(enricher, entries: list, budget: int) -> list:
        """Greedily group entries into budget-sized batches."""
        batches: list = []
        current: list = []
        current_tokens = 0
        for entry in entries:
            entry_tokens = enricher.count_tokens(entry)
            if entry_tokens > budget:
                entry = enricher.truncate_to_tokens(entry, budget)
                entry_tokens = budget
            if current and current_tokens + entry_tokens > budget:
                batches.append(current)
                current = [entry]
                current_tokens = entry_tokens
            else:
                current.append(entry)
                current_tokens += entry_tokens
        if current:
            batches.append(current)
        return batches


# ── Helpers ──────────────────────────────────────────────────────────

def _count_nodes(node: dict) -> int:
    count = 1
    for child in node.get("children", []):
        count += _count_nodes(child)
    return count
