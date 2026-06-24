"""
Summarization service.

Always runs hierarchical summarization:
- Primary path : tree-based (bottom-up walk of the document's TreeIndex)
- Fallback path: chunk-based map-reduce when no tree index exists yet

Output language is always Vietnamese (see config.settings.pipeline_output_lang_clause).
"""
from config.settings import pipeline_output_lang_clause, settings
from data.database import get_db_manager
from data.db_models import Summary
from services.base_service import BaseTaskService
from services.task_manager import task_manager


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
            coro=self._summarize(document_id, summary_id),
        )
        return task_id, summary_id

    async def _summarize(self, document_id: str, summary_id: str = None):
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
            task_id = self._find_task_id(document_id, "HIERARCHICAL_SUMMARIZE")

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

            self._progress(task_id, 95, "Saving summary")

            with db_manager.session() as db:
                if summary_id:
                    s = db.query(Summary).filter(Summary.id == summary_id).first()
                    if s:
                        s.content = summary_text.strip() if summary_text else ""
                        s.status = "COMPLETED"
                else:
                    db.add(Summary(
                        document_id=document_id,
                        summary_type="hierarchical",
                        content=(summary_text or "").strip(),
                        status="COMPLETED",
                    ))

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

        async def summarise_node(node: dict) -> str:
            children = node.get("children", [])

            child_summaries = []
            for child in children:
                child_summary = await summarise_node(child)
                child_summaries.append(
                    f"- {child.get('title', 'Section')}: {child_summary}"
                )

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
                    "TASK: Synthesise a concise summary (2-4 sentences) for this section "
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
            except Exception:
                summary = own_content[:200] + ("..." if len(own_content) > 200 else "")

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
        enricher = BaseEnricher(llm)
        chunks = enricher.chunk_text(text, max_tokens=settings.ai_chunk_tokens)
        lang_clause = pipeline_output_lang_clause()

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
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
            chunk_summaries.append(s.strip())
            pct = int(10 + ((i + 1) / len(chunks)) * 75)
            self._progress(task_id, pct, f"Summarized section {i + 1}/{len(chunks)}")

        combined = "\n\n".join(
            f"### Section {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
        )
        final_prompt = (
            "You are a document analysis assistant.\n\n"
            "TASK: Below are section-by-section summaries of a document. "
            "Write a comprehensive summary in markdown.\n\n"
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


# ── Helpers ──────────────────────────────────────────────────────────

def _count_nodes(node: dict) -> int:
    count = 1
    for child in node.get("children", []):
        count += _count_nodes(child)
    return count
