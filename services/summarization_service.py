"""
Summarization service.

Generates short, detailed, or hierarchical tree-based summaries of document text.
"""
from data.database import get_db_manager
from data.db_models import Summary
from services.base_service import BaseTaskService
from services.task_manager import task_manager, TaskManager


class SummarizationService(BaseTaskService):
    """Document summarization (background task)."""

    def submit(self, db, document_id: str, summary_type: str = "short") -> str:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")
        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="SUMMARIZE",
            coro=self._summarize(document_id, summary_type),
        )
        return task_id

    async def _summarize(self, document_id: str, summary_type: str):
        db_manager = get_db_manager()

        # ── Hierarchical tree summarisation (new path) ───────────────
        if summary_type == "hierarchical":
            return await self._hierarchical_tree_summarize(document_id)

        text = self._read_text(document_id)
        task_id = self._find_task_id(document_id, "SUMMARIZE")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        # Choose prompt based on type
        if summary_type == "short":
            prompt = (
                "Provide a concise summary of the following document in 3-5 sentences. "
                "Focus on the main topic, key findings, and conclusions.\n\n"
                f"Document:\n{text[:12000]}\n\nSummary:"
            )
        else:
            prompt = (
                "Provide a comprehensive detailed summary of the following document. "
                "Include section-by-section breakdown, key arguments, methodology, "
                "findings, and conclusions. Use markdown formatting.\n\n"
                f"Document:\n{text[:20000]}\n\nDetailed Summary:"
            )

        self._progress(task_id, 30, "Generating summary")

        # For very long documents, do hierarchical summarization
        if llm.count_tokens(text) > 10000 and summary_type == "detailed":
            summary_text = await self._hierarchical_summarize(llm, text, task_id)
        else:
            summary_text = await llm.chat_completion(prompt)

        self._progress(task_id, 90, "Saving summary")

        # Store
        with db_manager.session() as db:
            s = Summary(
                document_id=document_id,
                summary_type=summary_type,
                content=summary_text.strip(),
            )
            db.add(s)

        return {"summary_type": summary_type, "length": len(summary_text)}

    async def _hierarchical_summarize(self, llm, text: str, task_id: str = None):
        """Summarize chunks, then summarize the summaries."""
        from pageindex.enrichment.base import BaseEnricher
        enricher = BaseEnricher(llm)
        chunks = enricher.chunk_text(text, max_tokens=6000)

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"Summarize this section of a document:\n\n{chunk}\n\nSection summary:"
            )
            s = await llm.chat_completion(prompt)
            chunk_summaries.append(s.strip())
            pct = int(30 + ((i + 1) / len(chunks)) * 50)
            self._progress(task_id, pct, f"Summarizing chunk {i+1}/{len(chunks)}")

        combined = "\n\n".join(chunk_summaries)
        final_prompt = (
            "Below are section summaries of a document. Provide a comprehensive "
            "overall summary that synthesizes all sections. Use markdown formatting.\n\n"
            f"{combined}\n\nOverall Summary:"
        )
        return await llm.chat_completion(final_prompt)

    async def _hierarchical_tree_summarize(self, document_id: str) -> dict:
        """
        Walk the document's tree index BOTTOM-UP and generate LLM summaries
        at every node.  Parent nodes synthesise from their own text plus
        their children's summaries.
        """
        db_manager = get_db_manager()

        task_id = self._find_task_id(document_id, "SUMMARIZE")

        # ── Load tree index ─────────────────────────────────────────
        with db_manager.session() as db:
            from data.db_models import TreeIndex
            tree_index = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )
            if not tree_index:
                raise ValueError(
                    "No tree index found — build a tree index first."
                )
            tree_data = dict(tree_index.tree_data)
            tree_index_id = tree_index.id

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        # ── Walk tree bottom-up ─────────────────────────────────────
        total_nodes = _count_nodes(tree_data)
        processed = [0]  # mutable counter for closure

        async def summarise_node(node: dict) -> str:
            """Recursively summarise children first, then this node."""
            children = node.get("children", [])

            # Recurse into children first (bottom-up)
            child_summaries = []
            for child in children:
                child_summary = await summarise_node(child)
                child_summaries.append(f"- {child.get('title', 'Section')}: {child_summary}")

            own_content = (
                node.get("content")
                or node.get("text")
                or node.get("text_content")
                or ""
            )

            if child_summaries:
                synthesis_input = ""
                if own_content.strip():
                    synthesis_input += f"Section text:\n{own_content[:2000]}\n\n"
                synthesis_input += "Child section summaries:\n" + "\n".join(child_summaries)
                prompt = (
                    "Synthesise a concise summary (2-4 sentences) for this section, "
                    "incorporating its own content and its sub-sections:\n\n"
                    f"{synthesis_input}\n\nSummary:"
                )
            elif own_content.strip():
                prompt = (
                    "Summarise this section in 1-3 sentences:\n\n"
                    f"{own_content[:2000]}\n\nSummary:"
                )
            else:
                node["summary"] = node.get("title", "")
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
                self._progress(task_id, pct, f"Summarised {processed[0]}/{total_nodes} nodes")

            return summary

        self._progress(task_id, 5, "Starting tree summarisation")

        document_summary = await summarise_node(tree_data)

        # ── Persist updated tree_data ───────────────────────────────
        with db_manager.session() as db:
            from data.db_models import TreeIndex
            tree_index = db.query(TreeIndex).filter(TreeIndex.id == tree_index_id).first()
            if tree_index:
                tree_index.tree_data = tree_data

        self._progress(task_id, 92, "Saving summary")

        # ── Store document-level summary record ─────────────────────
        with db_manager.session() as db:
            s = Summary(
                document_id=document_id,
                summary_type="hierarchical",
                content=document_summary,
            )
            db.add(s)

        self._progress(task_id, 100, "Done")

        return {
            "summary_type": "hierarchical",
            "nodes_summarised": processed[0],
            "length": len(document_summary),
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _count_nodes(node: dict) -> int:
    """Count all nodes in a tree dict recursively."""
    count = 1
    for child in node.get("children", []):
        count += _count_nodes(child)
    return count

