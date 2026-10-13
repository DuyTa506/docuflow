"""
Summarization service.

Always runs hierarchical summarization:
- Primary path : tree-based (bottom-up walk of the document's TreeIndex)
- Fallback path: chunk-based map-reduce when no tree index exists yet

Output language is always Vietnamese (see config.settings.pipeline_output_lang_clause).
"""

import asyncio
import logging
import re
from typing import Optional

from config.settings import pipeline_output_lang_clause, settings
from core.spatial.zone_classifier import split_chapter_heading
from data.database import get_db_manager
from data.db_models import Summary
from services.base_service import BaseTaskService
from services.domain_context import current_domain_clause, current_glossary, use_domain_context
from services.task_manager import task_manager
from services.term_consistency import enforce_termbase
from utils.doc_kind import BOOK, PROCEEDINGS, resolve_doc_kind
from utils.tree_payload import get_tree_payload

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

        existing_task = task_manager.get_active_task_id(db, document_id, "HIERARCHICAL_SUMMARIZE")
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

    async def submit_async(self, db, document_id: str, summary_type: str = "short") -> tuple:
        """Temporal-aware submit. Summarising a book-length document runs for
        hours; in-process it had no timeout, heartbeat, retry or resume."""
        from config.settings import settings

        if not settings.stage_rerun_use_temporal:
            return self.submit(db, document_id, summary_type)

        from services.stage_dispatch import submit_stage_with_resource

        return await submit_stage_with_resource(
            db,
            document_id,
            "HIERARCHICAL_SUMMARIZE",
            Summary,
            summary_type="hierarchical",
        )

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
        use_domain_context(document_id)

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
                self._progress(task_id, 5, "Không có cây mục lục — dùng tóm tắt theo từng phần")
                summary_text = await self._chunk_summarize(llm, text, task_id)
                meta = {"summary_type": "chunk_based", "length": len(summary_text)}

            self._progress(task_id, 98, "Đang lưu bản tóm tắt")

            with db_manager.session() as db:
                if summary_id:
                    s = db.query(Summary).filter(Summary.id == summary_id).first()
                    if s:
                        s.content = summary_text.strip() if summary_text else ""
                        s.status = "IN_PROGRESS"
                else:
                    db.add(
                        Summary(
                            document_id=document_id,
                            summary_type="hierarchical",
                            content=(summary_text or "").strip(),
                            status="IN_PROGRESS",
                        )
                    )
                    db.flush()
                    summary_id = (
                        db.query(Summary)
                        .filter(Summary.document_id == document_id)
                        .order_by(Summary.created_at.desc())
                        .first()
                        .id
                    )

            self._progress(task_id, 99, "Đang chuẩn bị file tải xuống…")

            from services.export_service import export_service

            export_service.mark_digest_dirty(document_id)
            if summary_id:
                await export_service.cache_summary_export(document_id, summary_id)

            with db_manager.session() as db:
                if summary_id:
                    s = db.query(Summary).filter(Summary.id == summary_id).first()
                    if s:
                        s.status = "COMPLETED"

            self._progress(task_id, 100, "Hoàn tất")

            return meta
        except Exception:
            _set_status("FAILED")
            raise

    # ── Tree-based (primary) ─────────────────────────────────────────

    async def _hierarchical_tree_summarize(
        self, document_id: str, llm, task_id: str = None
    ) -> tuple:
        """
        Walk the document's TreeIndex BOTTOM-UP (height-ordered levels, bounded
        concurrency per level) and generate LLM summaries at every node.
        Parent nodes synthesise from their own text plus their children's
        summaries.  Above `summarize_cluster_threshold` total nodes, leaf
        siblings are batched per parent into one prompt per cluster.  Node
        summaries are checkpointed back to the tree every
        `summarize_checkpoint_nodes` nodes so a retry resumes via the
        existing-summary reuse instead of redoing the whole stage.
        Output is always Vietnamese.
        """
        from core.pageindex.enrichment.base import BaseEnricher
        from services.translators._parallel import run_parallel

        db_manager = get_db_manager()

        with db_manager.session() as db:
            from data.db_models import TreeIndex

            tree_index = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )
            tree_data = get_tree_payload(db, tree_index)
            tree_index_id = tree_index.id

        total_nodes = _count_nodes(tree_data)
        processed = [0]
        degraded = [0]
        persist_lock = asyncio.Lock()

        enricher = BaseEnricher(llm)
        node_budget = settings.summarize_node_content_tokens
        checkpoint_every = settings.summarize_checkpoint_nodes
        parallelism = settings.ai_max_concurrent_requests
        lang_clause = pipeline_output_lang_clause()

        def _own_content(node: dict) -> str:
            return node.get("content") or node.get("text") or node.get("text_content") or ""

        def _has_summary(node: dict) -> bool:
            s = node.get("summary")
            return bool(s and isinstance(s, str) and s.strip())

        async def _mark_done() -> None:
            processed[0] += 1
            if total_nodes > 0:
                pct = min(90, int((processed[0] / total_nodes) * 85) + 5)
                self._progress(
                    task_id,
                    pct,
                    f"Đã tóm tắt {processed[0]}/{total_nodes} nút",
                    unit_kind="tree_node",
                    units_done=processed[0],
                    units_total=total_nodes,
                )
            if checkpoint_every and processed[0] % checkpoint_every == 0:
                async with persist_lock:
                    _persist_tree(tree_index_id, tree_data)

        async def _summarise_single(node: dict) -> None:
            children = node.get("children", [])
            child_summaries = [
                f"- {c.get('title', 'Section')}: {c.get('summary', '')}"
                for c in children
                if _has_summary(c)
            ]
            own_content = _own_content(node)
            own_excerpt = enricher.truncate_to_tokens(own_content, node_budget)

            if child_summaries:
                synthesis_input = ""
                if own_content.strip():
                    synthesis_input += f"Section text:\n{own_excerpt}\n\n"
                # Unbounded on a wide tree (e.g. a root node with hundreds of
                # direct chapters) this alone can dwarf the context window —
                # observed 104k/161k-token synthesis calls on real books,
                # which the LLM hard-rejects and the except-fallback below
                # silently turns into an EMPTY document abstract.
                child_block = enricher.truncate_to_tokens("\n".join(child_summaries), batch_budget)
                synthesis_input += "Sub-section summaries:\n" + child_block
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    f"{current_domain_clause(synthesis_input)}"
                    "TASK: Synthesise a document-level abstract (10-15 sentences) for this section "
                    "that preserves all key facts, arguments, and domain-specific terms.\n\n"
                    "CONSTRAINTS:\n"
                    "- Every factual claim MUST be directly supported in the source text below.\n"
                    "- Do NOT add external knowledge, interpretation, or inference.\n"
                    "- Preserve all numbers, names, dates, and domain terms verbatim.\n"
                    "- If the source lacks sufficient information, write a short summary anyway "
                    "— but never fabricate.\n\n"
                    f"{lang_clause}\n\n"
                    f"{synthesis_input}\n\n"
                    # Repeated right before the generation cue: confirmed live
                    # against the pipeline LLM (qwen3.5-9b) that a language
                    # instruction stated only once, followed by a large
                    # non-Vietnamese source block, gets ignored in favor of
                    # mirroring the source language.
                    f"{lang_clause}\n\nSummary:"
                )
            elif own_content.strip():
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    f"{current_domain_clause(own_excerpt)}"
                    "TASK: Summarise this section in 1-3 sentences, preserving all key "
                    "facts, findings, and domain-specific terms.\n\n"
                    "CONSTRAINTS:\n"
                    "- Every claim MUST be directly supported in the source text below.\n"
                    "- Do NOT add external knowledge or interpretation.\n"
                    "- Preserve numbers, names, dates, and technical terms verbatim.\n\n"
                    f"{lang_clause}\n\n"
                    f"{own_excerpt}\n\n"
                    f"{lang_clause}\n\nSummary:"
                )
            else:
                node["summary"] = node.get("title", "")
                await _mark_done()
                return

            try:
                summary = (await llm.chat_completion(prompt)).strip()
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
            await _mark_done()

        async def _summarise_batch(batch: list) -> None:
            if len(batch) == 1:
                await _summarise_single(batch[0])
                return
            sections = []
            for i, node in enumerate(batch, 1):
                excerpt = enricher.truncate_to_tokens(_own_content(node), node_budget)
                sections.append(f"SECTION n{i}: {node.get('title', '')}\n{excerpt}")
            prompt = (
                "You are a document analysis assistant.\n\n"
                f"{current_domain_clause(chr(10).join(sections))}"
                f"TASK: Summarise EACH of the {len(batch)} sections below INDEPENDENTLY "
                "in 1-3 sentences, preserving all key facts, findings, and "
                "domain-specific terms.\n\n"
                "CONSTRAINTS:\n"
                "- Every claim MUST be directly supported by that section's own text.\n"
                "- Do NOT add external knowledge or interpretation.\n"
                "- Preserve numbers, names, dates, and technical terms verbatim.\n\n"
                f"{lang_clause}\n\n" + "\n\n".join(sections) + f"\n\n{lang_clause}\n\n"
                'Return ONLY valid JSON: {"summaries": [{"id": "n1", "summary": "..."}, ...]}\n'
                "JSON:"
            )
            mapping: dict = {}
            try:
                response = await llm.chat_completion(prompt)
                data = llm.extract_json(response)
                items = data.get("summaries") if isinstance(data, dict) else data
                for item in items or []:
                    if isinstance(item, dict) and item.get("summary"):
                        mapping[str(item.get("id"))] = str(item["summary"]).strip()
            except Exception as exc:
                logger.warning(
                    "Cluster batch summarization failed for document %s (%s) — "
                    "falling back to per-node prompts",
                    document_id,
                    exc,
                )
            for i, node in enumerate(batch, 1):
                mapped = mapping.get(f"n{i}")
                if mapped:
                    node["summary"] = mapped
                    await _mark_done()
                else:
                    await _summarise_single(node)

        cluster_mode = total_nodes > settings.summarize_cluster_threshold
        levels, parents = _levels_with_parents(tree_data)
        batch_budget = max(1000, settings.ai_input_budget_tokens - 1500)

        self._progress(
            task_id,
            5,
            f"Bắt đầu tóm tắt cây mục lục ({total_nodes} nút"
            + (", chế độ gom cụm)" if cluster_mode else ")"),
        )

        for level_idx, level_nodes in enumerate(levels):
            pending = [n for n in level_nodes if not _has_summary(n)]
            if not pending:
                continue

            if level_idx == 0 and cluster_mode:
                content_leaves = []
                for n in pending:
                    if _own_content(n).strip():
                        content_leaves.append(n)
                    else:
                        n["summary"] = n.get("title", "")
                        await _mark_done()
                # Cluster leaves per parent so each batch shares local context.
                groups: dict = {}
                order: list = []
                for n in content_leaves:
                    key = id(parents.get(id(n), tree_data))
                    if key not in groups:
                        groups[key] = []
                        order.append(key)
                    groups[key].append(n)
                batches: list = []
                for key in order:
                    batches.extend(
                        _pack_leaf_batches(
                            groups[key],
                            enricher,
                            node_budget,
                            batch_budget,
                            settings.summarize_cluster_max_nodes,
                        )
                    )
                await run_parallel(
                    batches,
                    lambda _i, b: _summarise_batch(b),
                    parallelism=parallelism,
                )
            else:
                await run_parallel(
                    pending,
                    lambda _i, n: _summarise_single(n),
                    parallelism=parallelism,
                )

        # Deterministic resolution only: this stage runs in parallel with
        # MAIN_CONTENT, so it cannot read the kind that stage settles on, and a
        # second LLM classification could disagree with it. A kỷ yếu whose
        # language the vocabulary misses gets book-worded §2.1 — the quality
        # report flags the mismatch rather than letting it pass unnoticed.
        document_summary = await compose_document_summary(
            llm, tree_data, doc_kind=_document_kind(document_id)
        )

        # Persist node-level summaries back into the tree (final checkpoint)
        _persist_tree(tree_index_id, tree_data)

        self._progress(task_id, 92, "Tóm tắt cây mục lục hoàn tất")

        return document_summary, {
            "summary_type": "hierarchical",
            "nodes_summarised": processed[0],
            "degraded_nodes": degraded[0],
            "cluster_mode": cluster_mode,
            "length": len(document_summary),
        }

    # ── Chunk-based fallback ─────────────────────────────────────────

    async def _chunk_summarize(self, llm, text: str, task_id: str = None) -> str:
        """
        Map-reduce over chunks sized to the model's context window.
        Used when no tree index exists.
        """
        from core.pageindex.enrichment.base import BaseEnricher
        from services.translators._parallel import run_parallel

        enricher = BaseEnricher(llm)
        chunks = enricher.chunk_text(text, max_tokens=settings.ai_input_budget_tokens)
        lang_clause = pipeline_output_lang_clause()

        async def _summarise_chunk(_idx: int, chunk: str) -> str:
            prompt = (
                "You are a document analysis assistant.\n\n"
                f"{current_domain_clause(chunk)}"
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
                f"Section:\n{chunk}\n\n"
                f"{lang_clause}\n\nSection Summary:"
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
            progress_label="Phần đã tóm tắt",
        )

        entries = [f"### Section {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)]
        combined = await self._collapse_to_budget(enricher, llm, entries, lang_clause, task_id)
        final_prompt = (
            "You are a document analysis assistant.\n\n"
            f"{current_domain_clause(combined)}"
            "TASK: Below are section-by-section summaries of a document. "
            # §2.1 lands in a Word paragraph, which has no markdown. Asking for
            # markdown here printed `##` and `**` literally in the digest.
            "Write a comprehensive summary for the library digest abstract (§2.1).\n\n"
            "LENGTH: Write 10 to 15 complete sentences.\n\n"
            "WHAT TO INCLUDE:\n"
            "- Preserve all key facts, arguments, results, and conclusions from every section\n"
            "- Group related ideas into paragraphs, one theme per paragraph\n"
            "- Keep numbers, dates, names, and domain terms verbatim\n\n"
            "WHAT TO AVOID:\n"
            "- Do NOT use markdown: no `#` headings, no `**bold**`, no bullet lists\n"
            "- Do NOT add information not present in the section summaries\n"
            "- Do NOT invent data, statistics, or references\n"
            "- Do NOT merge distinct sections into one paragraph if they cover different topics\n\n"
            f"{lang_clause}\n\n"
            f"Section Summaries:\n{combined}\n\n"
            f"{lang_clause}\n\nComprehensive Summary:"
        )
        self._progress(task_id, 88, "Đang tổng hợp bản tóm tắt cuối")
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
            enricher.count_tokens(combined) > budget and len(entries) > 1 and round_num < max_rounds
        ):
            round_num += 1
            batches = self._pack_into_batches(enricher, entries, budget)

            async def _collapse_batch(_idx: int, batch_entries: list) -> str:
                batch_text = "\n\n".join(batch_entries)
                prompt = (
                    "You are a document analysis assistant.\n\n"
                    f"{current_domain_clause(batch_text)}"
                    "TASK: Merge the following section summaries into a single, more "
                    "concise summary that preserves all key facts, arguments, data, "
                    "and domain terms.\n\n"
                    "WHAT TO AVOID:\n"
                    "- Do NOT add information not present in the summaries below\n"
                    "- Do NOT drop distinct topics — condense wording, not content\n\n"
                    f"{lang_clause}\n\n"
                    f"{batch_text}\n\n"
                    f"{lang_clause}\n\nMerged Summary:"
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
                f"Đang gộp các bản tóm tắt (vòng {round_num})",
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


def _document_kind(document_id: str) -> str:
    """Book or kỷ yếu, from the override plus title vocabulary. Never raises."""
    try:
        from data.database import get_db_manager
        from data.db_models import Document

        with get_db_manager().session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc is None:
                return BOOK
            return resolve_doc_kind(doc.digest_doc_kind, title=doc.title or "")["doc_kind"]
    except Exception as exc:
        logger.warning("Không đọc được thể loại tài liệu %s (%s) — coi là sách", document_id, exc)
        return BOOK


# Fewer top-level entries than this and §2.1 is outlined one level down: a
# journal article's tree is one root (its title) with the sections under it.
OUTLINE_MIN_ENTRIES = 3
# A top-level entry lists its children's titles (so a part can be said to span
# "chương 1 đến chương 5") only while the top level is small enough to be parts.
OUTLINE_MAX_PARTS = 8
OUTLINE_MAX_CHILD_TITLES = 25
# The prompt names "section 2.1 (Tóm tắt)" and the model echoes it as a first
# line; the digest adds its own heading, so §2.1 printed it twice.
_ECHOED_HEADING_RE = re.compile(r"^[#*\s]*(?:2\.1\.?\s*)?tóm tắt[\s:*]*\n", re.IGNORECASE)
SUMMARY_FORM_EXAMPLE = (
    "Tài liệu bao gồm 2 phần: Phần I - Cơ sở lý thuyết (bao gồm từ chương 1 đến chương 3), "
    "trình bày các khái niệm, mô hình và phương pháp nền tảng của lĩnh vực; Phần II - Ứng dụng "
    "(bao gồm từ chương 4 đến chương 6), áp dụng các nền tảng đó vào thiết kế và đánh giá hệ "
    "thống trong thực tế. Nội dung gồm các khái niệm cơ bản, phương pháp tính toán và các vấn "
    "đề thiết kế điển hình. Hai phần hợp thành một hệ thống kiến thức đi từ lý thuyết đến "
    "thực hành."
)


def _structure_outline(children: list, expand: bool) -> str:
    """The document's organisation for §2.1: top-level entries with their summaries,
    and under each either its sections' summaries (``expand``) or its chapter titles."""

    def kids(node: dict) -> list:
        return [
            c
            for c in (node.get("children") or node.get("child_nodes") or [])
            if isinstance(c, dict)
        ]

    lines = []
    for child in children:
        title = str(child.get("title") or "Section").strip()
        summary = str(child.get("summary") or "").strip()
        subs = kids(child)
        sub_summaries = [
            (str(c.get("title") or "").strip(), str(c.get("summary") or "").strip()) for c in subs
        ]
        sub_summaries = [(t, s) for t, s in sub_summaries if s]
        if not summary and not (expand and sub_summaries):
            continue
        lines.append(f"- {title}: {summary}" if summary else f"- {title}")
        if expand:
            lines += [f"    - {t or 'Mục'}: {s}" for t, s in sub_summaries]
        elif len(children) <= OUTLINE_MAX_PARTS:
            titles = [str(c.get("title") or "").strip() for c in subs]
            titles = [t for t in titles if t][:OUTLINE_MAX_CHILD_TITLES]
            if titles:
                lines.append(f"    (gồm: {'; '.join(titles)})")
    return "\n".join(lines)


def _roman(n: int) -> str:
    out = ""
    for value, sym in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def chapter_outline(units: list) -> str:
    """§2.1 outline from the chapter units §2.2 lists: Parts with their chapter range,
    then every chapter with its summary — so the two sections describe one structure."""

    def line(unit: dict) -> str:
        title = str(unit.get("title") or "").strip()
        name = split_chapter_heading(title)[1] or title
        number = unit.get("chapter_ordinal")
        head = f"Chương {number}. {name}" if number else name
        summary = str(unit.get("summary") or "").strip()
        return f"- {head}: {summary}" if summary else f"- {head}"

    parts: dict = {}
    for unit in units:
        part = unit.get("part")
        key = (part["ordinal"], part["name"]) if part else None
        parts.setdefault(key, []).append(unit)
    if set(parts) == {None}:
        return "\n".join(line(u) for u in units)

    lines = []
    for key, members in parts.items():
        if key is not None:
            numbers = [u["chapter_ordinal"] for u in members if u.get("chapter_ordinal")]
            span = ""
            if numbers:
                lo, hi = min(numbers), max(numbers)
                span = f" (chương {lo})" if lo == hi else f" (chương {lo} đến chương {hi})"
            lines.append(f"Phần {_roman(key[0])} - {key[1]}{span}:")
        lines.extend(line(u) for u in members)
    return "\n".join(lines)


def _chapter_units(tree_data: dict, doc_kind: str) -> list:
    """The units §2.2 will list, when they are a real chapter structure."""
    if doc_kind != BOOK:
        return []
    from utils.chapter_units import select_chapter_units

    try:
        units, meta = select_chapter_units(tree_data)
    except Exception as exc:  # the old outline still works
        logger.warning("§2.1 chapter outline unavailable: %s", exc)
        return []
    if meta.get("unit_selection_tier") == "mass_segmentation" or len(units) < 3:
        return []
    return units


async def compose_document_summary(llm, tree_data: dict, doc_kind: str = BOOK) -> str:
    """Return the document-level abstract for digest §2.1.

    Composed from the top-level node summaries in the digest form. The level
    walk also summarises the root, but with the generic section prompt; that
    text is only the fallback (nothing to compose from, or the call failed) —
    returning it as-is meant this prompt never ran.
    """
    existing = (tree_data.get("summary") or "").strip()

    children = [
        child
        for child in (tree_data.get("children") or tree_data.get("child_nodes") or [])
        if isinstance(child, dict)
    ]
    entries = [
        (str(child.get("title") or "Section").strip(), str(child.get("summary") or "").strip())
        for child in children
    ]
    entries = [(title, summary) for title, summary in entries if summary]
    if not entries:
        return existing

    from core.pageindex.enrichment.base import BaseEnricher

    enricher = BaseEnricher(llm)
    units = _chapter_units(tree_data, doc_kind)
    outline = (
        chapter_outline(units)
        if units
        else _structure_outline(children, expand=len(entries) < OUTLINE_MIN_ENTRIES)
    )
    outline = enricher.truncate_to_tokens(outline, max(1000, settings.ai_chunk_tokens - 1000))
    lang_clause = pipeline_output_lang_clause()
    # The official template words §2.1 differently for a kỷ yếu: "Tài liệu là
    # tuyển tập N bài báo khoa học (BBKH), chia thành M phần, bao gồm…". The
    # number of parts is countable here; the number of papers is not, so the
    # model is told to leave it out rather than produce a plausible figure.
    if doc_kind == PROCEEDINGS:
        kind_clause = (
            f"This document is a collection of independent scientific papers (BBKH) "
            f"grouped into {len(entries)} parts, not a single continuous work.\n"
            "Open by saying it is a tuyển tập of BBKH and how many parts it has, then "
            "cover the fields and topics the papers span.\n"
            "- Do NOT state a total number of papers — it is not given below.\n"
            "- Do NOT describe a single narrative thread; there is none.\n"
        )
    else:
        kind_clause = ""
    if units:
        numbers = [u["chapter_ordinal"] for u in units if u.get("chapter_ordinal")]
        kind_clause += (
            f"The document has exactly {max(numbers) if numbers else len(units)} chapters — "
            "state this number as given; do not count them yourself.\n"
        )
    # "Do NOT enumerate the sections" produced the abstract a tester rejected
    # as "chưa xuôi theo một bản tổng thuật tài liệu". The tester's worked
    # example states the organisation, then walks through it — quoted below
    # as the form cue.
    prompt = (
        "You are a document analyst.\n\n"
        f"{current_domain_clause(outline)}"
        "TASK: Write section 2.1 (Tóm tắt) of a library digest from the outline "
        "and section summaries below.\n"
        f"{kind_clause}"
        "Form: open by saying how the document is organised — how many parts or "
        "chapters, and for a part the chapters it spans — then walk through them in "
        "reading order, one or two sentences each on what they present (group "
        "neighbouring chapters when there are many), and close with how they fit "
        "together.\n"
        "Example of the form (another document — copy the form, never the content): "
        f'"{SUMMARY_FORM_EXAMPLE}"\n'
        "Length: 10-15 sentences of continuous prose.\n\n"
        "CONSTRAINTS:\n"
        "- Do NOT use bullet points or headings.\n"
        "- Every claim MUST be supported by the summaries below.\n"
        "- Preserve numbers, names, dates, and technical terms verbatim.\n"
        "- A chapter listed without a summary is still part of the document: count it, "
        "describe it from its title, and never say that a chapter or its summary is "
        "missing.\n\n"
        f"{lang_clause}"
        f"Outline with section summaries:\n{outline}\n\n"
        f"{lang_clause}"
        "Abstract:"
    )

    try:
        composed = (
            await llm.chat_completion(prompt, max_tokens=settings.ai_output_reserve_tokens)
        ).strip()
        composed = _ECHOED_HEADING_RE.sub("", composed, count=1).strip()
        composed = await enforce_termbase(llm, composed, current_glossary())
    except Exception as exc:
        logger.warning("Document abstract composition failed (%s) — using node summaries", exc)
        composed = ""
    # Degrade to the raw section summaries rather than back to the placeholder.
    return composed or existing or " ".join(summary for _, summary in entries)


def _levels_with_parents(root: dict) -> tuple:
    """Group tree nodes by height (leaves first) and map child→parent.

    Processing one height level at a time guarantees every node's children
    are already summarised when the node itself runs, while `run_parallel`
    bounds in-flight coroutines (the old recursive gather materialised one
    coroutine per node for the whole tree at once).
    """
    heights: dict = {}
    parents: dict = {}

    def walk(node: dict) -> int:
        children = node.get("children", []) or []
        h = 0
        for child in children:
            parents[id(child)] = node
            h = max(h, walk(child) + 1)
        heights.setdefault(h, []).append(node)
        return h

    walk(root)
    return [heights[h] for h in sorted(heights)], parents


def _pack_leaf_batches(
    nodes: list, enricher, node_budget: int, batch_budget: int, max_nodes: int
) -> list:
    """Pack sibling leaves into batches bounded by node count AND token mass."""
    batches: list = []
    current: list = []
    current_tokens = 0
    for node in nodes:
        content = node.get("content") or node.get("text") or node.get("text_content") or ""
        tokens = min(enricher.count_tokens(content), node_budget)
        if current and (len(current) >= max_nodes or current_tokens + tokens > batch_budget):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(node)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _persist_tree(tree_index_id: str, tree_data: dict) -> None:
    """Checkpoint node summaries back to the tree's storage location.

    MinIO when the tree is offloaded (`tree_data_key`), else the JSON column —
    never the column for an offloaded tree, which would re-bloat it.
    Best-effort: a failed checkpoint must not kill the summarization run
    (the Summary row itself is persisted separately).
    """
    try:
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.db_models import TreeIndex

            ti = db.query(TreeIndex).filter(TreeIndex.id == tree_index_id).first()
            if not ti:
                return
            if getattr(ti, "tree_data_key", None):
                import json as _json

                from services.object_storage import get_object_storage

                get_object_storage().put_bytes(
                    ti.tree_data_key,
                    _json.dumps(tree_data, ensure_ascii=False).encode("utf-8"),
                )
            else:
                ti.tree_data = tree_data
    except Exception as exc:
        logger.warning("Tree summary checkpoint failed (non-fatal): %s", exc)
