"""
Main content extraction service — chapter breakdown for digest §2.2.

Tree-first: walk TreeIndex chapter nodes and LLM-summarize each section.
Fallback: detect markdown headings in OCR text.
"""

import logging
import re
from typing import Dict, List, Optional

from config.settings import pipeline_output_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from core.spatial.zone_classifier import split_chapter_heading
from data.database import get_db_manager
from data.db_models import MainContent
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.doc_kind import BOOK, FRONT_MATTER_CHARS, PROCEEDINGS, resolve_doc_kind_async

logger = logging.getLogger(__name__)


def _collect_chapter_nodes(node: dict) -> tuple[List[dict], dict]:
    """Select the §2.2 units of a document.

    This used to be "every direct child of the tree root", unfiltered and
    uncapped. Tree levels come from a percentile cut, so that count tracks
    document length rather than structure — an 816-page book produced 265
    "chapters" including table captions, a formula and publisher adverts.

    Selection now lives in utils.chapter_units, which anchors on structural
    headings in reading order and reports which tier fired.
    """
    from utils.chapter_units import select_chapter_units

    units, meta = select_chapter_units(node)
    return [{"node": unit, "number": i} for i, unit in enumerate(units, start=1)], meta


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

# Measured on chapter 4 of N4.11.160, 7 verifiable facts, 3 runs per configuration:
#   qwen3.5-35B old prompt 19/21 · gemma-4-26B old prompt 13/21
#   qwen3.5-35B + this block 21/21 · gemma-4-26B + this block 21/21
# "Preserve numbers … verbatim" was already in the constraint list but got buried.
# Naming the failure mode outright — and repeating it close to the point of
# generation — is what actually worked.
NUMERIC_FIDELITY = (
    "NUMERIC FIDELITY (highest priority):\n"
    "- Reproduce EVERY number that appears in the text: bit widths, counts of "
    "signals or registers, cycle counts, model numbers, years, page or section "
    "counts.\n"
    "- Writing 'several control signals' where the text says '29 control signals' "
    "is an ERROR. Writing 'memory ports' where the text says '32-bit and 8-bit "
    "ports' is an ERROR.\n"
    "- Never round, generalise or omit a figure that is stated in the text.\n\n"
)


# How to name the unit being summarised. The §2.2 prompt used to say "chapter"
# for every unit, which is where "Phụ lục B … Chương này phân tích…" came from.
_UNIT_NOUNS_VI = {"chapter": "Chương", "appendix": "Phụ lục", "part": "Phần", "section": "Mục"}
_UNIT_NOUNS_EN = {
    "chapter": "a chapter",
    "appendix": "an appendix",
    "part": "a part",
    "section": "a section",
}


def _sample_chapter_text(llm, node: dict) -> str:
    """Representative text from across a whole chapter, not just its opening.

    `gather_node_text` walks pre-order and stops at its char cap, so on a
    90-page chapter the model only ever saw the first few pages and summarised
    the introduction. `build_stratified_sample` allocates the budget across the
    chapter's sections proportionally to their mass, with a per-section floor.
    """
    from utils.doc_sampling import build_stratified_sample

    enricher = BaseEnricher(llm)
    sampled = build_stratified_sample(
        _with_preferred_summaries(node),
        _gather_node_text(node),
        token_budget=settings.main_content_chapter_sample_tokens,
        count_tokens=enricher.count_tokens,
        truncate=enricher.truncate_to_tokens,
    )
    return (sampled or _gather_node_text(node)).strip()


def _with_preferred_summaries(node: dict) -> dict:
    """Swap in per-section summaries the summarize stage already computed.

    Those summaries are written for every tree node and checkpointed back into
    the tree, but the digest only ever consumed the root's — so on a re-run
    (where they exist) a chapter can be described from condensed section
    summaries instead of raw excerpts. No-op on a first run.
    """
    if not settings.main_content_prefer_node_summaries:
        return node
    children = node.get("children") or node.get("child_nodes") or []
    if not children:
        return node
    swapped = []
    for child in children:
        summary = (child.get("summary") or "").strip()
        content = child.get("content") or ""
        use_summary = summary and len(summary) < len(content)
        swapped.append({**child, "content": summary if use_summary else content})
    return {**node, "children": swapped}


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

    async def submit_async(self, db, document_id: str) -> tuple:
        """Temporal-aware submit — see SummarizationService.submit_async."""
        from config.settings import settings

        if not settings.stage_rerun_use_temporal:
            return self.submit(db, document_id)

        from services.stage_dispatch import submit_stage_with_resource

        return await submit_stage_with_resource(db, document_id, "MAIN_CONTENT", MainContent)

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

    def _front_matter(self, document_id: str) -> str:
        """Opening pages only — where a document declares what it is.

        Never fatal: an unreadable text layer just means detection falls back
        to the title, which is still the strongest single signal.
        """
        try:
            return (self._read_text(document_id) or "")[:FRONT_MATTER_CHARS]
        except Exception:
            return ""

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
        doc_kind: str = BOOK,
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
            # A unit whose title declares `Глава N` / `Приложение X` is structure
            # the author stated, not noise to be judged. The gate exists to drop
            # UNNUMBERED matter (publisher adverts, TOC fragments, copyright
            # pages); letting it overrule an authored chapter number turned
            # «Глава 9. Библиография» into "Các mục phụ trợ" on one run and left
            # it alone on the next.
            numbered = split_chapter_heading(item["node"].get("title"))[0] is not None
            if not numbered and labels.get(item["number"]) in GATE_AUX_LABELS:
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
            llm, to_summarize, task_id, doc_kind=doc_kind
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

        self._progress(task_id, 92, "Translating chapter titles")
        await self._translate_titles(llm, chapters)
        return chapters, degraded_count, raw_count, auxiliary_sections, gate_degraded

    async def _summarize_chapter(
        self, llm, node: dict, number: int, doc_kind: str = BOOK
    ) -> tuple[dict, bool, bool]:
        """Returns (chapter_dict, degraded, raw_passthrough).

        degraded=True: the LLM call failed and the raw-text fallback excerpt
        was used instead (a runtime failure).
        raw_passthrough=True: the LLM was never called because the node's
        (own + descendant) content was too short to summarize meaningfully —
        not a failure, but still means this chapter's digest entry is raw
        source text rather than a real summary, which the caller should be
        able to surface as a quality signal distinct from `degraded`.
        """
        default_title = f"BBKH {number}" if doc_kind == PROCEEDINGS else f"Chương {number}"
        title = (node.get("title") or default_title).strip()
        # "Глава 1. Введение" → the label is rendered in Vietnamese downstream, so
        # carrying the source-language one inside the title printed it twice.
        heading, bare_title = split_chapter_heading(title)
        unit_kind = heading[0] if heading else "chapter"
        unit_noun_vi = _UNIT_NOUNS_VI.get(unit_kind, _UNIT_NOUNS_VI["chapter"])
        unit_noun_en = _UNIT_NOUNS_EN.get(unit_kind, _UNIT_NOUNS_EN["chapter"])
        content = _sample_chapter_text(llm, node)
        lang_clause = pipeline_output_lang_clause()
        degraded = False
        raw_passthrough = False
        paper_count = None

        if len(content.strip()) < 150:
            body = content.strip() or title
            raw_passthrough = True
        elif doc_kind == PROCEEDINGS:
            body, paper_count, degraded = await self._summarize_papers(
                llm, node, title, content, lang_clause
            )
        else:
            prompt = (
                "You are a document analyst.\n\n"
                "TASK: Write a synthesis of this book chapter for a library digest.\n"
                "The text below spans an ENTIRE chapter made up of many sections; excerpts "
                "are sampled from across the whole chapter, so summarise the chapter as a "
                "whole — its subject, approach, and main results.\n"
                "Format: 5-8 sentences (~120-180 words) in Vietnamese.\n"
                "Preserve technical terms; add English/Russian originals in parentheses when helpful.\n\n"
                "CONSTRAINTS:\n"
                "- Do NOT list or enumerate the section headings, and do NOT use bullet "
                "points — write continuous prose about what the chapter covers.\n"
                # "Phụ lục A. Số nhị phân (Двоичные числа). Phụ lục A (Приложение А)
                # trình bày về các số nhị phân…" — 8 of the 12 entries in N4.11.160
                # opened with the exact label just printed. Label and title are
                # rendered separately, as the title-translation prompt already says.
                '- Do NOT open by restating the unit label or title ("Chương 2 trình bày…", '
                '"Phụ lục A (Приложение А) trình bày…"). Both are printed separately '
                "before your text. Start with the substance.\n"
                # The prompt called every unit a "chapter", so both appendix entries
                # of N4.11.160 described themselves as "Chương này" — an official
                # document calling appendix B a chapter. The kind is known here.
                f"- This unit is {unit_noun_en}. When you refer to it, write "
                f'"{unit_noun_vi} này" — never "{_UNIT_NOUNS_VI["chapter"]} này" unless '
                "that is what it is.\n"
                # Appendix C's entry opened "Phụ lục B tập trung vào…", describing
                # itself under another unit's number. A number the model picks is a
                # number it can pick wrong, and the heading above already carries it.
                "- Never identify this unit by its number or letter "
                '("Phụ lục B tập trung vào…", "Chương 5 trình bày…"). The heading is '
                "printed above your text. Refer to a different chapter by number only "
                "when you genuinely mean that other chapter.\n"
                "- Every claim MUST be directly supported by the chapter text below.\n"
                "- Do NOT add external knowledge about the book, author, or subject — rely only "
                "on what this excerpt actually says.\n"
                "- Preserve numbers, names, dates, and technical terms verbatim.\n"
                "- If this excerpt is front matter (title page, author listing, table of "
                "contents) rather than real chapter content, say so briefly instead of "
                "inventing topics, methods, or findings it doesn't contain.\n\n"
                f"{NUMERIC_FIDELITY}"
                f"{lang_clause}"
                f"Chapter title: {title}\n\n"
                f"Chapter text:\n{content}\n\n"
                f"{NUMERIC_FIDELITY}"
                # Repeated immediately before the generation point: with a long
                # non-Vietnamese source block in context, smaller local models
                # (confirmed live: qwen3.5-9b on a Russian-source chapter)
                # otherwise mirror the source language instead of following an
                # instruction stated only once, further back in the prompt.
                f"{lang_clause}"
                "Summary (in Vietnamese):"
            )
            try:
                body = (
                    await llm.chat_completion(
                        prompt, max_tokens=settings.main_content_chapter_max_tokens
                    )
                ).strip()
            except Exception:
                body = content[:500] + ("..." if len(content) > 500 else "")
                degraded = True

        # Bilingual titles already in "Vi (Original)" form keep the old split;
        # otherwise the original is the title minus its structural label, and the
        # Vietnamese side is filled in later by `_translate_titles`.
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", bare_title)
        if m:
            title_vi = m.group(1).strip()
            title_original = m.group(2).strip()
        else:
            title_original = bare_title
            title_vi = bare_title

        return (
            {
                "number": number,
                "title_vi": title_vi,
                "title_original": title_original,
                "content": body,
                **({"heading_kind": heading[0], "heading_ordinal": heading[2]} if heading else {}),
                # Only set on a kỷ yếu, and only when the model could actually
                # count. Absent → the entry renders as a single BBKH, which is
                # the honest fallback: an invented "gồm 47 BBKH" would be worse
                # than no count at all.
                **({"paper_count": paper_count} if paper_count else {}),
            },
            degraded,
            raw_passthrough,
        )

    async def _translate_titles(self, llm, chapters: List[dict]) -> None:
        """Fill each entry's `title_vi` from one batched call. Mutates in place.

        The mẫu's §2.2 line is `Chương 1. Giới thiệu (Введение).` — the Vietnamese
        name is required, and on a Russian source it has to be translated.

        Asking for it inside the summarising call looked cheaper but measured
        worse: on N4.11.160 three of twelve chapters came back as plain prose,
        losing the title and leaving `Chương 3. Цифровой логический уровень.` in
        an official document. Translating a dozen short titles is one small task
        the model does reliably, and it keeps the summary call free of any
        parsing at all.

        Never fatal: on any failure every entry keeps the title it already has.
        """
        pending = [c for c in chapters if (c.get("title_original") or "").strip()]
        if not pending:
            return

        listing = "\n".join(f"{c['number']}. {c['title_original']}" for c in pending)
        prompt = (
            "You are a translator working on a library catalogue entry.\n\n"
            "TASK: Translate each chapter title below into Vietnamese.\n\n"
            "RULES:\n"
            "- Translate the title ONLY. Do NOT add a label such as "
            '"Chương 1", "Phụ lục B" or "Part II" — those are added separately.\n'
            "- Keep technical terms and proper nouns as they are conventionally "
            "written in Vietnamese technical literature.\n"
            "- Keep the numbering `n` exactly as given.\n\n"
            f"TITLES:\n{listing}\n\n"
            'OUTPUT: JSON array only — [{"n": 1, "title_vi": "..."}, ...]\n'
            f"{pipeline_output_lang_clause(json_values=True)}"
            "JSON:"
        )

        try:
            raw = await llm.chat_completion(prompt, max_tokens=1000)
        except Exception as exc:
            logger.warning("Chapter title translation failed (%s) — keeping the original", exc)
            return

        rows, parse_failed = self._parse_json_list(llm, raw, list_key="titles")
        if parse_failed or not rows:
            logger.warning("Title translation response was not JSON — keeping the original")
            return

        by_number = {c["number"]: c for c in pending}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                number = int(row.get("n"))
            except (TypeError, ValueError):
                continue
            target = by_number.get(number)
            translated = str(row.get("title_vi") or "").strip()
            if target is None or not translated:
                continue
            # The model prepends a label anyway often enough to matter — observed
            # "Phụ lục B. Số thực…", which renders as "Phụ lục B. Phụ lục B. …".
            target["title_vi"] = split_chapter_heading(translated)[1]

    async def _summarize_papers(
        self, llm, node: dict, title: str, content: str, lang_clause: str
    ) -> tuple[str, Optional[int], bool]:
        """§2.2 for a kỷ yếu / journal issue: a run of independent papers.

        Two things differ from the book path. The unit is not "a chapter of one
        work" — summarising it as such produced a false narrative thread. And
        the entry may cover several BBKH, in which case the official form states
        how many; the count comes from the unit's *complete* heading list rather
        than the sampled excerpt, because a sample cannot be counted.
        """
        member_titles = [
            str(t).strip() for t in (node.get("member_titles") or []) if str(t).strip()
        ]
        titles_block = (
            "HEADINGS INSIDE THIS SECTION (complete list, reading order):\n"
            + "\n".join(f"- {t}" for t in member_titles[:200])
            + "\n\n"
            if member_titles
            else ""
        )

        prompt = (
            "You are a document analyst working on conference proceedings.\n\n"
            "TASK: Write the digest entry for one section of the proceedings.\n"
            "This section contains one or more independent scientific papers (BBKH) by "
            "different authors — it is NOT a chapter of a single continuous work, so do "
            "not describe a narrative thread running through it.\n\n"
            f"{titles_block}"
            "ALSO: count how many DISTINCT PAPERS this section contains.\n"
            "- The headings above mix paper titles with headings *inside* a paper "
            "(Introduction, Method, Results, Conclusion, References). Count only the "
            "paper titles.\n"
            "- If you cannot tell, return null. Do NOT guess a number.\n\n"
            "CONSTRAINTS:\n"
            "- 5-8 sentences (~120-180 words) in Vietnamese, continuous prose, no bullets.\n"
            "- Say what the papers are about collectively: topics, methods, results.\n"
            "- Every claim MUST be supported by the text below. Do NOT add outside "
            "knowledge about the conference, its authors, or the field.\n"
            "- Preserve numbers, names, dates and technical terms verbatim.\n\n"
            f"{NUMERIC_FIDELITY}"
            f"{lang_clause}"
            f"Section title: {title}\n\n"
            f"Section text:\n{content}\n\n"
            f"{NUMERIC_FIDELITY}"
            f"{lang_clause}"
            'OUTPUT: JSON only — {"summary": "<tiếng Việt>", "paper_count": <số hoặc null>}\n'
            "JSON:"
        )

        try:
            raw = await llm.chat_completion(
                prompt, max_tokens=settings.main_content_chapter_max_tokens
            )
        except Exception:
            return content[:500] + ("..." if len(content) > 500 else ""), None, True

        parsed = self._extract_json_object(raw)
        if parsed is None:
            # Prose instead of JSON is still a usable summary — the count is
            # what gets lost, and losing it only costs the "gồm N BBKH" clause.
            return str(raw).strip(), None, False

        summary = str(parsed.get("summary") or "").strip() or str(raw).strip()
        try:
            count = int(parsed.get("paper_count"))
        except (TypeError, ValueError):
            count = 0
        return summary, (count if count >= 2 else None), False

    @staticmethod
    def _extract_json_object(raw) -> Optional[dict]:
        import json
        import re

        match = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _summarize_chapters(
        self,
        llm,
        nodes: List[dict],
        task_id: Optional[str],
        summarize=None,
        doc_kind: str = BOOK,
    ) -> tuple[List[dict], int, int]:
        """Summarise chapters with bounded concurrency, preserving document
        order. Returns (chapters, degraded_count, raw_passthrough_count).
        This is the pipeline's longest stage — one-at-a-time chapter calls
        left the LLM idle between chapters on large books."""
        from services.translators._parallel import run_parallel

        summarize = summarize or self._summarize_chapter
        total = len(nodes)

        async def worker(_idx: int, item: dict):
            return await summarize(llm, item["node"], item["number"], doc_kind)

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
            doc_title = ""
            doc_kind_override = None
            with db_manager.session() as db:
                from data.db_models import Document, TreeIndex
                from utils.tree_payload import get_tree_payload

                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc is not None:
                    doc_title = doc.title or ""
                    doc_kind_override = doc.digest_doc_kind

                tree_index = (
                    db.query(TreeIndex)
                    .filter(TreeIndex.document_id == document_id)
                    .order_by(TreeIndex.created_at.desc())
                    .first()
                )
                if tree_index:
                    tree_data = get_tree_payload(db, tree_index)

            # Book or kỷ yếu decides both the §2.2 prompt and the line form the
            # renderer emits, so it has to be settled before any summarising.
            kind_meta = await resolve_doc_kind_async(
                llm,
                doc_kind_override,
                title=doc_title,
                text=self._front_matter(document_id),
            )
            doc_kind = kind_meta["doc_kind"]

            chapters: List[dict] = []
            degraded_chapters = 0
            raw_passthrough_chapters = 0
            auxiliary_sections = 0
            gate_degraded = False

            selection_meta: dict = {}
            if tree_data:
                self._progress(task_id, 15, "Walking tree for chapters")
                nodes, selection_meta = _collect_chapter_nodes(tree_data)
                (
                    chapters,
                    degraded_chapters,
                    raw_passthrough_chapters,
                    auxiliary_sections,
                    gate_degraded,
                ) = await self._summarize_with_gate(llm, nodes, task_id, doc_kind=doc_kind)

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
                # Which mode §2.2 was written in, and on whose authority — the
                # digest renderer must use the same one the prompt did.
                **kind_meta,
                "degraded_chapters": degraded_chapters,
                "raw_passthrough_chapters": raw_passthrough_chapters,
                "auxiliary_sections": auxiliary_sections,
                "gate_degraded": gate_degraded,
                # How the §2.2 units were chosen — read by the quality report so a
                # fragmented or machine-cut digest cannot ship unnoticed.
                **selection_meta,
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
