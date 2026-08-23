"""
Keyword extraction service.

Hybrid approach — outline *and* body, never one alone:
1. Node titles from the TreeIndex give structure-aware candidates.
2. YAKE over the full document text supplies terms the headings never name.
3. An LLM map pass over chunks of the full text supplies terms no surface
   statistic ranks highly.
4. The LLM re-ranks the merged pool against a stratified excerpt.

Step 2 used to run only when the tree produced fewer than 10 candidates, so
any document with a real outline had its keywords chosen from headings alone.

Step 3 exists because step 4 only ever sees ~8k tokens. On an 816-page book that
is about 2% of the text, so anything the candidate list does not carry is
unreachable no matter how good the re-ranker is.
"""

import json
import logging
import re
from inspect import isawaitable
from typing import Dict, List, Optional

from config.settings import pipeline_keyword_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import DocumentKeyword, Keyword, KeywordExtraction
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.structural_titles import is_structural_title

logger = logging.getLogger(__name__)

# Weight assigned to each candidate source tier
_WEIGHT_TITLE = 1.0

# Headings are plentiful and cheap; leave room for content-derived terms.
_MAX_TREE_CANDIDATES = 40
_MAX_STATISTICAL_CANDIDATES = 50
_MAX_LLM_CANDIDATES = 60
_MAX_CANDIDATES = 120

# YAKE parameters. `n=3` matches the old n-gram range. `dedup_lim` only collapses
# near-identical strings — measured, it does NOT remove overlapping phrase windows
# like "algebra describes digital", so do not rely on it for that. Those windows
# now merely rank low instead of tying with real terms, and the re-rank drops them.
_YAKE_MAX_NGRAM = 3
_YAKE_DEDUP_LIMIT = 0.9

# What is stored is a ranked pool, not the final §2.3 list. The digest's assembly
# filters — title/authors, chapter headings — cannot run any earlier: the
# bibliographic stage runs in parallel with this one and main_content runs after
# it, so neither input exists here. Storing exactly what §2.3 prints therefore
# turned every rejected keyword into a lost slot (7 of 20 on N4.11.160).
#
# 1.5 is set from that measured 35% loss with room to spare. The pool is a
# ceiling and never a floor: a document with nine real subjects still yields nine.
_KEYWORD_RESERVE_RATIO = 1.5


def keyword_pool_size(target: int) -> int:
    """How many keywords to ask for and store so §2.3 can still fill `target`."""
    if target <= 0:
        return 0
    return max(target + 1, round(target * _KEYWORD_RESERVE_RATIO))


# Cost ceiling for the LLM map pass. An 816-page book at `ai_chunk_tokens` is
# well past this, so the cap does bite in practice — and is logged when it does,
# because silently covering less than the whole document reads as covering all
# of it.
_MAX_MAP_CHUNKS = 24
_KEYWORDS_PER_CHUNK = 8

# Character class for the whitespace tokenizer: basic Latin, Latin Extended
# (including the Vietnamese block), Greek and Cyrillic. The old pattern was Latin
# only, so for a Russian book the extracted vocabulary was just the scattered
# Latin part numbers (Core i7, FPGA, ASCII) — the content tier was effectively
# empty and chapter headings took every slot in §2.3.
_WORD_CHAR = r"a-zA-ZÀ-ɏͰ-ϿЀ-ԯḀ-ỿ"
_TOKEN_PATTERN = rf"(?u)\b[{_WORD_CHAR}][{_WORD_CHAR}0-9\-]{{1,}}\b"

# Han, kana, Hangul — written without spaces, so a whitespace tokenizer yields
# exactly one "word" as long as the sentence. Without a segmenter (jieba, MeCab)
# character n-grams are the workable option: the spans they produce still appear
# verbatim in the document, so the grounding rule holds, and the LLM filters
# again afterwards.
_CJK_CHAR_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_CJK_HEAVY_RATIO = 0.2

# Which script the text is mostly written in, for picking a stopword list. The
# `vi` pattern is the letters no other language in this corpus (en/ru/zh) uses:
# the seven Vietnamese-only Latin letters plus the whole Latin Extended
# Additional block, where its tone-marked vowels live.
_LETTER_RE = re.compile(r"[^\W\d_]")
_SCRIPT_RES = {
    "ru": re.compile(r"[Ѐ-ԯ]"),
    "el": re.compile(r"[Ͱ-Ͽ]"),
    "vi": re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯḀ-ỿ]"),
}

# A share of all letters, not a count: the loser here is Cyrillic at 0.4% of a
# Vietnamese summary (quoted source terms), and the winners are 17% Vietnamese
# in that same summary and 96.5% Cyrillic in the Russian source. Anything in the
# low single digits separates them; 5% keeps a margin on both sides.
_SCRIPT_DOMINANCE = 0.05

# YAKE ships 35 stopword lists and no Vietnamese one, so these are passed in by
# hand. Closed classes only — pronouns, determiners, prepositions, conjunctions,
# aspect markers, copulas. Deliberately absent: "số", "phần", "bộ", "hệ", "mức",
# "lớp", which are function-like in isolation but are exactly the head nouns of
# the technical terms this corpus is about ("bộ nhớ", "số nguyên", "lớp logic").
_VIETNAMESE_STOPWORDS = frozenset(
    """
    và hoặc hay nhưng mà còn thì nên vì do bởi nếu khi lúc trong ngoài trên dưới
    giữa cùng với của cho từ đến tới tại về theo bằng qua sau trước
    là gồm có không chưa được bị phải cần nên sẽ đã đang vẫn cũng chỉ rất khá hơn
    nhất này đó kia ấy các những mọi mỗi một cái sự việc điều
    tôi ta chúng nó họ ai gì nào đâu sao
    ra vào lên xuống nữa rồi luôn thường hầu như tuy dù nhờ
    """.split()
)


class KeywordService(BaseTaskService):
    """Keyword extraction (background task)."""

    def submit(self, db, document_id: str, max_keywords: int = 20) -> tuple:
        """Create a KeywordExtraction(PENDING) record and submit background task.

        Returns (task_id, extraction_id, reused).
        """
        from data.repositories import DocumentRepository

        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        existing_task = task_manager.get_active_task_id(db, document_id, "KEYWORDS")
        if existing_task:
            extraction = (
                db.query(KeywordExtraction)
                .filter(KeywordExtraction.document_id == document_id)
                .order_by(KeywordExtraction.created_at.desc())
                .first()
            )
            return existing_task, (extraction.id if extraction else None), True

        extraction = KeywordExtraction(
            document_id=document_id,
            status="PENDING",
            max_keywords=max_keywords,
        )
        db.add(extraction)
        db.commit()
        db.refresh(extraction)
        extraction_id = extraction.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="KEYWORDS",
            coro_factory=lambda tid: self._extract(document_id, max_keywords, extraction_id, tid),
        )
        return task_id, extraction_id, False

    async def submit_async(self, db, document_id: str, max_keywords: int = 20) -> tuple:
        """Temporal-aware submit — see SummarizationService.submit_async."""
        from config.settings import settings

        if not settings.stage_rerun_use_temporal:
            return self.submit(db, document_id, max_keywords)

        from services.stage_dispatch import submit_stage_with_resource

        return await submit_stage_with_resource(
            db,
            document_id,
            "KEYWORDS",
            KeywordExtraction,
            max_keywords=max_keywords,
        )

    async def run_for_pipeline(
        self,
        document_id: str,
        task_id: Optional[str] = None,
        max_keywords: int = 20,
    ):
        db_manager = get_db_manager()
        with db_manager.session() as db:
            e = (
                db.query(KeywordExtraction)
                .filter(KeywordExtraction.document_id == document_id)
                .order_by(KeywordExtraction.created_at.desc())
                .first()
            )
            if not e:
                e = KeywordExtraction(
                    document_id=document_id,
                    status="PENDING",
                    max_keywords=max_keywords,
                )
                db.add(e)
                db.commit()
                db.refresh(e)
            extraction_id = e.id
        return await self._do_extract(document_id, max_keywords, extraction_id, task_id)

    # ── Tree candidate extraction ─────────────────────────────────────

    def _tree_candidates(self, tree_data: dict) -> List[Dict]:
        """
        Walk tree_data recursively and collect candidate keywords from node titles.

        Handles mixed key shapes:
          children  : 'children' or 'child_nodes'
          title     : 'title' or 'name'

        Summaries and body text are deliberately NOT candidates: a whole
        paragraph is not a keyword, and emitting each as one candidate crowded
        out every real term once the list was truncated to 50. Content reaches
        the model through the content tiers and the excerpt instead.

        Returns de-duplicated list of {"keyword": str, "weight": float}.
        """
        seen: dict[str, float] = {}  # keyword → max weight seen

        def _walk(node: dict):
            title = (node.get("title") or node.get("name") or "").strip()
            label = node.get("label") or node.get("node_type")
            body = (
                node.get("content")
                or node.get("text")
                or node.get("text_content")
                or node.get("text_full")
                or ""
            )
            if title and is_structural_title(title, label=label, body=body):
                seen[title] = max(seen.get(title, 0.0), _WEIGHT_TITLE)

            children = node.get("children") or node.get("child_nodes") or []
            for child in children:
                _walk(child)

        _walk(tree_data)

        return [
            {"keyword": kw, "weight": w}
            for kw, w in sorted(seen.items(), key=lambda x: x[1], reverse=True)
            if kw
        ]

    # ── Content candidate extraction ─────────────────────────────────

    @staticmethod
    def _is_cjk_heavy(text: str) -> bool:
        """Enough Han/kana/Hangul that whitespace tokenization is meaningless."""
        dense = "".join(text.split())
        if not dense:
            return False
        return len(_CJK_CHAR_RE.findall(dense)) / len(dense) >= _CJK_HEAVY_RATIO

    def _content_candidates(self, text: str, max_candidates: int = 50) -> List[Dict]:
        """Candidate terms from the document body, ranked.

        Returns list of {"keyword": str, "score": float}, best first, where a
        HIGHER score is better — the merged pool downstream is ordered that way.

        The ranker is chosen by script. YAKE handles anything space-separated:
        it is built for single documents (no corpus needed, which is exactly the
        situation here), and scores on position, casing, dispersion and
        co-occurrence rather than raw count. CJK has no spaces for it to work
        with, so character n-grams stand in there.

        This does not remove overlapping phrase windows — YAKE's `dedup_lim` was
        measured and does not do that. What changes is that they no longer tie
        with real terms for the top slots.

        What ran before was named TF-IDF but was not: `corpus = [text, ""]` puts
        every term in exactly one of two documents, so IDF is constant and
        cancels — measured as one distinct IDF value across the whole
        vocabulary. It ranked by frequency, which is why sliding-window junk
        ("algebra describes digital") kept taking slots.

        Nothing extractable returns an empty list. Keywords is a non-critical
        stage; failing all of it because a document has no usable vocabulary is
        too blunt.
        """
        body = (text or "").strip()
        if not body:
            return []
        if self._is_cjk_heavy(body):
            return self._char_ngram_candidates(body, max_candidates)
        return self._yake_candidates(body, max_candidates)

    @staticmethod
    def _yake_language(text: str) -> str:
        """YAKE stopword list to use, chosen by the script most of the text is in.

        This used to ask whether the text contained *any* Cyrillic, so a single
        quoted Russian term made a whole document Russian. That is not a rare
        shape here — every §2.2 summary this pipeline writes cites its source
        terms in parentheses. Measured on that Vietnamese prose: Cyrillic is 0.4%
        of letters and Vietnamese diacritics 17.0%, against 96.5% Cyrillic in the
        Russian source itself, so dominance separates them with room to spare.

        Latin script splits into `vi` and `en` by diacritics no other language in
        this corpus uses. YAKE ships no Vietnamese pack; `_yake_stopwords`
        supplies one.
        """
        letters = len(_LETTER_RE.findall(text))
        if not letters:
            return "en"
        counts = {code: len(rx.findall(text)) for code, rx in _SCRIPT_RES.items()}
        best = max(counts, key=lambda code: counts[code])
        if counts[best] / letters >= _SCRIPT_DOMINANCE:
            return best
        return "en"

    @staticmethod
    def _yake_stopwords(language: str) -> Optional[set]:
        """Vietnamese has no YAKE language pack, so its function words are passed in.

        Without them they are not merely present but dominant: on real Vietnamese
        prose `các` ranked first of fifty candidates and 11 of 50 candidates were
        at least half function words. `None` keeps YAKE's own list for every
        language that has one.
        """
        return set(_VIETNAMESE_STOPWORDS) if language == "vi" else None

    def _yake_candidates(self, body: str, max_candidates: int) -> List[Dict]:
        import yake

        language = self._yake_language(body)
        try:
            extractor = yake.KeywordExtractor(
                lan=language,
                n=_YAKE_MAX_NGRAM,
                dedup_lim=_YAKE_DEDUP_LIMIT,
                top=max_candidates,
                stopwords=self._yake_stopwords(language),
            )
            scored = extractor.extract_keywords(body)
        except Exception as exc:
            logger.warning("YAKE keyword extraction failed (%s) — no content candidates", exc)
            return []

        out: List[Dict] = []
        seen: set[str] = set()
        for kw, score in scored:
            term = str(kw).strip()
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            # YAKE scores lower-is-better; invert so the pool stays
            # higher-is-better and the prompt's weights read consistently.
            out.append({"keyword": term, "score": round(1.0 / (1.0 + float(score)), 4)})
        return out

    def _char_ngram_candidates(self, body: str, max_candidates: int) -> List[Dict]:
        """CJK fallback: frequency-ranked character n-grams.

        Not a segmenter — `计算机组` is not a word — but every span appears
        verbatim in the document, so the grounding rule holds and the re-rank
        discards the ones that are not terms.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform([body, ""])
        except ValueError:
            return []

        scored = sorted(
            zip(vectorizer.get_feature_names_out(), matrix[0].toarray()[0]),
            key=lambda x: x[1],
            reverse=True,
        )

        out: List[Dict] = []
        seen: set[str] = set()
        for kw, score in scored:
            if score <= 0:
                continue
            # char_wb pads each word with spaces, so an extracted n-gram can
            # carry whitespace on either end or be blank entirely.
            term = str(kw).strip()
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            out.append({"keyword": term, "score": round(float(score), 4)})
            if len(out) >= max_candidates:
                break
        return out

    # ── LLM map pass over the full text ──────────────────────────────

    async def _llm_candidates(
        self,
        llm,
        text: str,
        chunk_tokens: int,
        task_id: Optional[str] = None,
    ) -> List[Dict]:
        """Ask the model for terms chunk by chunk across the WHOLE document.

        The final re-rank grounds against an excerpt of at most ~8k tokens — on
        an 816-page book roughly 2% of the text. Terms from the other 98% can
        only arrive through the candidate list, and no statistical ranker
        recovers a term it has no surface signal for.

        Runs on the llama-server already serving the pipeline; no model is
        loaded onto the GPU. One failing chunk costs that chunk only: keywords
        is non-critical and losing the other chunks with it would be worse.
        """
        body = (text or "").strip()
        if not body:
            return []

        from services.translators._parallel import run_parallel

        enricher = BaseEnricher(llm)
        chunks = enricher.chunk_text(body, max_tokens=chunk_tokens)
        total = len(chunks)
        sampled = total
        if total > _MAX_MAP_CHUNKS:
            step = total / _MAX_MAP_CHUNKS
            chunks = [chunks[min(total - 1, int(i * step))] for i in range(_MAX_MAP_CHUNKS)]
            sampled = _MAX_MAP_CHUNKS
            logger.warning(
                "Keyword map pass: %d chunks exceeds the cap of %d — sampling every "
                "~%.1f chunks across the document (%.0f%% coverage)",
                total,
                _MAX_MAP_CHUNKS,
                step,
                100.0 * _MAX_MAP_CHUNKS / total,
            )
        self._last_map_diagnostics = {
            "total_chunks": total,
            "sampled_chunks": sampled,
            "chunk_successes": 0,
            "chunk_failures": 0,
        }
        if not chunks:
            return []

        lang_clause = pipeline_keyword_lang_clause()

        async def _one(_idx: int, chunk: str) -> List[str]:
            prompt = (
                "You are a subject indexer for a technical library.\n\n"
                f"TASK: List up to {_KEYWORDS_PER_CHUNK} subject terms this section "
                "would be indexed under.\n\n"
                "RULES:\n"
                "- Every term MUST appear verbatim as a contiguous phrase in the section.\n"
                "- Prefer specific technical terms and proper nouns over generic words.\n"
                "- Do NOT return section headings — they name a part, not a subject.\n"
                "- Return fewer terms rather than padding with weak ones.\n\n"
                f"{lang_clause}\n\n"
                'OUTPUT: a JSON array of strings only, e.g. ["term one", "term two"]\n\n'
                f"SECTION:\n{chunk}\n\nJSON:"
            )
            try:
                raw = await llm.chat_completion(prompt)
                parsed, parse_failed = self._parse_json_list(llm, raw)
            except Exception as exc:
                logger.warning("Keyword map chunk %d failed (%s) — skipping it", _idx, exc)
                self._last_map_diagnostics["chunk_failures"] += 1
                return []
            if parse_failed or not parsed:
                self._last_map_diagnostics["chunk_failures"] += 1
                return []
            self._last_map_diagnostics["chunk_successes"] += 1
            return [str(t).strip() for t in parsed if isinstance(t, (str, int, float))]

        async def _on_progress(pct: int, msg: str) -> None:
            self._progress(task_id, 30 + int(pct * 0.10), msg)

        per_chunk = await run_parallel(
            chunks,
            _one,
            parallelism=settings.ai_max_concurrent_requests,
            on_progress=_on_progress,
            progress_label="Đoạn ánh xạ từ khóa",
        )

        out: List[Dict] = []
        seen: set[str] = set()
        for terms in per_chunk:
            for term in terms or []:
                key = term.casefold()
                if not term or key in seen:
                    continue
                seen.add(key)
                out.append({"keyword": term, "score": 1.0})
        return out

    @staticmethod
    def _rerank_instructions(pool_size: int) -> str:
        return (
            "You are a keyword extraction expert for academic and technical documents.\n\n"
            f"TASK: Select the {pool_size} most relevant academic/technical keywords "
            "from the candidates above. Re-rank with an importance weight from 0.0 to 1.0.\n\n"
            "GROUNDING RULES:\n"
            "- Every selected keyword MUST appear verbatim as a contiguous phrase in the document text.\n"
            "- Do NOT generate synonyms, hypernyms, or related terms not in the source.\n"
            "- If a candidate is a paraphrase or abstraction, REJECT it.\n"
            "- Prefer noun phrases and proper nouns over generic terms.\n"
            "- Do NOT return the document's own title or its authors' names: those "
            "belong to the bibliographic record, not the subject index.\n"
            "- Do NOT return chapter or section headings as keywords — a table of "
            "contents entry names a part of the book, not a subject someone would "
            "search for.\n\n"
            "WEIGHT CALIBRATION:\n"
            "- 0.9-1.0: Core topics, appears in title or abstract, discussed in multiple sections\n"
            "- 0.7-0.89: Important concept, discussed in one section or multiple mentions\n"
            "- 0.5-0.69: Mentioned term, relevant but not central\n"
            "- <0.5: do not include\n\n"
            f"{pipeline_keyword_lang_clause()}"
            "DISPLAY FORMAT (for digest template §2.3):\n"
            "- Also return a `display` string: Vietnamese term (Original term) for non-Vietnamese docs.\n"
            '- Example: {"keyword": "Adaptive radar", "display": "Radar thích ứng (Adaptive radar)", "weight": 0.9}\n'
            "- For Vietnamese source docs, display may equal keyword.\n\n"
            "Return ONLY valid JSON as a list:\n"
            '[{"keyword": "example term", "display": "Việt (example term)", "weight": 0.95}, ...]\n\nJSON:'
        )

    def _build_rerank_prompt(
        self,
        *,
        candidate_lines: str,
        excerpt: str,
        pool_size: int,
    ) -> str:
        context_block = f"DOCUMENT EXCERPT:\n{excerpt}"
        return (
            f"CANDIDATES (from document structure):\n{candidate_lines}\n\n"
            f"{context_block}\n\n"
            f"{self._rerank_instructions(pool_size)}"
        )

    async def _chat_keyword_rerank(self, llm, prompt: str) -> str:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "display": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["keyword", "weight"],
            },
        }
        try:
            return await llm.chat_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "keywords", "schema": schema},
                },
            )
        except Exception as exc:
            logger.warning(
                "Provider rejected keyword response schema (%s) — retrying unconstrained",
                exc,
            )
            return await llm.chat_completion(prompt)

    @staticmethod
    async def _parse_rerank_response(llm, response: str) -> tuple[list, bool]:
        """Parse the strictly list-rooted reranker reply.

        Production clients expose a synchronous ``extract_json`` method. The
        awaitable branch keeps the service compatible with async test doubles
        without weakening the list-root contract.
        """
        try:
            parsed = llm.extract_json(response, expected_root="list")
            if isawaitable(parsed):
                parsed = await parsed
            # Some older async test doubles have an unconfigured async
            # ``extract_json``. A direct JSON decode remains root-strict and
            # does not reintroduce the nested-object recovery bug.
            if not isinstance(parsed, list):
                parsed = json.loads(response)
        except Exception as exc:
            logger.warning("Keyword refinement JSON parse failed: %s", exc)
            return [], True
        if isinstance(parsed, list):
            return parsed, False
        return [], True

    async def _refine_keywords(
        self,
        llm,
        *,
        candidates: List[Dict],
        candidate_lines: str,
        document_id: str,
        text: str,
        pool_size: int,
        task_id: Optional[str],
    ) -> tuple[list[dict], dict]:
        from utils.keyword_validation import validate_keyword_batch
        from utils.prompt_budget import PromptBudget, allocate_document_sample, build_pipeline_sample

        enricher = BaseEnricher(llm)
        budget = PromptBudget(
            context_tokens=settings.ai_model_context_window,
            output_reserve=settings.ai_output_reserve_tokens,
        )
        instructions = self._rerank_instructions(pool_size)
        excerpt, budget_meta = allocate_document_sample(
            document_id=document_id,
            text=text,
            enricher=enricher,
            budget=budget,
            fixed_parts=[f"CANDIDATES (from document structure):\n{candidate_lines}", instructions],
            sample_builder=lambda sample_budget: build_pipeline_sample(
                document_id, text, enricher, sample_budget
            ),
        )
        prompt = self._build_rerank_prompt(
            candidate_lines=candidate_lines,
            excerpt=excerpt,
            pool_size=pool_size,
        )

        response = await self._chat_keyword_rerank(llm, prompt)
        self._progress(task_id, 70, "Đang xử lý kết quả tinh lọc từ khóa")
        keywords_list, parse_failed = await self._parse_rerank_response(llm, response)

        if parse_failed:
            compact_lines = "\n".join(
                f"  {i+1}. {c['keyword']} (weight={c.get('weight', c.get('score', 1.0)):.2f})"
                for i, c in enumerate(candidates[:30])
            )
            logger.warning(
                "Keyword refinement parse failed for %s — one compact retry (budget=%s)",
                document_id,
                budget_meta,
            )
            compact_excerpt, _ = allocate_document_sample(
                document_id=document_id,
                text=text,
                enricher=enricher,
                budget=budget,
                fixed_parts=[f"CANDIDATES:\n{compact_lines}", instructions],
                sample_builder=lambda sample_budget: build_pipeline_sample(
                    document_id, text, enricher, min(sample_budget, 4000)
                ),
            )
            retry_prompt = self._build_rerank_prompt(
                candidate_lines=compact_lines,
                excerpt=compact_excerpt,
                pool_size=pool_size,
            )
            response = await self._chat_keyword_rerank(llm, retry_prompt)
            keywords_list, parse_failed = await self._parse_rerank_response(llm, response)

        if parse_failed:
            raise ValueError("Keyword refinement JSON parse failed after retry")

        validated, diagnostics = validate_keyword_batch(
            keywords_list,
            source_text=text,
            pool_size=pool_size,
        )
        diagnostics["prompt_budget"] = budget_meta
        diagnostics["parse_failed"] = False
        if len(validated) < diagnostics["min_required"]:
            raise ValueError(
                f"Too few valid keywords after validation: {len(validated)}/"
                f"{diagnostics['min_required']} required"
            )
        return validated, diagnostics

    # ── Main extraction coroutine ─────────────────────────────────────

    async def _extract(
        self,
        document_id: str,
        max_keywords: int,
        extraction_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()

        def _update_extraction(**fields):
            if not extraction_id:
                return
            with db_manager.session() as db:
                e = (
                    db.query(KeywordExtraction)
                    .filter(KeywordExtraction.id == extraction_id)
                    .first()
                )
                if e:
                    for k, v in fields.items():
                        setattr(e, k, v)

        _update_extraction(status="IN_PROGRESS")

        try:
            await self._do_extract(document_id, max_keywords, extraction_id, task_id)
        except Exception as exc:
            _update_extraction(status="FAILED", error=str(exc))
            raise

    async def _do_extract(
        self,
        document_id: str,
        max_keywords: int,
        extraction_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()
        # `max_keywords` is what §2.3 prints; this is what gets asked for and
        # stored, so the assembly filters have something to refill from.
        pool_size = keyword_pool_size(max_keywords)

        from api.dependencies import get_llm_client

        llm = get_llm_client()

        # ── Phase A: load candidates ───────────────────────────────
        self._progress(task_id, 10, "Đang tải ứng viên từ khóa")

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

        use_tree = bool(tree_data)
        candidates: List[Dict] = []

        if use_tree:
            candidates = self._tree_candidates(tree_data)[:_MAX_TREE_CANDIDATES]
            self._progress(task_id, 25, f"Cây mục lục: {len(candidates)} tiêu đề ứng viên")

        # Always supplement from the body. Headings name sections, not concepts:
        # a term discussed throughout the book but never used in a heading was
        # previously unreachable whenever a tree existed.
        text = self._read_text(document_id)
        existing_kws = {c["keyword"].lower() for c in candidates}

        def _absorb(found: List[Dict]) -> None:
            for c in found:
                key = c["keyword"].lower()
                if key in existing_kws:
                    continue
                candidates.append({"keyword": c["keyword"], "weight": min(c["score"], 1.0)})
                existing_kws.add(key)

        self._progress(task_id, 28, "Đang xếp hạng thuật ngữ trên toàn văn bản")
        _absorb(self._content_candidates(text, max_candidates=_MAX_STATISTICAL_CANDIDATES))

        # The re-rank below only sees ~8k tokens of excerpt. This pass is how
        # terms from the rest of the document reach the pool at all.
        llm_found = await self._llm_candidates(
            llm, text, chunk_tokens=settings.ai_chunk_tokens, task_id=task_id
        )
        _absorb(llm_found[:_MAX_LLM_CANDIDATES])

        candidates = candidates[:_MAX_CANDIDATES]

        self._progress(task_id, 40, "Đang tinh lọc ứng viên từ khóa bằng LLM")

        # ── Phase B: LLM reranking ────────────────────────────────────
        candidate_lines = "\n".join(
            f"  {i+1}. {c['keyword']} (weight={c.get('weight', c.get('score', 1.0)):.2f})"
            for i, c in enumerate(candidates)
        )

        validated, diagnostics = await self._refine_keywords(
            llm,
            candidates=candidates,
            candidate_lines=candidate_lines,
            document_id=document_id,
            text=text,
            pool_size=pool_size,
            task_id=task_id,
        )
        diagnostics["map"] = getattr(self, "_last_map_diagnostics", {})
        diagnostics["tree_candidates"] = len(candidates)

        # ── Phase C: persist (atomic — validated set first, then replace) ──
        with db_manager.session() as db:
            db.query(DocumentKeyword).filter(DocumentKeyword.document_id == document_id).delete()

            stored = []
            for item in validated:
                kw_name = item["keyword"]
                weight = float(item["weight"])
                display = (item.get("display") or "").strip()

                kw = db.query(Keyword).filter(Keyword.keyword_name == kw_name).first()
                if not kw:
                    kw = Keyword(keyword_name=kw_name)
                    db.add(kw)
                    db.flush()

                assoc = DocumentKeyword(
                    document_id=document_id,
                    keyword_id=kw.id,
                    weight=weight,
                    display=display or None,
                )
                db.add(assoc)
                stored.append({"keyword": kw_name, "display": display, "weight": weight})

            if extraction_id:
                e = (
                    db.query(KeywordExtraction)
                    .filter(KeywordExtraction.id == extraction_id)
                    .first()
                )
                if e:
                    e.status = "COMPLETED"
                    e.total_keywords = len(stored)

        diagnostics["stored"] = len(stored)
        logger.info(
            "Keyword extraction complete document_id=%s stored=%s diagnostics=%s",
            document_id,
            len(stored),
            diagnostics,
        )
        return {"keywords": stored, "count": len(stored), "diagnostics": diagnostics}
