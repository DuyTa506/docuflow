"""
Keyword extraction service.

Hybrid approach — outline *and* body, never one alone:
1. Node titles from the TreeIndex give structure-aware candidates.
2. TF-IDF over the full document text supplies terms the headings never name.
3. The LLM re-ranks against a stratified excerpt of the document itself.

Step 2 used to run only when the tree produced fewer than 10 candidates, so
any document with a real outline had its keywords chosen from headings alone.
"""

from typing import Dict, List, Optional

from config.settings import pipeline_keyword_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import DocumentKeyword, Keyword, KeywordExtraction
from services.base_service import BaseTaskService
from services.task_manager import task_manager

# Weight assigned to each candidate source tier
_WEIGHT_TITLE = 1.0

# Headings are plentiful and cheap; leave room for content-derived terms.
_MAX_TREE_CANDIDATES = 40
_MAX_TFIDF_CANDIDATES = 50
_MAX_CANDIDATES = 80


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
        the model through TF-IDF terms and the excerpt instead.

        Returns de-duplicated list of {"keyword": str, "weight": float}.
        """
        seen: dict[str, float] = {}  # keyword → max weight seen

        def _walk(node: dict):
            title = (node.get("title") or node.get("name") or "").strip()
            if title:
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

    # ── TF-IDF candidate extraction ──────────────────────────────────

    def _tfidf_candidates(self, text: str, max_candidates: int = 50) -> List[Dict]:
        """
        Extract top candidate keywords from the full document text using TF-IDF.

        Returns list of {"keyword": str, "tfidf_score": float}, sorted descending.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [text, ""]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF][a-zA-Z\u00C0-\u024F\u1E00-\u1EFF0-9\-]{1,}\b",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        doc_scores = tfidf_matrix[0].toarray()[0]

        scored = sorted(
            zip(feature_names, doc_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"keyword": kw, "tfidf_score": round(float(score), 4)}
            for kw, score in scored[:max_candidates]
            if score > 0
        ]

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

        from api.dependencies import get_llm_client

        llm = get_llm_client()

        # ── Phase A: load candidates ───────────────────────────────
        self._progress(task_id, 10, "Loading keyword candidates")

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
            self._progress(task_id, 25, f"Tree index: {len(candidates)} heading candidates")

        # Always supplement with TF-IDF. Headings name sections, not concepts:
        # a term discussed throughout the book but never used in a heading was
        # previously unreachable whenever a tree existed.
        text = self._read_text(document_id)
        self._progress(task_id, 30, "Running TF-IDF on full document text")
        tfidf = self._tfidf_candidates(text, max_candidates=_MAX_TFIDF_CANDIDATES)
        existing_kws = {c["keyword"].lower() for c in candidates}
        for c in tfidf:
            if c["keyword"].lower() not in existing_kws:
                candidates.append({"keyword": c["keyword"], "weight": min(c["tfidf_score"], 1.0)})
                existing_kws.add(c["keyword"].lower())
        candidates = candidates[:_MAX_CANDIDATES]

        self._progress(task_id, 40, "LLM refinement of keyword candidates")

        # ── Phase B: LLM reranking ────────────────────────────────────
        candidate_lines = "\n".join(
            f"  {i+1}. {c['keyword']} (weight={c.get('weight', c.get('tfidf_score', 1.0)):.2f})"
            for i, c in enumerate(candidates)
        )

        # The excerpt is the grounding evidence: "must appear verbatim in the
        # document" is unverifiable for a model that only sees an outline.
        from utils.doc_sampling import build_pipeline_doc_sample

        budget = min(settings.ai_chunk_tokens - 2000, 8000)
        excerpt = build_pipeline_doc_sample(document_id, text, BaseEnricher(llm), budget)
        context_block = f"DOCUMENT EXCERPT:\n{excerpt}"

        prompt = (
            "You are a keyword extraction expert for academic and technical documents.\n\n"
            f"CANDIDATES (from document structure):\n{candidate_lines}\n\n"
            f"{context_block}\n\n"
            f"TASK: Select the {max_keywords} most relevant academic/technical keywords "
            "from the candidates above. Re-rank with an importance weight from 0.0 to 1.0.\n\n"
            "GROUNDING RULES:\n"
            "- Every selected keyword MUST appear verbatim as a contiguous phrase in the document text.\n"
            "- Do NOT generate synonyms, hypernyms, or related terms not in the source.\n"
            "- If a candidate is a paraphrase or abstraction, REJECT it.\n"
            "- Prefer noun phrases and proper nouns over generic terms.\n\n"
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

        response = await llm.chat_completion(prompt)
        self._progress(task_id, 70, "Parsing refined keywords")

        keywords_list = self._extract_json(llm, response, list_key="keywords")

        # Fallback: use top candidates directly if LLM failed
        if not keywords_list and candidates:
            keywords_list = [
                {
                    "keyword": c["keyword"],
                    "weight": min(c.get("weight", c.get("tfidf_score", 1.0)), 1.0),
                }
                for c in candidates[:max_keywords]
            ]

        # ── Phase C: persist ───────────────────────────────────
        with db_manager.session() as db:
            db.query(DocumentKeyword).filter(DocumentKeyword.document_id == document_id).delete()

            stored = []
            for item in keywords_list[:max_keywords]:
                kw_name = item.get("keyword", "").strip()
                weight = float(item.get("weight", 1.0))
                display = (item.get("display") or "").strip()
                if not kw_name:
                    continue

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

            # Mark extraction COMPLETED in same session
            if extraction_id:
                e = (
                    db.query(KeywordExtraction)
                    .filter(KeywordExtraction.id == extraction_id)
                    .first()
                )
                if e:
                    e.status = "COMPLETED"
                    e.total_keywords = len(stored)

        return {"keywords": stored, "count": len(stored)}
