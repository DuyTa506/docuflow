"""
Keyword extraction service.

Tree-first hybrid approach:
1. If a TreeIndex exists for the document, extract candidates from node titles,
   summaries, and body text (structure-aware, no truncation).
2. If no tree index, or the tree yields < 10 candidates, fall back to full-text
   TF-IDF (original approach).
3. LLM refines and re-ranks the final candidate list.
"""
from typing import List, Dict, Optional

from config.settings import pipeline_keyword_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import Keyword, DocumentKeyword, KeywordExtraction
from services.base_service import BaseTaskService
from services.task_manager import task_manager

# Weight assigned to each candidate source tier
_WEIGHT_TITLE = 1.0
_WEIGHT_SUMMARY = 0.7
_WEIGHT_BODY = 0.4

# Minimum number of tree candidates before we supplement with TF-IDF
_TREE_MIN_CANDIDATES = 10


class KeywordService(BaseTaskService):
    """Keyword extraction (background task)."""

    def submit(self, db, document_id: str, max_keywords: int = 20) -> tuple:
        """Create a KeywordExtraction(PENDING) record and submit background task.

        Returns (task_id, extraction_id).
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

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
            coro=self._extract(document_id, max_keywords, extraction_id),
        )
        return task_id, extraction_id

    # ── Tree candidate extraction ─────────────────────────────────────

    def _tree_candidates(self, tree_data: dict) -> List[Dict]:
        """
        Walk tree_data recursively and collect candidate keywords from:
          - node title  (weight=1.0)
          - node summary (weight=0.7)
          - node body text (weight=0.4)

        Handles mixed key shapes:
          children  : 'children' or 'child_nodes'
          body text : 'content' / 'text' / 'text_content'
          title     : 'title' or 'name'

        Returns de-duplicated list of {"keyword": str, "weight": float}.
        """
        seen: dict[str, float] = {}  # keyword → max weight seen

        def _walk(node: dict):
            title = (node.get("title") or node.get("name") or "").strip()
            summary = (node.get("summary") or "").strip()
            body = (
                node.get("content")
                or node.get("text")
                or node.get("text_content")
                or ""
            ).strip()

            if title:
                seen[title] = max(seen.get(title, 0.0), _WEIGHT_TITLE)
            if summary:
                # Split summary into phrases; use full summary as one candidate
                seen[summary] = max(seen.get(summary, 0.0), _WEIGHT_SUMMARY)
            if body:
                seen[body] = max(seen.get(body, 0.0), _WEIGHT_BODY)

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

    async def _extract(self, document_id: str, max_keywords: int, extraction_id: str = None):
        db_manager = get_db_manager()

        def _update_extraction(**fields):
            if not extraction_id:
                return
            with db_manager.session() as db:
                e = db.query(KeywordExtraction).filter(KeywordExtraction.id == extraction_id).first()
                if e:
                    for k, v in fields.items():
                        setattr(e, k, v)

        _update_extraction(status="IN_PROGRESS")

        try:
            await self._do_extract(document_id, max_keywords, extraction_id)
        except Exception as exc:
            _update_extraction(status="FAILED", error=str(exc))
            raise

    async def _do_extract(self, document_id: str, max_keywords: int, extraction_id: str = None):
        db_manager = get_db_manager()
        task_id = self._find_task_id(document_id, "KEYWORDS")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        # ── Phase A: load candidates ───────────────────────────────
        self._progress(task_id, 10, "Loading keyword candidates")

        # Try tree index first
        with db_manager.session() as db:
            from data.db_models import TreeIndex
            tree_index = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )

        use_tree = tree_index is not None
        candidates: List[Dict] = []

        if use_tree:
            tree_candidates = self._tree_candidates(tree_index.tree_data)
            candidates = tree_candidates[:50]
            self._progress(task_id, 25, f"Tree index: {len(candidates)} candidates")

        # Supplement with TF-IDF if no tree or too few candidates
        if len(candidates) < _TREE_MIN_CANDIDATES:
            text = self._read_text(document_id)
            self._progress(task_id, 30, "Running TF-IDF on full document text")
            tfidf = self._tfidf_candidates(text, max_candidates=50)
            # Merge: keep tree candidates, add TF-IDF ones not already present
            existing_kws = {c["keyword"].lower() for c in candidates}
            for c in tfidf:
                if c["keyword"].lower() not in existing_kws:
                    candidates.append({"keyword": c["keyword"], "weight": min(c["tfidf_score"], 1.0)})
                    existing_kws.add(c["keyword"].lower())
            candidates = candidates[:50]

        self._progress(task_id, 40, "LLM refinement of keyword candidates")

        # ── Phase B: LLM reranking ────────────────────────────────────
        candidate_lines = "\n".join(
            f"  {i+1}. {c['keyword']} (weight={c.get('weight', c.get('tfidf_score', 1.0)):.2f})"
            for i, c in enumerate(candidates)
        )

        # Use tree section titles as context if available, else first 4k of text
        if use_tree:
            context_lines = "\n".join(
                f"- {c['keyword']}" for c in candidates if c.get("weight", 0) >= _WEIGHT_TITLE
            )
            context_block = f"DOCUMENT SECTIONS:\n{context_lines}"
        else:
            text = self._read_text(document_id)
            budget = min(settings.ai_chunk_tokens - 2000, 8000)
            excerpt = BaseEnricher(llm).truncate_to_tokens(text, budget)
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
            "Return ONLY valid JSON as a list:\n"
            '[{"keyword": "example term", "weight": 0.95}, ...]\n\nJSON:'
        )

        response = await llm.chat_completion(prompt)
        self._progress(task_id, 70, "Parsing refined keywords")

        keywords_list = self._extract_json(llm, response, list_key="keywords")

        # Fallback: use top candidates directly if LLM failed
        if not keywords_list and candidates:
            keywords_list = [
                {"keyword": c["keyword"], "weight": min(c.get("weight", c.get("tfidf_score", 1.0)), 1.0)}
                for c in candidates[:max_keywords]
            ]

        # ── Phase C: persist ───────────────────────────────────
        with db_manager.session() as db:
            db.query(DocumentKeyword).filter(DocumentKeyword.document_id == document_id).delete()

            stored = []
            for item in keywords_list[:max_keywords]:
                kw_name = item.get("keyword", "").strip()
                weight = float(item.get("weight", 1.0))
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
                )
                db.add(assoc)
                stored.append({"keyword": kw_name, "weight": weight})

            # Mark extraction COMPLETED in same session
            if extraction_id:
                e = db.query(KeywordExtraction).filter(KeywordExtraction.id == extraction_id).first()
                if e:
                    e.status = "COMPLETED"
                    e.total_keywords = len(stored)

        return {"keywords": stored, "count": len(stored)}
