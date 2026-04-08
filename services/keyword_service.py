"""
Keyword extraction service.

Uses a hybrid TF-IDF + LLM approach:
1. TF-IDF (scikit-learn) processes the FULL document text to extract statistical
   candidate keywords — no truncation, handles 100+ page documents.
2. The LLM refines the candidates: filters noise, re-ranks with semantic weights,
   and may add up to 5 additional keywords not caught by statistics.
"""
from typing import List, Dict

from data.database import get_db_manager
from data.db_models import Keyword, DocumentKeyword
from services.base_service import BaseTaskService
from services.task_manager import task_manager


class KeywordService(BaseTaskService):
    """Keyword extraction (background task)."""

    def submit(self, db, document_id: str, max_keywords: int = 20) -> str:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")
        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="KEYWORDS",
            coro=self._extract(document_id, max_keywords),
        )
        return task_id

    # ── TF-IDF candidate extraction ──────────────────────────────────

    def _tfidf_candidates(self, text: str, max_candidates: int = 50) -> List[Dict]:
        """
        Extract top candidate keywords from the full document text using TF-IDF.

        Args:
            text: Full document text (no length limit).
            max_candidates: Number of candidates to return.

        Returns:
            List of {"keyword": str, "tfidf_score": float}, sorted descending by score.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Treat the whole document as a single "document" for scoring purposes.
        # Wrapping in a tiny corpus of the doc + a dummy blank doc lets TF-IDF
        # produce non-trivial IDF weights without needing a full corpus.
        corpus = [text, ""]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",   # removes common English stopwords;
                                    # Vietnamese stopwords are handled by the LLM
                                    # refinement step in Phase B
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF][a-zA-Z\u00C0-\u024F\u1E00-\u1EFF0-9\-]{1,}\b",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Scores from the first document row
        doc_scores = tfidf_matrix[0].toarray()[0]

        # Pair and sort descending
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

    async def _extract(self, document_id: str, max_keywords: int):
        db_manager = get_db_manager()

        # Load FULL text — no truncation
        text = self._read_text(document_id)
        task_id = self._find_task_id(document_id, "KEYWORDS")

        self._progress(task_id, 10, "Running TF-IDF on full document text")

        # Phase A: statistical candidates from full text
        candidates = self._tfidf_candidates(text, max_candidates=50)

        self._progress(task_id, 40, "LLM refinement of keyword candidates")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        # Build candidate list string for the prompt
        candidate_lines = "\n".join(
            f"  {i+1}. {c['keyword']} (tfidf={c['tfidf_score']})"
            for i, c in enumerate(candidates)
        )

        # Phase B: LLM refines candidates using a text sample for context
        text_sample = text[:8000]

        prompt = (
            f"You are a keyword extraction expert.\n\n"
            f"Below are statistical keyword candidates extracted via TF-IDF from a document, "
            f"followed by the first portion of the document for context.\n\n"
            f"STATISTICAL CANDIDATES:\n{candidate_lines}\n\n"
            f"DOCUMENT EXCERPT (first 8 000 chars):\n{text_sample}\n\n"
            f"TASK: Select the {max_keywords} most relevant academic/technical keywords. "
            f"Re-rank with an importance weight from 0.0 to 1.0. "
            f"You may add up to 5 keywords NOT in the candidate list if they are clearly "
            f"important and grounded in the document text.\n\n"
            f"Return ONLY valid JSON as a list:\n"
            f'[{{"keyword": "example term", "weight": 0.95}}, ...]\n\nJSON:'
        )

        response = await llm.chat_completion(prompt)

        self._progress(task_id, 70, "Parsing refined keywords")

        keywords_list = self._extract_json(llm, response, list_key="keywords")

        # Fallback: if LLM returned nothing usable, use top TF-IDF candidates directly.
        # Clamp tfidf_score to [0.0, 1.0] — raw TF-IDF scores can exceed 1.0 with
        # sublinear_tf=True on a 2-doc corpus.
        if not keywords_list and candidates:
            keywords_list = [
                {"keyword": c["keyword"], "weight": min(c["tfidf_score"], 1.0)}
                for c in candidates[:max_keywords]
            ]

        # Store keywords
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

        return {"keywords": stored, "count": len(stored)}
