"""
Unit tests for KeywordService tree-first extraction.

RED phase: these tests are written BEFORE the implementation and must fail
until the tree-first logic is added to KeywordService.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_TREE = {
    "title": "Root Document",
    "summary": "Root summary about machine learning",
    "content": "root body text",
    "children": [
        {
            "title": "Chapter 1: Neural Networks",
            "summary": "Summary of neural network architecture",
            "content": "body content about backpropagation",
            "children": [],
        },
        {
            "title": "Chapter 2: Data Processing",
            "summary": "Summary of data pipeline",
            "content": "body text about preprocessing",
            "children": [
                {
                    "title": "Section 2.1: Tokenization",
                    "summary": "Tokenization methods explained",
                    "content": "token content",
                    "children": [],
                }
            ],
        },
    ],
}


# ── Tests ─────────────────────────────────────────────────────────────

class TestTreeCandidates:
    """_tree_candidates() collects titles and summaries from tree_data."""

    def test_tree_candidates_returns_titles_and_summaries(self):
        """All node titles and summaries appear as candidates."""
        from services.keyword_service import KeywordService

        svc = KeywordService()
        candidates = svc._tree_candidates(SAMPLE_TREE)

        texts = [c["keyword"] for c in candidates]
        assert any("Neural Networks" in t for t in texts), \
            "Node title 'Neural Networks' should appear in candidates"
        assert any("Tokenization" in t for t in texts), \
            "Leaf title 'Tokenization' should appear in candidates"

    def test_tree_candidates_title_has_higher_weight_than_body(self):
        """Title-sourced candidates outrank body-text candidates."""
        from services.keyword_service import KeywordService

        svc = KeywordService()
        candidates = svc._tree_candidates(SAMPLE_TREE)

        by_kw = {c["keyword"]: c["weight"] for c in candidates}
        title_kw = next(
            (c for c in candidates if "Neural Networks" in c["keyword"]), None
        )
        body_kw = next(
            (c for c in candidates if "backpropagation" in c["keyword"]), None
        )
        if title_kw and body_kw:
            assert title_kw["weight"] >= body_kw["weight"], \
                "Title candidates should have weight >= body candidates"

    def test_tree_candidates_normalizes_child_nodes_key(self):
        """Handles tree nodes that use 'child_nodes' instead of 'children'."""
        from services.keyword_service import KeywordService

        tree_alt = {
            "title": "Root",
            "summary": "root summary",
            "content": "",
            "child_nodes": [
                {
                    "title": "Alt Child Section",
                    "summary": "alt child summary",
                    "content": "",
                    "child_nodes": [],
                }
            ],
        }
        svc = KeywordService()
        candidates = svc._tree_candidates(tree_alt)
        texts = [c["keyword"] for c in candidates]
        assert any("Alt Child Section" in t for t in texts), \
            "Should handle 'child_nodes' key shape"


class TestExtractRouting:
    """_extract() routes to tree or TF-IDF depending on tree index presence."""

    @pytest.mark.asyncio
    async def test_extract_uses_tree_when_available(self):
        """When TreeIndex exists, _tfidf_candidates is NOT called."""
        from services.keyword_service import KeywordService

        svc = KeywordService()

        fake_tree_index = MagicMock()
        fake_tree_index.tree_data = SAMPLE_TREE

        with patch.object(svc, "_read_text", return_value="full document text"), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_tfidf_candidates", wraps=svc._tfidf_candidates) as mock_tfidf, \
             patch("services.keyword_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client") as mock_llm_factory:

            # DB returns a TreeIndex
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = fake_tree_index
            mock_dbm.return_value.session.return_value = mock_session

            mock_llm = AsyncMock()
            mock_llm.chat_completion = AsyncMock(
                return_value='[{"keyword": "neural network", "weight": 0.9}]'
            )
            mock_llm_factory.return_value = mock_llm
            svc._extract_json = MagicMock(
                return_value=[{"keyword": "neural network", "weight": 0.9}]
            )

            await svc._extract("DOC_001", 10)

            mock_tfidf.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_falls_back_to_tfidf_when_no_tree(self):
        """When no TreeIndex exists, _tfidf_candidates IS called."""
        from services.keyword_service import KeywordService

        svc = KeywordService()

        with patch.object(svc, "_read_text", return_value="full document text"), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_tfidf_candidates", return_value=[
                 {"keyword": "machine learning", "tfidf_score": 0.8}
             ]) as mock_tfidf, \
             patch("services.keyword_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client") as mock_llm_factory:

            # DB returns no TreeIndex
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            mock_llm = AsyncMock()
            mock_llm.chat_completion = AsyncMock(
                return_value='[{"keyword": "machine learning", "weight": 0.8}]'
            )
            mock_llm.count_tokens = MagicMock(return_value=10)
            mock_llm_factory.return_value = mock_llm
            svc._extract_json = MagicMock(
                return_value=[{"keyword": "machine learning", "weight": 0.8}]
            )

            await svc._extract("DOC_001", 10)

            mock_tfidf.assert_called_once()


class TestKeywordLanguage:
    @pytest.mark.asyncio
    async def test_prompt_requires_source_language_keywords(self):
        from services.keyword_service import KeywordService

        svc = KeywordService()

        with patch.object(svc, "_read_text", return_value="document text"), \
             patch.object(svc, "_progress"), \
             patch.object(svc, "_tfidf_candidates", return_value=[
                 {"keyword": "machine learning", "tfidf_score": 0.8}
             ]), \
             patch("services.keyword_service.get_db_manager") as mock_dbm, \
             patch("api.dependencies.get_llm_client") as mock_llm_factory:

            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = None
            mock_dbm.return_value.session.return_value = mock_session

            mock_llm = AsyncMock()
            mock_llm.chat_completion = AsyncMock(return_value="[]")
            mock_llm.count_tokens = MagicMock(return_value=10)
            mock_llm_factory.return_value = mock_llm
            svc._extract_json = MagicMock(return_value=[])

            await svc._extract("DOC_001", 10)

            prompt = mock_llm.chat_completion.call_args.args[0]
            assert "verbatim" in prompt
            assert "same language" in prompt
            assert "Do NOT translate" in prompt
