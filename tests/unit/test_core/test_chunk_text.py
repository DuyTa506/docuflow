from unittest.mock import MagicMock

from core.pageindex.enrichment.base import BaseEnricher


def _make_llm():
    """Fake LLM client whose count_tokens approximates 1 token per word."""
    llm = MagicMock()
    llm.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
    return llm


class TestChunkTextParagraphs:
    def test_respects_max_tokens_across_paragraphs(self):
        enricher = BaseEnricher(_make_llm())
        text = "\n\n".join(f"para {i} word " * 5 for i in range(10))
        chunks = enricher.chunk_text(text, max_tokens=20)
        assert all(enricher.count_tokens(c) <= 20 for c in chunks)
        assert len(chunks) > 1

    def test_no_text_duplicated_across_chunk_boundaries(self):
        """Regression test: the old overlap design re-emitted a chunk's last
        sentence as the next chunk's seed, so both chunks' *outputs* contained
        it — every chunk boundary duplicated text once translated."""
        enricher = BaseEnricher(_make_llm())
        paragraphs = [f"unique-paragraph-{i} " * 6 for i in range(8)]
        text = "\n\n".join(paragraphs)
        chunks = enricher.chunk_text(text, max_tokens=15)

        for i, para in enumerate(paragraphs):
            marker = f"unique-paragraph-{i}"
            occurrences = sum(marker in c for c in chunks)
            assert occurrences == 1, f"{marker} appeared in {occurrences} chunks"

    def test_single_line_text_still_chunks(self):
        enricher = BaseEnricher(_make_llm())
        text = "word " * 100  # no paragraph breaks at all
        chunks = enricher.chunk_text(text, max_tokens=20)
        assert len(chunks) > 1
        assert all(enricher.count_tokens(c) <= 20 for c in chunks)

    def test_oversized_paragraph_falls_back_to_sentence_split(self):
        enricher = BaseEnricher(_make_llm())
        # A single paragraph (no blank-line breaks) that alone exceeds max_tokens.
        text = "Sentence one is here. Sentence two follows. Sentence three ends it."
        chunks = enricher.chunk_text(text, max_tokens=5)
        assert len(chunks) > 1
        # No sentence should be silently dropped.
        joined = " ".join(chunks)
        assert "Sentence one" in joined
        assert "Sentence two" in joined
        assert "Sentence three" in joined

    def test_vietnamese_paragraph_without_ascii_period_space_still_splits(self):
        enricher = BaseEnricher(_make_llm())
        # Vietnamese sentences ending in '.', no following space captured by
        # the old ASCII '. ' splitter when punctuation differs; paragraph
        # breaks are what should carry the split here.
        text = "\n\n".join(f"Đoạn văn số {i} có nội dung tiếng Việt dài" for i in range(6))
        chunks = enricher.chunk_text(text, max_tokens=10)
        assert len(chunks) > 1
        assert all(enricher.count_tokens(c) <= 10 for c in chunks)

    def test_empty_text_returns_no_chunks(self):
        enricher = BaseEnricher(_make_llm())
        assert enricher.chunk_text("", max_tokens=100) == []
        assert enricher.chunk_text("   ", max_tokens=100) == []
