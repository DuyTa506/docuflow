"""
Base class for document enrichment operations.

Provides common utilities for enrichment features like translation,
summarization, keyword extraction, etc.
"""

import re
from typing import Dict, List, Optional
from ..llm.llm_client_base import BaseLLMClient

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


class BaseEnricher:
    """
    Base class for document enrichment operations.
    
    All enrichers (translator, summarizer, etc.) should inherit from this.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize enricher with LLM client.
        
        Args:
            llm_client: LLM client instance for making completions
        """
        self.llm_client = llm_client
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using LLM client's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.llm_client.count_tokens(text)
    
    def chunk_text(
        self,
        text: str,
        max_tokens: int = 2000,
        overlap: int = 100,
    ) -> List[str]:
        """
        Split long text into chunks that fit within token limit.

        Splits on paragraph boundaries first (matches markdown OCR output
        shape, and works for languages without ASCII sentence punctuation).
        A single paragraph that alone exceeds `max_tokens` falls back to a
        punctuation-based sentence split. Chunks are non-overlapping — an
        earlier "overlap" design re-emitted a chunk's last sentence as the
        next chunk's seed, but since each chunk is translated/summarised
        independently and the outputs are joined verbatim, that duplicated
        the seed sentence's text in both outputs.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Unused (kept for backward-compatible signature)

        Returns:
            List of text chunks
        """
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        units: List[str] = []
        for para in paragraphs:
            if self.count_tokens(para) <= max_tokens:
                units.append(para)
                continue
            for sentence in (s for s in _SENTENCE_BOUNDARY.split(para) if s.strip()):
                if self.count_tokens(sentence) <= max_tokens:
                    units.append(sentence)
                else:
                    # No sentence punctuation to split on either (e.g. raw
                    # OCR text with no periods) — fall back to a hard,
                    # word-boundary split so a single unit still respects
                    # max_tokens.
                    units.extend(self._split_oversized_unit(sentence, max_tokens))

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if current and current_tokens + unit_tokens > max_tokens:
                chunks.append("\n\n".join(current))
                current = [unit]
                current_tokens = unit_tokens
            else:
                current.append(unit)
                current_tokens += unit_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_oversized_unit(self, unit: str, max_tokens: int) -> List[str]:
        """Hard word-boundary split for a unit with no punctuation to break on."""
        words = unit.split()
        parts: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for word in words:
            word_tokens = self.count_tokens(word)
            if current and current_tokens + word_tokens > max_tokens:
                parts.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens
        if current:
            parts.append(" ".join(current))
        return parts or [unit]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Trim text so it encodes to at most max_tokens.

        Returns text unchanged if already within budget.
        OpenAI path uses tiktoken for exact truncation; Ollama path falls back
        to a ~4 chars/token heuristic (Ollama's token counter is itself heuristic).
        """
        if self.count_tokens(text) <= max_tokens:
            return text
        enc = getattr(self.llm_client, "encoding", None)
        if enc is not None:
            ids = enc.encode(text)
            return enc.decode(ids[:max_tokens])
        return text[: max_tokens * 4]

    async def process_with_retry(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> str:
        """
        Make LLM completion with retry logic.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retries
            
        Returns:
            LLM response
        """
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat_completion(prompt)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{max_retries} due to error: {e}")
                continue
        
        raise Exception("Max retries exceeded")
