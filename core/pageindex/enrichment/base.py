"""
Base class for document enrichment operations.

Provides common utilities for enrichment features like translation,
summarization, keyword extraction, etc.
"""

import logging
import re
from typing import Dict, List, Optional

from ..llm.llm_client_base import BaseLLMClient

logger = logging.getLogger(__name__)

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
            # disallowed_special=(): source text may contain literal
            # '<|endoftext|>'-style strings — treat them as plain data.
            ids = enc.encode(text, disallowed_special=())
            return enc.decode(ids[:max_tokens])
        return text[: max_tokens * 4]

    async def process_with_retry(self, prompt: str, max_retries: int = 3, **llm_kwargs) -> str:
        """
        Make LLM completion with retry logic.

        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retries
            **llm_kwargs: Extra completion parameters (e.g. max_tokens)

        Returns:
            LLM response
        """
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat_completion(prompt, **llm_kwargs)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning("Retry %d/%d due to error: %s", attempt + 1, max_retries, e)
                continue

        raise Exception("Max retries exceeded")

    async def process_validated(
        self,
        prompt: str,
        *,
        validator,
        max_retries: int = 3,
        corrective: str = "",
        **llm_kwargs,
    ):
        """LLM completion with content validation on top of error retry.

        `validator(output, finish_reason)` returns a ValidationResult. On
        validation failure the call is retried (with `corrective` appended to
        the prompt from the second attempt). Returns `(output, reason)` where
        reason is None when valid — after exhausting retries the best
        candidate is returned with its failure reason instead of raising, so
        one bad unit degrades instead of failing the whole document.
        A `truncated` verdict short-circuits: retrying an over-budget chunk
        can't help — the caller must split it.
        """
        last_output = None
        last_reason = "error"
        for attempt in range(max_retries):
            p = prompt if attempt == 0 or not corrective else f"{prompt}\n\n{corrective}"
            try:
                client = self.llm_client
                if hasattr(client, "chat_completion_with_finish_reason"):
                    output, finish_reason = await client.chat_completion_with_finish_reason(
                        p, **llm_kwargs
                    )
                else:
                    output = await client.chat_completion(p, **llm_kwargs)
                    finish_reason = None
            except Exception as e:
                logger.warning("LLM error on attempt %d/%d: %s", attempt + 1, max_retries, e)
                last_reason = f"error: {e}"
                continue

            result = validator(output, finish_reason)
            if result.ok:
                return output, None
            last_output, last_reason = output, result.reason
            if result.reason == "truncated":
                break
            logger.warning(
                "Output validation failed (%s) on attempt %d/%d",
                result.reason,
                attempt + 1,
                max_retries,
            )
        return last_output, last_reason
