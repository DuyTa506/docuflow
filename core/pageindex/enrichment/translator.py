"""
Structured Document Translator.

Translates document structure while preserving hierarchy and organization.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from config.settings import lang_name, normalize_lang_code

from .base import BaseEnricher

logger = logging.getLogger(__name__)

# Shared with the PDF-overlay adapter (core/pdf_overlay/llm_adapter.py) so
# every translation path carries the same domain terminology guidance.
DOMAIN_INSTRUCTIONS = {
    "military": (
        "You are a professional military and defense translator. "
        "Use precise military terminology, ranks, unit designations, and tactical vocabulary. "
        "Preserve acronyms (NATO, etc.) and weapon/equipment designations exactly."
    ),
    "education": (
        "You are a professional academic and educational translator. "
        "Use clear pedagogical language appropriate for educational materials. "
        "Preserve curriculum terms, subject-area vocabulary, and academic conventions."
    ),
    "science": (
        "You are a professional scientific and technical translator. "
        "Use precise scientific terminology, SI units, and field-specific vocabulary. "
        "Preserve chemical formulas, mathematical notation, and technical abbreviations exactly."
    ),
    "general": (
        "You are a professional translator. "
        "Preserve the original meaning, tone, and style of the text."
    ),
}


class StructuredTranslator(BaseEnricher):
    """
    Translates document structure from source to target language.

    Preserves hierarchical organization while translating:
    - Titles
    - Text content
    - Summaries
    - Other textual fields
    """

    _DOMAIN_INSTRUCTIONS = DOMAIN_INSTRUCTIONS

    # Texts longer than this skip the per-run memo cache — duplicates are
    # rare past header/footer scale and the dict would just hold memory.
    _MEMO_MAX_CHARS = 500

    def __init__(
        self,
        llm_client,
        source_lang: str,
        target_lang: str,
        chunk_size: int = 8000,
        domain: str = "general",
    ):
        """
        Initialize translator.

        Args:
            llm_client: LLM client for translations
            source_lang: Source language code (e.g., 'en', 'vi')
            target_lang: Target language code (e.g., 'en', 'vi')
            chunk_size: Maximum tokens per translation chunk
            domain: Translation domain hint — 'general', 'military', 'education', 'science'
        """
        super().__init__(llm_client)
        self.source_lang = normalize_lang_code(source_lang)
        self.target_lang = normalize_lang_code(target_lang)
        self.chunk_size = chunk_size
        self.domain = domain
        # Units whose output failed validation after retries and fell back
        # to source text — surfaced in the task result for transparency.
        self.degraded_units = 0
        # Per-run translation memo: (kind, normalized_text) → asyncio.Future.
        # Instance is created per translation run, so scope/invalidations
        # are automatic.
        self._memo: dict = {}
        # Optional persistent unit cache (services/translators/_cache.py) —
        # when set, completed units survive crashes/retries so a re-run only
        # translates what's missing.
        self.unit_cache = None
        self._system_instruction = self._DOMAIN_INSTRUCTIONS.get(
            domain, self._DOMAIN_INSTRUCTIONS["general"]
        )

    def _output_budget(self, text: str) -> int:
        """max_tokens for a translation call — whatever slot budget remains
        after the source text and prompt scaffold. Without this the model
        default applies and long chunks are silently truncated mid-slot."""
        from config.settings import settings

        source_tokens = self.count_tokens(text)
        return max(
            256,
            settings.ai_chunk_tokens - source_tokens - settings.translation_prompt_overhead_tokens,
        )

    async def _with_memo(self, kind: str, text: str, produce) -> str:
        """Share one in-flight/completed translation per (kind, text) —
        repeated short strings (page headers/footers) recur on every page of
        a large book. Concurrent duplicates await the same future."""
        key = (kind, text)
        fut = self._memo.get(key)
        if fut is not None:
            return await fut
        fut = asyncio.get_running_loop().create_future()
        self._memo[key] = fut
        try:
            result = await produce()
        except BaseException as exc:
            self._memo.pop(key, None)
            fut.set_exception(exc)
            raise
        fut.set_result(result)
        return result

    async def _cached(self, kind: str, text: str, produce) -> str:
        """Consult/populate the persistent unit cache around `produce`.
        Degraded results (fell back to source after failed validation) are
        never cached — freezing a bad output would survive retries."""
        cache = self.unit_cache
        if cache is not None:
            hit = cache.get(kind, text)
            if hit is not None:
                return hit
        degraded_before = self.degraded_units
        result = await produce()
        if cache is not None and self.degraded_units == degraded_before:
            cache.put(kind, text, result)
        return result

    async def translate_text(self, text: str, _split_depth: int = 0) -> str:
        """
        Translate a piece of text, validating the output (empty / truncated /
        degenerate / wrong-language completions are retried, a truncated
        chunk is split in half deterministically, and after retries the unit
        degrades to source text instead of failing the document).

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        if self.source_lang == self.target_lang:
            return text

        if _split_depth == 0 and len(text) <= self._MEMO_MAX_CHARS:
            return await self._with_memo(
                "text",
                text.strip(),
                lambda: self._cached("text", text, lambda: self._translate_text_impl(text)),
            )
        if _split_depth == 0:
            return await self._cached("text", text, lambda: self._translate_text_impl(text))
        return await self._translate_text_impl(text, _split_depth)

    async def _translate_text_impl(self, text: str, _split_depth: int = 0) -> str:
        src = lang_name(self.source_lang)
        tgt = lang_name(self.target_lang)
        prompt = f"""{self._system_instruction}

TASK: Translate the following text from {src} to {tgt}.

TERMINOLOGY PRESERVATION:
- Preserve ALL proper nouns, acronyms, and technical identifiers exactly as-is.
- Proper nouns (names, places, organizations) → do NOT translate.
- Acronyms → keep original (optionally add target-language gloss in parentheses on first use).
- Codes, IDs, formulas, equations, variable names → copy verbatim.
- For domain-specific terms with no exact equivalent, keep the source term and add a brief gloss in brackets.

STRUCTURE PRESERVATION:
- Preserve all markdown formatting: **bold**, *italic*, `code`, links, headers, lists.
- Preserve paragraph breaks and section boundaries.
- Preserve numbered lists and bullet structures.
- Do NOT translate content inside code blocks or inline code spans.

OUTPUT: Return ONLY the translated text. No preamble, no commentary, no explanation.

SOURCE TEXT:
{text}

TRANSLATED TEXT:"""

        from .validation import validate_translation

        output, reason = await self.process_validated(
            prompt,
            validator=lambda out, finish: validate_translation(text, out, self.target_lang, finish),
            corrective=(
                f"IMPORTANT: Your previous output was incomplete or not in {tgt}. "
                f"Output the COMPLETE translation in {tgt} only."
            ),
            max_tokens=self._output_budget(text),
        )
        if reason is None:
            return output.strip()

        # Truncated chunk → deterministic fix: split in half and recurse.
        if reason == "truncated" and _split_depth < 3 and self.count_tokens(text) > 512:
            halves = self.chunk_text(text, max_tokens=max(256, self.count_tokens(text) // 2))
            if len(halves) > 1:
                parts = await asyncio.gather(
                    *(self.translate_text(h, _split_depth + 1) for h in halves)
                )
                return "\n\n".join(parts)

        logger.warning("Translation unit degraded to source text after retries (%s)", reason)
        self.degraded_units += 1
        return (output or text).strip()

    async def translate_text_chunked(self, text: str) -> str:
        """
        Translate long text by chunking.

        Args:
            text: Long text to translate

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        # Check if chunking needed
        if self.count_tokens(text) <= self.chunk_size:
            return await self.translate_text(text)

        # Split into chunks
        chunks = self.chunk_text(text, max_tokens=self.chunk_size)

        # Translate chunks concurrently (bounded), preserving order
        from config.settings import settings as _settings

        semaphore = asyncio.Semaphore(max(1, _settings.translation_parallelism))

        async def _translate_chunk(chunk: str) -> str:
            async with semaphore:
                return await self.translate_text(chunk)

        translated_chunks = await asyncio.gather(*(_translate_chunk(chunk) for chunk in chunks))

        # Combine chunks
        return "\n\n".join(translated_chunks)

    async def translate_title(self, title: str) -> str:
        """
        Translate a title/heading.

        Args:
            title: Title to translate

        Returns:
            Translated title
        """
        if not title or not title.strip():
            return title
        if self.source_lang == self.target_lang:
            return title

        if len(title) <= self._MEMO_MAX_CHARS:
            return await self._with_memo(
                "title",
                title.strip(),
                lambda: self._cached("title", title, lambda: self._translate_title_impl(title)),
            )
        return await self._cached("title", title, lambda: self._translate_title_impl(title))

    async def _translate_title_impl(self, title: str) -> str:
        src = lang_name(self.source_lang)
        tgt = lang_name(self.target_lang)
        prompt = f"""{self._system_instruction}

TASK: Translate this title/heading from {src} to {tgt}.
Keep it concise. Preserve any markdown formatting markers (#, ##, **, etc.).

Title: {title}

Translated title:"""

        from .validation import validate_translation

        output, reason = await self.process_validated(
            prompt,
            validator=lambda out, finish: validate_translation(
                title, out, self.target_lang, finish
            ),
            max_tokens=self._output_budget(title),
        )
        if reason is not None:
            self.degraded_units += 1
            return (output or title).strip()
        return output.strip()

    async def translate_node(self, node: Dict, depth: int = 0) -> Dict:
        """
        Translate a single node recursively.

        Args:
            node: Node dictionary to translate
            depth: Current depth (for progress display)

        Returns:
            Translated node
        """
        indent = "  " * depth
        title = node.get("title", "Untitled")
        logger.debug("%sTranslating: %s...", indent, title[:50])

        # Create copy to avoid modifying original
        translated_node = node.copy()

        # Translate title
        if "title" in node and node["title"]:
            translated_node["title"] = await self.translate_title(node["title"])

        # Translate text content if present
        if "text" in node and node["text"]:
            logger.debug("%s  → Translating text content...", indent)
            translated_node["text"] = await self.translate_text_chunked(node["text"])

        # Translate summary if present
        if "summary" in node and node["summary"]:
            logger.debug("%s  → Translating summary...", indent)
            translated_node["summary"] = await self.translate_text(node["summary"])

        # Recursively translate child nodes
        if "nodes" in node and node["nodes"]:
            logger.debug("%s  → Translating %d child nodes...", indent, len(node["nodes"]))
            translated_children = []
            for child in node["nodes"]:
                translated_child = await self.translate_node(child, depth + 1)
                translated_children.append(translated_child)
            translated_node["nodes"] = translated_children

        return translated_node

    async def translate_structure(
        self,
        structure: List[Dict],
        *,
        on_progress: Optional[Callable[[int, str], Any]] = None,
    ) -> List[Dict]:
        """
        Translate entire document structure.

        Every (node, field) unit across the whole tree is independent, so
        they are collected in one DFS pass and translated with bounded
        concurrency instead of one awaited call per node (the old serial
        walk left the LLM idle between nodes on large trees).

        Args:
            structure: List of root nodes
            on_progress: optional callback fired as units complete, so callers
                on very large trees (hours-long runs) get incremental progress
                instead of silence until the very end.

        Returns:
            Translated structure (deep copy — source nodes untouched)
        """
        import copy

        from config.settings import settings as _settings
        from services.translators._parallel import run_parallel

        logger.info(
            "Starting translation: %s → %s (%d root nodes)",
            self.source_lang,
            self.target_lang,
            len(structure),
        )

        translated_structure = copy.deepcopy(structure)

        units: List[tuple] = []  # (node_ref, field, translate_fn)

        def collect(node: Dict) -> None:
            if node.get("title"):
                units.append((node, "title", self.translate_title))
            if node.get("text"):
                units.append((node, "text", self.translate_text_chunked))
            if node.get("summary"):
                units.append((node, "summary", self.translate_text))
            for child in node.get("nodes") or []:
                collect(child)

        for root in translated_structure:
            collect(root)

        async def worker(_idx: int, unit: tuple) -> None:
            node, field, translate = unit
            node[field] = await translate(node[field])

        await run_parallel(
            units,
            worker,
            parallelism=max(1, _settings.translation_parallelism),
            on_progress=on_progress,
            progress_label="Unit",
        )

        logger.info("Translation complete (%d units)", len(units))
        return translated_structure
