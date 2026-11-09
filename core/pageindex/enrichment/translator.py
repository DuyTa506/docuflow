"""
Structured Document Translator.

Translates document structure while preserving hierarchy and organization.
"""

import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from config.settings import lang_name, normalize_lang_code
from utils.glossary import glossary_clause

from .base import BaseEnricher

logger = logging.getLogger(__name__)

# Table cells go to the model as "⟦k⟧ text" lines and come back keyed by k, so a
# dropped or reordered line cannot shift every cell after it.
_CELL_BATCH_SIZE = 40
_CELL_LINE_RE = re.compile(r"^\s*⟦(\d+)⟧\s?(.*)$")

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

# Shared with block/element/tree paths and the PDF-overlay adapter.
TRANSLATION_CONSTRAINTS = """\
TRANSLATE into the target language ALL readable content: body prose, headings,
section titles, table-of-contents entries, captions, blurbs, and imprint
descriptive sentences.

KEEP UNCHANGED only these identifiers (copy verbatim):
- Person names (authors, editors, photographers)
- Publisher / imprint house names
- ISBNs, library catalog codes (e.g. УДК, ББК, CIP), pure alphanumeric IDs
- Acronyms and technical identifiers (optional short target-language gloss once)
- Mathematical formulas, variable names, code spans
- Place names that are proper nouns (cities, streets)

Do NOT leave whole sentences, TOC lines, or descriptive phrases in the source
language. If a line mixes a proper noun with descriptive words, translate the
descriptive words and keep the proper noun."""


def _has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


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
        # Source term → Vietnamese term (utils/glossary.py), set by the caller.
        self.glossary: Dict[str, str] = {}
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
        if not _has_letters(text):
            return text  # "(" or "12.5 %": nothing to translate, the LLM only asks back

        # TOC pages are usually one fat layout element (dozens of leader-dot
        # lines). Even when under the token chunk budget, a single LLM call
        # truncates or leaves the source language — translate line batches.
        if _split_depth == 0:
            from utils.toc_text import looks_like_toc_text

            if looks_like_toc_text(text):
                return await self._translate_toc_lines(text)

        # Long TOC / imprint blocks must chunk first — a single call leaves a
        # tiny output budget, truncates, then degrades to source (still Russian).
        if _split_depth == 0 and self.count_tokens(text) > self.chunk_size:
            return await self.translate_text_chunked(text)

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

{TRANSLATION_CONSTRAINTS}
{glossary_clause(self.glossary, text)}
STRUCTURE PRESERVATION:
- Preserve all markdown formatting: **bold**, *italic*, `code`, links, headers, lists.
- Preserve paragraph breaks and section boundaries.
- Preserve numbered lists, bullet structures, and TOC leader dots (.....).
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

    async def _translate_toc_lines(self, text: str) -> str:
        """Translate a TOC/index block as small line batches (order preserved)."""
        from utils.toc_text import split_toc_line_batches

        batches = split_toc_line_batches(text)
        if len(batches) <= 1:
            # Still one unit — fall through to normal path with depth>0 so we
            # do not re-enter TOC detection.
            return await self.translate_text(text, _split_depth=1)

        from config.settings import settings as _settings

        semaphore = asyncio.Semaphore(max(1, _settings.translation_parallelism))

        async def _one(batch: str) -> str:
            async with semaphore:
                return await self.translate_text(batch, _split_depth=1)

        parts = await asyncio.gather(*(_one(b) for b in batches))
        return "\n".join(parts)

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

    async def translate_cells(self, cells: List[str]) -> List[str]:
        """Translate short table cells in marked batches; order and duplicates kept."""
        if self.source_lang == self.target_lang:
            return list(cells)
        done: Dict[str, str] = {}
        pending: List[str] = []
        for cell in dict.fromkeys(c for c in cells if c and c.strip()):
            hit = self.unit_cache.get("cell", cell) if self.unit_cache is not None else None
            if hit is not None:
                done[cell] = hit
            else:
                pending.append(cell)
        batches = [
            pending[i : i + _CELL_BATCH_SIZE] for i in range(0, len(pending), _CELL_BATCH_SIZE)
        ]
        results = await asyncio.gather(*(self._translate_cell_batch(b) for b in batches))
        for batch, translated in zip(batches, results):
            done.update(zip(batch, translated))
        return [done.get(cell, cell) for cell in cells]

    async def _translate_cell_batch(self, batch: List[str]) -> List[str]:
        src = lang_name(self.source_lang)
        tgt = lang_name(self.target_lang)
        listing = "\n".join(f"⟦{k}⟧ {' '.join(c.split())}" for k, c in enumerate(batch, 1))
        prompt = f"""{self._system_instruction}

TASK: Translate each table cell below from {src} to {tgt}.

{TRANSLATION_CONSTRAINTS}
{glossary_clause(self.glossary, listing)}
RULES:
- One output line per cell, starting with the same ⟦k⟧ marker.
- Keep numbers, units, symbols, part numbers and code unchanged.

CELLS:
{listing}

TRANSLATED CELLS:"""
        try:
            raw = await self.llm_client.chat_completion(
                prompt, max_tokens=self._output_budget(listing)
            )
        except Exception as exc:
            logger.warning("Table cell batch failed (%s) — translating cells one by one", exc)
            raw = ""
        parsed: Dict[int, str] = {}
        for line in str(raw or "").splitlines():
            matched = _CELL_LINE_RE.match(line)
            if matched and matched.group(2).strip():
                parsed[int(matched.group(1))] = matched.group(2).strip()

        out: List[str] = []
        for k, cell in enumerate(batch, 1):
            translated = parsed.get(k)
            if translated is None:
                translated = await self.translate_title(cell)
            elif self.unit_cache is not None:
                self.unit_cache.put("cell", cell, translated)
            out.append(translated)
        return out

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
        if not _has_letters(title):
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
{glossary_clause(self.glossary, title)}Keep it concise. Preserve any markdown formatting markers (#, ##, **, etc.).
Translate descriptive words; keep only person names / publisher names / codes unchanged.

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
