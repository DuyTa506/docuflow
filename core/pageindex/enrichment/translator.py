"""
Structured Document Translator.

Translates document structure while preserving hierarchy and organization.
"""

import asyncio
from typing import Dict, List, Optional

from config.settings import lang_name, normalize_lang_code

from .base import BaseEnricher


class StructuredTranslator(BaseEnricher):
    """
    Translates document structure from source to target language.

    Preserves hierarchical organization while translating:
    - Titles
    - Text content
    - Summaries
    - Other textual fields
    """

    _DOMAIN_INSTRUCTIONS = {
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
        self._system_instruction = self._DOMAIN_INSTRUCTIONS.get(
            domain, self._DOMAIN_INSTRUCTIONS["general"]
        )
    
    async def translate_text(self, text: str) -> str:
        """
        Translate a piece of text.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        if self.source_lang == self.target_lang:
            return text

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

        response = await self.process_with_retry(prompt)
        return response.strip()
    
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

        translated_chunks = await asyncio.gather(
            *(_translate_chunk(chunk) for chunk in chunks)
        )

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

        src = lang_name(self.source_lang)
        tgt = lang_name(self.target_lang)
        prompt = f"""{self._system_instruction}

TASK: Translate this title/heading from {src} to {tgt}.
Keep it concise. Preserve any markdown formatting markers (#, ##, **, etc.).

Title: {title}

Translated title:"""

        response = await self.process_with_retry(prompt)
        return response.strip()
    
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
        title = node.get('title', 'Untitled')
        print(f"{indent}Translating: {title[:50]}...")
        
        # Create copy to avoid modifying original
        translated_node = node.copy()
        
        # Translate title
        if 'title' in node and node['title']:
            translated_node['title'] = await self.translate_title(node['title'])
        
        # Translate text content if present
        if 'text' in node and node['text']:
            print(f"{indent}  → Translating text content...")
            translated_node['text'] = await self.translate_text_chunked(node['text'])
        
        # Translate summary if present
        if 'summary' in node and node['summary']:
            print(f"{indent}  → Translating summary...")
            translated_node['summary'] = await self.translate_text(node['summary'])
        
        # Recursively translate child nodes
        if 'nodes' in node and node['nodes']:
            print(f"{indent}  → Translating {len(node['nodes'])} child nodes...")
            translated_children = []
            for child in node['nodes']:
                translated_child = await self.translate_node(child, depth + 1)
                translated_children.append(translated_child)
            translated_node['nodes'] = translated_children
        
        return translated_node
    
    async def translate_structure(self, structure: List[Dict]) -> List[Dict]:
        """
        Translate entire document structure.
        
        Args:
            structure: List of root nodes
            
        Returns:
            Translated structure
        """
        print(f"\n🌐 Starting translation: {self.source_lang} → {self.target_lang}")
        print(f"Processing {len(structure)} root nodes...\n")
        
        translated_structure = []
        for i, node in enumerate(structure, 1):
            print(f"\n[{i}/{len(structure)}] Root node:")
            translated_node = await self.translate_node(node)
            translated_structure.append(translated_node)
        
        print("\n✅ Translation complete!")
        return translated_structure
