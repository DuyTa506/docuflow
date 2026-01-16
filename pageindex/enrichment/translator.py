"""
Structured Document Translator.

Translates document structure while preserving hierarchy and organization.
"""

import asyncio
from typing import Dict, List, Optional
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
    
    def __init__(
        self,
        llm_client,
        source_lang: str,
        target_lang: str,
        chunk_size: int = 8000
    ):
        """
        Initialize translator.
        
        Args:
            llm_client: LLM client for translations
            source_lang: Source language code (e.g., 'en', 'vi')
            target_lang: Target language code (e.g., 'en', 'vi')
            chunk_size: Maximum tokens per translation chunk
        """
        super().__init__(llm_client)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.chunk_size = chunk_size
    
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
        
        prompt = f"""Translate the following text from {self.source_lang} to {self.target_lang}.
Preserve the original meaning and tone. Do not add explanations.

Text to translate:
{text}

Translated text:"""
        
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
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Translating chunk {i+1}/{len(chunks)}...")
            translated = await self.translate_text(chunk)
            translated_chunks.append(translated)
        
        # Combine chunks
        return ' '.join(translated_chunks)
    
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
        
        prompt = f"""Translate this title/heading from {self.source_lang} to {self.target_lang}.
Keep it concise and preserve any formatting markers.

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
        print(f"{indent}📝 Translating: {title[:50]}...")
        
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
    
    async def translate_document(self, document: Dict) -> Dict:
        """
        Translate complete document with metadata.
        
        Args:
            document: Document dict with 'doc_name' and 'structure'
            
        Returns:
            Translated document
        """
        result = {
            'doc_name': document.get('doc_name', 'untitled'),
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'structure': await self.translate_structure(document.get('structure', []))
        }
        
        return result
