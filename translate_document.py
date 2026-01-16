#!/usr/bin/env python3
"""
Translate document from English to Vietnamese.

Uses the PageIndex translator module with Ollama.
"""
import asyncio
import json
import sys
from pathlib import Path

from data.database import session_scope
from serving.storage_service import DocumentStorageService
from serving.tree_indexing_service import TreeIndexingService
from pageindex.enrichment.translator import StructuredTranslator
from pageindex.llm import LLMClientFactory


async def translate_document(document_id: str, output_path: str = None):
    """Translate document markdown and tree to Vietnamese."""
    
    print("=" * 70)
    print("📝 Document Translator (EN → VI)")
    print("=" * 70)
    
    # Get document data - store values before leaving session
    with session_scope() as session:
        storage = DocumentStorageService(session)
        document = storage.get_document(document_id)
        
        if not document:
            print(f"❌ Document not found: {document_id}")
            return
        
        # Extract values while in session
        filename = document.filename
        total_pages = document.total_pages
        
        print(f"\n📄 Document: {filename}")
        print(f"📊 Pages: {total_pages}")
        
        # Get markdown
        markdown = storage.get_document_markdown(document_id)
        
        # Get tree
        tree_service = TreeIndexingService(session)
        tree = tree_service.get_tree_index(document_id)
        
        if not tree:
            print("❌ No tree index found. Build it first with --use-spatial")
            return
        
        tree_id = tree['tree_index_id']
        tree_data = tree['tree_data']
    
    print(f"\n🌲 Tree: {tree_id}")
    
    # Initialize Ollama client for translation
    print("\n⏳ Initializing Ollama (qwen3:30b) for translation...")
    
    llm_client = LLMClientFactory.create_client(
        provider='ollama',
        model='qwen3:30b',
        ollama_base_url='http://localhost:11434',
        ollama_timeout=300
    )
    
    # Create translator
    translator = StructuredTranslator(
        llm_client=llm_client,
        source_lang='English',
        target_lang='Vietnamese',
        chunk_size=1500
    )
    
    # Translate tree structure
    print("\n" + "=" * 70)
    print("🌐 Translating document structure...")
    print("=" * 70)
    
    # Handle PageIndex format
    if isinstance(tree_data, dict) and 'structure' in tree_data:
        doc_data = tree_data
    else:
        doc_data = {'doc_name': filename, 'structure': [tree_data]}
    
    translated_doc = await translator.translate_document(doc_data)
    
    # Convert to markdown
    print("\n📝 Converting to Vietnamese markdown...")
    
    def structure_to_markdown(structure, level=1):
        """Convert structure to markdown."""
        md = []
        for node in structure:
            title = node.get('title', '')
            if title:
                md.append(f"{'#' * level} {title}\n")
            
            if node.get('summary'):
                md.append(f"{node['summary']}\n")
            
            if node.get('text'):
                md.append(f"\n{node['text']}\n")
            
            # Handle children
            children = node.get('nodes', node.get('children', []))
            if children:
                md.append(structure_to_markdown(children, level + 1))
        
        return '\n'.join(md)
    
    vietnamese_md = structure_to_markdown(translated_doc.get('structure', []))
    
    # Also translate original markdown
    print("\n⏳ Translating original markdown content...")
    
    # Split markdown into chunks and translate
    chunk_size = 2000
    chunks = [markdown[i:i+chunk_size] for i in range(0, len(markdown), chunk_size)]
    
    translated_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Translating chunk {i}/{len(chunks)}...")
        translated = await translator.translate_text(chunk)
        translated_chunks.append(translated)
    
    full_translated_md = '\n'.join(translated_chunks)
    
    # Save output
    if output_path is None:
        output_path = f"{filename}_vietnamese.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# {filename} - Bản dịch tiếng Việt\n\n")
        f.write("---\n\n")
        f.write("## Cấu trúc tài liệu\n\n")
        f.write(vietnamese_md)
        f.write("\n\n---\n\n")
        f.write("## Nội dung đầy đủ\n\n")
        f.write(full_translated_md)
    
    print("\n" + "=" * 70)
    print(f"✅ Translation complete!")
    print(f"📄 Output: {output_path}")
    print(f"📊 Size: {len(full_translated_md)} characters")
    print("=" * 70)
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_document.py <document_id> [output.md]")
        print("\nExample:")
        print("  python translate_document.py d0a54df7-3a1f-4279-990f-16e9af2c1ca1")
        sys.exit(1)
    
    document_id = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(translate_document(document_id, output_path))
