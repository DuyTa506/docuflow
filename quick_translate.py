#!/usr/bin/env python3
"""
Quick translate - only translates titles and summaries.

Faster version that skips full content translation.
"""
import asyncio
import sys

from data.database import session_scope
from serving.storage_service import DocumentStorageService
from serving.tree_indexing_service import TreeIndexingService
from pageindex.llm import LLMClientFactory


async def quick_translate(document_id: str, output_path: str = None):
    """Quick translate - just titles and structure."""
    
    print("=" * 70)
    print("📝 Quick Translator (EN → VI) - Structure Only")
    print("=" * 70)
    
    # Get document data
    with session_scope() as session:
        storage = DocumentStorageService(session)
        document = storage.get_document(document_id)
        
        if not document:
            print(f"❌ Document not found: {document_id}")
            return
        
        filename = document.filename
        total_pages = document.total_pages
        markdown = storage.get_document_markdown(document_id)
        
        tree_service = TreeIndexingService(session)
        tree = tree_service.get_tree_index(document_id)
        
        if not tree:
            print("❌ No tree index found.")
            return
        
        tree_data = tree['tree_data']
    
    print(f"\n📄 Document: {filename}")
    print(f"📊 Pages: {total_pages}")
    
    # Initialize Ollama
    print("\n⏳ Initializing Ollama for translation...")
    
    llm_client = LLMClientFactory.create_client(
        provider='ollama',
        model='qwen3:30b',
        ollama_base_url='http://localhost:11434',
        ollama_timeout=300
    )
    
    async def translate(text: str) -> str:
        """Translate a piece of text."""
        if not text or not text.strip():
            return text
        
        prompt = f"""Dịch đoạn văn sau từ tiếng Anh sang tiếng Việt.
Giữ nguyên ý nghĩa và định dạng. Không giải thích thêm.

Văn bản cần dịch:
{text}

Bản dịch tiếng Việt (chỉ trả về bản dịch, không giải thích):"""
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'qwen3:30b',
                        'prompt': prompt,
                        'stream': False
                    }
                )
                result = response.json()
                return result.get('response', text).strip()
        except Exception as e:
            print(f"  ⚠️ Translation error: {e}")
            return text
    
    # Build Vietnamese markdown from structure
    print("\n🌐 Translating structure...")
    
    structure = tree_data.get('structure', [tree_data]) if isinstance(tree_data, dict) else [tree_data]
    
    vietnamese_md = []
    vietnamese_md.append(f"# {filename} - Bản dịch tiếng Việt\n")
    vietnamese_md.append("---\n")
    
    async def process_node(node, level=1):
        """Process a node and translate its title."""
        title = node.get('title', '')
        
        if title:
            print(f"  → Translating: {title[:50]}...")
            vi_title = await translate(title)
            vietnamese_md.append(f"{'#' * level} {vi_title}\n")
        
        summary = node.get('summary', '')
        if summary:
            print(f"    → Translating summary...")
            vi_summary = await translate(summary[:500])
            vietnamese_md.append(f"{vi_summary}\n")
        
        # Process children
        children = node.get('nodes', node.get('children', node.get('structure', [])))
        for child in children:
            await process_node(child, level + 1)
    
    for node in structure:
        await process_node(node, 1)
    
    # Add original markdown with headers
    vietnamese_md.append("\n---\n")
    vietnamese_md.append("## Nội dung gốc (tiếng Anh)\n\n")
    vietnamese_md.append(markdown)
    
    # Save
    if output_path is None:
        output_path = f"{filename}_vietnamese.md"
    
    content = '\n'.join(vietnamese_md)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "=" * 70)
    print(f"✅ Translation complete!")
    print(f"📄 Output: {output_path}")
    print(f"📊 Size: {len(content)} characters")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_translate.py <document_id> [output.md]")
        sys.exit(1)
    
    document_id = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(quick_translate(document_id, output_path))
