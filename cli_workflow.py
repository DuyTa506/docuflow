#!/usr/bin/env python3
"""
CLI workflow runner for OCR-to-indexed-document pipeline.

Provides command-line interface for batch processing documents.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.database import session_scope, get_db_manager
from serving.storage_service import DocumentStorageService
from serving.tree_indexing_service import TreeIndexingService
from serving.logic import process_page_api
from openai import AsyncOpenAI


# Configuration
API_KEY = os.getenv("VLLM_API_KEY", "123")
SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")


async def process_document_cli(
    file_path: str,
    build_index: bool = False,
    llm_provider: str = "openai",
    model: str = "gpt-4o-2024-11-20"
):
    """Process a document through OCR and optionally build tree index."""
    
    print("=" * 60)
    print(f"Processing: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Determine file type
    file_type = 'pdf' if file_path.lower().endswith('.pdf') else 'image'
    filename = os.path.basename(file_path)
    
    # Count pages
    num_pages = 1
    if file_type == 'pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
    
    print(f"File type: {file_type}")
    print(f"Total pages: {num_pages}")
    print()
    
    # Create document in database
    with session_scope() as session:
        storage = DocumentStorageService(session)
        document = storage.create_document(
            filename=filename,
            file_type=file_type,
            total_pages=num_pages
        )
        document_id = document.id
        print(f"✓ Document created: {document_id}")
    
    # Process with OCR
    print(f"\nProcessing with DeepSeek OCR...")
    client = AsyncOpenAI(api_key=API_KEY, base_url=SERVER_URL)
    
    total_elements = 0
    for page_num in range(1, num_pages + 1):
        print(f"  Page {page_num}/{num_pages}...", end=" ", flush=True)
        
        # Process page
        page_result = None
        async for event in process_page_api(
            client=client,
            pdf_path=file_path,
            page_num=page_num,
            stream_enabled=False
        ):
            if event.get("type") == "result":
                page_result = event["result"]
        
        # Save to database
        if page_result:
            with session_scope() as session:
                storage = DocumentStorageService(session)
                storage.save_page_result(document_id, page_result)
                elem_count = len(page_result.layout_elements) if page_result.layout_elements else 0
                total_elements += elem_count
                print(f"✓ ({elem_count} elements)")
        else:
            print("⚠️  No result")
    
    print(f"\n✓ OCR complete: {total_elements} total elements extracted")
    
    # Build tree index if requested
    if build_index:
        print(f"\nBuilding tree index with {llm_provider}...")
        with session_scope() as session:
            tree_service = TreeIndexingService(
                session=session,
                llm_provider=llm_provider,
                model=model
            )
            tree_result = await tree_service.build_tree_index(
                document_id=document_id,
                if_thinning=True,
                if_add_node_summary="yes"
            )
        
        print(f"✓ Tree index created: {tree_result['tree_index_id']}")
        print(f"  Nodes: {tree_result['node_count']}")
        print(f"  Depth: {tree_result['max_depth']}")
    
    print("\n" + "=" * 60)
    print(f"✓ Complete! Document ID: {document_id}")
    print("=" * 60)
    
    return document_id


def list_documents_cli():
    """List all documents in database."""
    with session_scope() as session:
        storage = DocumentStorageService(session)
        documents = storage.list_documents(limit=50)
        
        if not documents:
            print("No documents found in database.")
            return
        
        print(f"\nFound {len(documents)} documents:")
        print("-" * 80)
        print(f"{'ID':<38} {'Filename':<30} {'Pages':<6} {'Created'}")
        print("-" * 80)
        
        for doc in documents:
            created = doc.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{doc.id:<38} {doc.filename:<30} {doc.total_pages:<6} {created}")


def show_tree_cli(document_id: str):
    """Show tree structure for a document."""
    with session_scope() as session:
        tree_service = TreeIndexingService(session)
        tree = tree_service.get_tree_index(document_id)
        
        if not tree:
            print(f"❌ No tree index found for document: {document_id}")
            print("   Build it first with: python cli_workflow.py build-tree <document_id>")
            return
        
        print(f"\nTree Index: {tree['tree_index_id']}")
        print(f"Created: {tree['created_at']}")
        print("\nTree Structure:")
        print("-" * 60)
        
        def print_node(node, indent=0):
            """Print a single node."""
            title = node.get('title', node.get('name', node.get('node_id', 'Unknown')))
            node_id = node.get('node_id', '')
            print("  " * indent + f"├─ [{node_id}] {title}")
            
            if node.get('summary'):
                summary = node['summary']
                # Truncate long summaries
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                # Print summary indented
                for line in summary.split('\n')[:3]:
                    if line.strip():
                        print("  " * indent + f"   {line.strip()}")
            
            children = node.get('children', node.get('child_nodes', []))
            for child in children:
                print_node(child, indent + 1)
        
        tree_data = tree['tree_data']
        
        # Handle PageIndex structure: {"doc_name": "...", "structure": [...]}
        if isinstance(tree_data, dict):
            doc_name = tree_data.get('doc_name', 'Document')
            structure = tree_data.get('structure', [])
            
            if structure:
                print(f"Document: {doc_name}")
                print(f"Sections: {len(structure)}")
                print("-" * 60)
                for node in structure:
                    print_node(node, 0)
            elif 'title' in tree_data or 'children' in tree_data:
                # Single root node
                print_node(tree_data, 0)
            else:
                print(f"Raw data: {tree_data}")
        elif isinstance(tree_data, list):
            for node in tree_data:
                print_node(node, 0)
        else:
            print(f"Unknown format: {type(tree_data)}")


async def build_tree_cli(document_id: str, llm_provider: str, model: str, use_spatial: bool):
    """Build tree index for an existing document."""
    print("=" * 60)
    print(f"Building tree index for document: {document_id}")
    print(f"LLM Provider: {llm_provider}")
    print(f"Model: {model}")
    print(f"Use Spatial: {use_spatial}")
    print("=" * 60)
    
    with session_scope() as session:
        storage = DocumentStorageService(session)
        document = storage.get_document(document_id)
        
        if not document:
            print(f"❌ Document not found: {document_id}")
            return
        
        print(f"\nDocument: {document.filename}")
        print(f"Pages: {document.total_pages}")
        
        tree_service = TreeIndexingService(
            session=session,
            llm_provider=llm_provider,
            model=model
        )
        
        print(f"\n⏳ Building tree with {llm_provider}/{model}...")
        
        if use_spatial:
            tree_result = await tree_service.build_enhanced_tree_index(
                document_id=document_id,
                if_thinning=True,
                if_add_node_summary="yes"
            )
        else:
            tree_result = await tree_service.build_tree_index(
                document_id=document_id,
                if_thinning=True,
                if_add_node_summary="yes"
            )
        
        print(f"\n✓ Tree index created!")
        print(f"  Tree ID: {tree_result['tree_index_id']}")
        print(f"  Nodes: {tree_result['node_count']}")
        print(f"  Max Depth: {tree_result['max_depth']}")
        print("=" * 60)


def export_markdown_cli(document_id: str, output_path: str = None):
    """Export document markdown to file."""
    with session_scope() as session:
        storage = DocumentStorageService(session)
        document = storage.get_document(document_id)
        
        if not document:
            print(f"❌ Document not found: {document_id}")
            return
        
        markdown = storage.get_document_markdown(document_id)
        
        if output_path is None:
            output_path = f"{document.filename}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"✓ Markdown exported to: {output_path}")
        print(f"  Document: {document.filename}")
        print(f"  Pages: {document.total_pages}")
        print(f"  Size: {len(markdown)} characters")


def main():
    parser = argparse.ArgumentParser(
        description='OCR-to-indexed-document CLI workflow'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process document with OCR')
    process_parser.add_argument('file', type=str, help='PDF or image file to process')
    process_parser.add_argument('--build-index', action='store_true', help='Build tree index after OCR')
    process_parser.add_argument('--llm-provider', type=str, default='openai', choices=['openai', 'ollama'], help='LLM provider')
    process_parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='Model name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all documents')
    
    # Show tree command
    tree_parser = subparsers.add_parser('show-tree', help='Show tree structure for document')
    tree_parser.add_argument('document_id', type=str, help='Document ID')
    
    # Build tree command (standalone)
    build_parser = subparsers.add_parser('build-tree', help='Build tree index for existing document')
    build_parser.add_argument('document_id', type=str, help='Document ID')
    build_parser.add_argument('--llm-provider', type=str, default='ollama', choices=['openai', 'ollama'], help='LLM provider')
    build_parser.add_argument('--model', type=str, default='qwen3:30b', help='Model name')
    build_parser.add_argument('--use-spatial', action='store_true', help='Use spatial metadata for enhanced tree')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export document markdown')
    export_parser.add_argument('document_id', type=str, help='Document ID')
    export_parser.add_argument('-o', '--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        asyncio.run(process_document_cli(
            file_path=args.file,
            build_index=args.build_index,
            llm_provider=args.llm_provider,
            model=args.model
        ))
    elif args.command == 'list':
        list_documents_cli()
    elif args.command == 'show-tree':
        show_tree_cli(args.document_id)
    elif args.command == 'build-tree':
        asyncio.run(build_tree_cli(
            document_id=args.document_id,
            llm_provider=args.llm_provider,
            model=args.model,
            use_spatial=args.use_spatial
        ))
    elif args.command == 'export':
        export_markdown_cli(args.document_id, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
