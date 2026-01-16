# OCR-to-Indexed-Document Workflow

Complete end-to-end pipeline for converting PDF/images to structured, indexed documents with metadata.

## Quick Start

### 1. Initialize Database

```bash
# Initialize the database tables
python init_db.py
```

### 2. Start vLLM Server (DeepSeek OCR)

```bash
# Start vLLM with DeepSeek OCR model
bash serve_deepseek_ocr.sh
```

### 3. Process Documents

**Option A: Using CLI**
```bash
# Process a single document
python cli_workflow.py process research_paper.pdf --build-index

# List all documents
python cli_workflow.py list

# Show tree structure
python cli_workflow.py show-tree <document_id>

# Export markdown
# DeepSeek OCR + PageIndex Workflow System

## Overview

A complete document processing system that combines:
- **DeepSeek OCR** - High-quality OCR with spatial metadata extraction
- **PageIndex** - Intelligent document tree indexing and structure analysis
- **Spatial Enhancement** - Layout-aware hierarchy detection
- **Database Storage** - Persistent storage with SQLAlchemy
- **REST API** - FastAPI endpoints for all operations
- **CLI** - Command-line interface for batch processing

## Key Features

1. **OCR Processing** - Extract text and layout from PDFs/images using DeepSeek vLLM
2. **Spatial Metadata** - Capture bounding boxes, labels, and visual hierarchy
3. **Tree Indexing** - Build hierarchical document structure with PageIndex
4. **Enhanced Trees** - Combine markdown + spatial data for better accuracy
5. **Multi-LLM Support** - Use OpenAI or Ollama for tree generation
6. **Translation & Summarization** - Document enrichment workflows (via PageIndex)

### Process Document
```bash
curl -X POST "http://localhost:8002/process-document" \
  -F "file=@document.pdf" \
  -F "store_to_db=true"
```

Response:
```json
{
  "document_id": "abc-123-xyz",
  "filename": "document.pdf",
  "total_pages": 10,
  "element_count": 142,
  "stored_to_db": true
}
```

### Build Tree Index
```bash
curl -X POST "http://localhost:8002/build-index/abc-123-xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "if_thinning": true,
    "if_add_node_summary": "yes",
    "llm_provider": "openai"
  }'
```

Response:
```json
{
  "tree_index_id": "tree-456-def",
  "node_count": 45,
  "max_depth": 4
}
```

### Get Document
```bash
curl "http://localhost:8002/documents/abc-123-xyz"
```

### Get Layout Elements
```bash
# Get all elements
curl "http://localhost:8002/documents/abc-123-xyz/elements"

# Get only images
curl "http://localhost:8002/documents/abc-123-xyz/elements?label=image"

# Get only tables
curl "http://localhost:8002/documents/abc-123-xyz/elements?label=table"
```

### Get Tree Structure
```bash
curl "http://localhost:8002/documents/abc-123-xyz/tree"
```

### List Documents
```bash
curl "http://localhost:8002/documents?limit=50&offset=0"
```

## Python API Usage

```python
import asyncio
from serving.database import session_scope
from serving.storage_service import DocumentStorageService
from serving.tree_indexing_service import TreeIndexingService

# Get a document
with session_scope() as session:
    storage = DocumentStorageService(session)
    
    # Get document
    doc = storage.get_document(document_id)
    print(f"Document: {doc.filename}, {doc.total_pages} pages")
    
    # Get markdown
    markdown = storage.get_document_markdown(document_id)
    
    # Get all images
    images = storage.get_document_elements(document_id, label_filter='image')
    print(f"Found {len(images)} images")
    
    # Get tree index
    tree_service = TreeIndexingService(session)
    tree = tree_service.get_tree_index(document_id)
    print(f"Tree has {tree['node_count']} nodes")
```

## CLI Commands

### Process Document
```bash
# Basic processing
python cli_workflow.py process document.pdf

# With tree indexing
python cli_workflow.py process document.pdf --build-index

# With Ollama
python cli_workflow.py process document.pdf --build-index \
  --llm-provider ollama --model llama2
```

### List Documents
```bash
python cli_workflow.py list
```

Output:
```
ID                                   Filename                    Pages  Created
--------------------------------------------------------------------------------
abc-123-xyz                         research_paper.pdf          10     2026-01-16 09:30
def-456-uvw                         manual.pdf                  25     2026-01-16 08:15
```

### Show Tree Structure
```bash
python cli_workflow.py show-tree abc-123-xyz
```

Output:
```
Tree Index: tree-456-def
Nodes: 45
Created: 2026-01-16T09:35:00

Tree Structure:
------------------------------------------------------------
├─ Introduction
   Background and motivation for this research...
  ├─ Motivation
  ├─ Problem Statement
├─ Related Work
  ├─ Machine Learning Approaches
  ├─ Deep Learning Methods
├─ Methodology
  ...
```

### Export Markdown
```bash
# Export to default filename
python cli_workflow.py export abc-123-xyz

# Export to specific file
python cli_workflow.py export abc-123-xyz -o output/document.md
```

## Configuration

### Environment Variables

```bash
# vLLM Server
export VLLM_API_KEY="123"
export VLLM_SERVER_URL="http://localhost:8000/v1"

# Database
export DATABASE_URL="sqlite:///document_store.db"  # Default
# Or use PostgreSQL:
# export DATABASE_URL="postgresql://user:pass@localhost/ocr_docs"
```

### LLM Providers

**OpenAI (default)**
```python
{
  "llm_provider": "openai",
  "model": "gpt-4o-2024-11-20"
}
```

**Ollama**
```python
{
  "llm_provider": "ollama",
  "model": "llama2",
  "ollama_base_url": "http://localhost:11434",
  "ollama_timeout": 300
}
```

## Dependencies

Install with:
```bash
pip install -r requirements_storage.txt
```

Required packages:
- `sqlalchemy>=2.0.0` - Database ORM
- `Pillow>=10.0.0` - Image processing
- `fastapi` - API framework (if using API server)
- `uvicorn` - ASGI server (if using API server)

## Database Management

### View Database Contents
```bash
sqlite3 document_store.db

# List tables
.tables

# Count documents
SELECT COUNT(*) FROM documents;

# Count layout elements
SELECT COUNT(*) FROM layout_elements;

# View recent documents
SELECT id, filename, total_pages, created_at FROM documents ORDER BY created_at DESC LIMIT 5;
```

### Backup Database
```bash
cp document_store.db document_store_backup_$(date +%Y%m%d).db
```

### Reset Database
```bash
# WARNING: This deletes all data!
python init_db.py --drop-existing
```

## Troubleshooting

### vLLM Server Not Running
```
Error: Connection refused
```
**Solution**: Start vLLM server first:
```bash
bash serve_deepseek_ocr.sh
```

### SQLAlchemy Not Found
```
ModuleNotFoundError: No module named 'sqlalchemy'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements_storage.txt
```

### PageIndex Not Found
```
ModuleNotFoundError: No module named 'pageindex'
```
**Solution**: Ensure PageIndex is in the same parent directory:
```
duy_dev/
├── deepseek_ocr/
└── PageIndex/
```

## Performance

| Operation | Duration | Notes |
|-----------|----------|-------|
| Database initialization | <1s | One-time setup |
| OCR per page | 5-10s | Depends on content complexity |
| Tree indexing | 10-30s | Depends on document length |
| Element retrieval | <100ms | Database query |
| Markdown export | <500ms | All pages |

## Next Steps

After processing documents, you can:

1. **Query Layout Elements**: Find all tables, images, formulas
2. **Navigate Tree Structure**: Use hierarchical index for navigation
3. **Summarization**: Use tree structure for efficient summarization
4. **Translation**: Translate while preserving structure
5. **Search**: Build search index on structured content

## Examples

See `workflow_guide.md` for complete Python examples including:
- End-to-end workflow
- Custom queries
- Downstream processing
- Integration patterns
