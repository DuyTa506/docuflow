# DocuFlow

A unified document processing system combining **DeepSeek OCR** with **PageIndex** for intelligent document indexing and translation.

## Features

- 🔍 **DeepSeek OCR** - High-quality OCR with spatial metadata
- 🌲 **PageIndex** - Hierarchical document structure extraction
- 📊 **Spatial Analysis** - Layout-aware hierarchy detection
- 🌐 **Translation** - Document translation (EN→VI)
- 💾 **Storage** - SQLite database with SQLAlchemy
- 🚀 **API** - FastAPI REST endpoints
- ⌨️ **CLI** - Command-line interface

## Quick Start

### 1. Install with uv

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install vLLM (GPU required)
uv pip install vllm --torch-backend=auto

# Install other dependencies
uv pip install -r requirements.txt
```

### 2. Start vLLM Server

```bash
# DeepSeek OCR model
vllm serve ds-math-ocr --port 8000
```

### 3. Configure

```bash
# Create .env file
cp .env.example .env

# Edit with your settings
# VLLM_API_KEY=123
# VLLM_SERVER_URL=http://localhost:8000/v1
# OPENAI_API_KEY=sk-... (for OpenAI tree building)
```

### 4. Initialize Database

```bash
python init_db.py
```

### 5. Setup LLM for Tree Indexing

**Option A: Ollama (Local, Free)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3:30b  # or llama3, mistral, etc.

# Ollama starts automatically
```

**Option B: OpenAI API**
```bash
# Add to .env
PAGEINDEX_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### 6. Process Documents

```bash
# Process PDF with OCR
python cli_workflow.py process document.pdf

# Build tree index (with Ollama)
python cli_workflow.py build-tree <doc_id> --use-spatial

# Show tree structure
python cli_workflow.py show-tree <doc_id>

# Translate to Vietnamese
python quick_translate.py <doc_id> output_vi.md
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `process <file>` | OCR process PDF/image |
| `build-tree <id>` | Build tree index |
| `show-tree <id>` | Display tree structure |
| `list` | List all documents |
| `export <id>` | Export markdown |

**Options:**
- `--build-index` - Build tree after OCR
- `--use-spatial` - Use spatial metadata for tree
- `--llm-provider` - `openai` or `ollama`
- `--model` - Model name

## API

```bash
# Start server
uvicorn serving.workflow_api:app --port 8002

# Process document
curl -X POST http://localhost:8002/process-document -F "file=@doc.pdf"

# Build index
curl -X POST http://localhost:8002/build-index/{doc_id}
```

## Project Structure

```
docuflow/
├── core/          # Domain models
├── data/          # Database layer
├── utils/         # Utilities (OCR parsing, bbox, text)
├── services/      # OCR service
├── spatial/       # Spatial analysis (NEW: spatial-first pipeline)
│   ├── filters.py              # Preprocessing (header/footer removal)
│   ├── zone_classifier.py     # Zone classification (title, caption, etc.)
│   ├── reading_order.py       # Topological sorting for reading order
│   ├── grouping.py            # Column detection, line/block grouping
│   ├── hierarchy.py           # Hierarchy prediction with whitespace scoring
│   └── spatial_tree_builder.py # NEW: Spatial-first tree builder
├── pageindex/     # PageIndex library
├── serving/       # API & workflow
├── config/        # Settings
└── tests/         # Test suite
```

## Spatial-First Pipeline (NEW)

### Key Concepts

**Layout Elements** giờ được enrich với full text từ raw OCR:
```python
from utils import extract_layout_coordinates_v2

# Parse with full text extraction
elements = extract_layout_coordinates_v2(
    raw_ocr_output,
    img_width,
    img_height,
    page_number=1
)
# → [{'label': 'title', 'bbox': {...}, 'text_content': 'Introduction', 'text_full': '...'}]
```

**Build Tree** từ spatial elements (không cần markdown làm primary source):
```python
from spatial import build_spatial_tree

tree = build_spatial_tree(
    layout_elements=elements,
    use_filters=True,              # Remove header/footer
    use_zone_classification=True,  # Classify zones
    use_reading_order=True,        # Topological sort
    use_markdown_validation=True,  # Optional: cross-check with # syntax
    use_adaptive_thresholds=True   # Per-document calibration
)
```

### Migration from Old API

```python
# Old (deprecated)
# from spatial import build_enhanced_tree_v2
# tree = build_enhanced_tree_v2(markdown, layout_elements)

# New (recommended)
from spatial import build_spatial_tree
tree = build_spatial_tree(layout_elements)  # Simpler!
```

## Configuration

**Environment Variables:**

```bash
# vLLM (required for OCR)
VLLM_API_KEY=123
VLLM_SERVER_URL=http://localhost:8000/v1

# PageIndex (for tree building)
PAGEINDEX_LLM_PROVIDER=ollama  # or openai
PAGEINDEX_MODEL=qwen3:30b
```

## Requirements

- Python 3.10+
- CUDA GPU (for vLLM)
- vLLM installed via uv
- Ollama (optional, for local LLM)

## License

MIT
