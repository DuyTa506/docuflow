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
deepseek_ocr/
├── core/          # Domain models
├── data/          # Database layer
├── utils/         # Utilities
├── services/      # OCR service
├── spatial/       # Spatial analysis
├── pageindex/     # PageIndex library
├── serving/       # API & workflow
├── config/        # Settings
└── tests/         # Test suite
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
