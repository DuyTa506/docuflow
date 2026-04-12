# DocuFlow

A document processing and library management system. Accepts PDF, DOCX, DOC, and image files, extracts text via OCR or native parsing, and runs an AI pipeline (translation, summarization, keyword extraction, research directions, main content) through a REST API.

## Features

- DeepSeek OCR-2 via vLLM for vision-based text extraction with grounding metadata
- Spatial layout analysis for hierarchy and reading order detection
- PageIndex hierarchical tree indexing via LLM (OpenAI or Ollama)
- Full pipeline: extract, normalize, translate, summarize, keywords, main content, research directions
- Digest assembly endpoint that collects all pipeline outputs into a single structured response with .docx download
- Async task system with progress polling — no external queue
- JWT authentication with group (TEACHER/LIBRARY) and role (MEMBER/ADMIN) access control
- SQLite database with prefixed ID generation (DOC_001, USR_042, etc.)
- File upload override: users can replace auto-generated content by uploading .txt or .docx files

## Requirements

- Python 3.10+
- CUDA GPU (for vLLM / DeepSeek OCR-2)
- LibreOffice (`soffice`) for .doc conversion (optional)

## Setup

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install vLLM (GPU required)
uv pip install vllm --torch-backend=auto

# Install remaining dependencies
uv pip install -r requirements.txt
```

### 2. Configure

Copy or create a `.env` file:

```bash
# vLLM OCR server
VLLM_API_KEY=123
VLLM_SERVER_URL=http://localhost:8000/v1
VLLM_MODEL=deepseek-ai/DeepSeek-OCR-2

# Pipeline LLM (translation, summarization, etc.)
AI_PROVIDER=openai          # or ollama
AI_MODEL=gpt-4o-2024-11-20
AI_OPENAI_BASE_URL=         # optional, for OpenAI-compatible endpoints
AI_OLLAMA_BASE_URL=http://localhost:11434

# PageIndex tree building
PAGEINDEX_LLM_PROVIDER=openai   # or ollama
PAGEINDEX_MODEL=gpt-4o-2024-11-20

# Auth
JWT_SECRET_KEY=change-me-in-production

# Database
DATABASE_URL=sqlite:///document_store.db

# Storage
UPLOAD_DIR=./uploads
LIBREOFFICE_PATH=soffice
```

### 3. Initialize the database

```bash
python scripts/init_db.py
# Creates tables and a default admin account: admin / admin
```

### 4. Start the vLLM OCR server

```bash
bash serve_deepseek_ocr.sh
```

The script starts DeepSeek-OCR-2 on port 8000 with the `NGramPerReqLogitsProcessor` anti-hallucination processor registered server-side.

### 5. Start the API server

```bash
uvicorn serving.workflow_api:app --port 8002 --reload
```

API is available at `http://localhost:8002`. Interactive docs at `http://localhost:8002/docs`.

## API Overview

All endpoints are under `/api/v2/`.

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/auth/login` | Obtain JWT token |
| POST | `/api/v2/auth/register` | Register new account |
| GET | `/api/v2/auth/me` | Current user info |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents` | Upload a document (PDF, DOCX, DOC, image) |
| GET | `/api/v2/documents` | List documents |
| GET | `/api/v2/documents/{id}` | Get document metadata |
| DELETE | `/api/v2/documents/{id}` | Delete document |
| POST | `/api/v2/documents/{id}/extract` | Run extraction (OCR or native) |
| POST | `/api/v2/documents/{id}/text/upload` | Override normalized text |

### Pipeline (all async — returns task_id, poll /tasks/{task_id})

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents/{id}/translations` | Start translation |
| GET | `/api/v2/documents/{id}/translations` | Get translations |
| POST | `/api/v2/documents/{id}/translations/{tid}/upload` | Override translation text |
| POST | `/api/v2/documents/{id}/summaries` | Start summarization |
| GET | `/api/v2/documents/{id}/summaries` | Get summaries |
| POST | `/api/v2/documents/{id}/summaries/{sid}/upload` | Override summary text |
| POST | `/api/v2/documents/{id}/keywords` | Extract keywords |
| GET | `/api/v2/documents/{id}/keywords` | Get keywords |
| POST | `/api/v2/documents/{id}/main-content` | Extract main content |
| GET | `/api/v2/documents/{id}/main-content` | Get main content |
| POST | `/api/v2/documents/{id}/research-directions` | Extract research directions |
| GET | `/api/v2/documents/{id}/research-directions` | Get research directions |

### Digest

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents/{id}/digest` | Assemble all pipeline outputs (read-only) |
| GET | `/api/v2/documents/{id}/digest/download` | Download digest as .docx |

### Tasks and Search

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v2/tasks/{task_id}` | Poll task status and progress |
| GET | `/api/v2/search?q=...` | Full-text search across documents |

## Project Structure

```
docuflow/
├── serving/
│   ├── workflow_api.py          # FastAPI app, router registration, startup
│   ├── logic.py                 # OCR request logic, anti-hallucination guard
│   ├── tree_indexing_service.py # PageIndex tree building orchestration
│   └── routers/                 # One router per resource
├── services/
│   ├── base_service.py          # BaseTaskService (shared async helpers)
│   ├── task_manager.py          # Creates Task rows, launches asyncio tasks
│   ├── document_service.py      # Upload and extraction orchestration
│   ├── storage_service.py       # File I/O
│   ├── ocr_service.py           # OCR pipeline
│   ├── translation_service.py
│   ├── summarization_service.py
│   ├── keyword_service.py
│   ├── main_content_service.py
│   ├── research_direction_service.py
│   ├── digest_service.py        # Read-only digest assembly
│   ├── digest_renderer.py       # .docx rendering
│   ├── search_service.py
│   ├── auth_service.py
│   └── extractors/              # PDF (text + OCR), DOCX, DOC extractors
├── data/
│   ├── db_models.py             # SQLAlchemy ORM models
│   ├── database.py              # Session factory
│   ├── id_generator.py          # Prefixed ID generation
│   └── repositories/            # One repo per model (DB queries only)
├── core/
│   ├── constants.py             # OCR params, spatial weights, label hierarchy
│   ├── models.py                # Core dataclasses (ServicePageResult, etc.)
│   ├── spatial/                 # Layout analysis pipeline
│   └── pageindex/               # LLM-based hierarchical indexer
├── api/
│   ├── dependencies.py          # FastAPI dependency injection (auth, LLM clients)
│   └── schemas.py               # Pydantic request/response schemas
├── utils/
│   ├── image_utils.py           # PDF rendering, image resize, base64
│   ├── bbox_utils.py            # Grounding tag parsing, bounding boxes
│   ├── text_utils.py            # clean_grounding_format and other text helpers
│   ├── file_upload.py           # Extract text from uploaded .txt/.docx
│   └── soffice.py               # LibreOffice subprocess wrapper
├── config/
│   ├── settings.py              # Pydantic BaseSettings (all env vars)
│   └── spatial_config.py        # Spatial analysis tuning
├── scripts/
│   └── init_db.py               # DB init + default admin seed
├── tests/                       # pytest test suite
├── serve_deepseek_ocr.sh        # vLLM server startup script
└── CLAUDE.md                    # Developer guide for Claude Code
```

## OCR Anti-Hallucination

DeepSeek OCR-2 can enter repetition loops on dense documents. Three mechanisms are active:

1. **NGramPerReqLogitsProcessor** — hard-bans n-grams (size 20) seen in the last 50 tokens from being generated again. Registered server-side via `--logits-processors` in `serve_deepseek_ocr.sh`; activated per request via `extra_body.logits_processors` in `serving/logic.py`.
2. **Image size cap** — `max_image_size=1344` in `core/constants.py` keeps A4 pages to a 2-tile vLLM layout, reducing tiling-induced hallucination.
3. **Degenerate output guard** — `_is_degenerate()` in `serving/logic.py` scans the last 300 chars of model output for numbered-list loops, excessive blank lines, or repeated phrases, and returns an error event rather than passing corrupt text to downstream services.

## License

MIT
