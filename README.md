# DocuFlow

A document processing and library management system. Accepts PDF, DOCX, DOC, and image files, extracts text via OCR or native parsing, and runs an AI pipeline (translation, summarization, keyword extraction, research directions, main content) through a REST API.

## Features

- DeepSeek OCR-2 via vLLM for vision-based text extraction with grounding metadata
- Spatial layout analysis for hierarchy and reading order detection
- PageIndex hierarchical tree indexing via LLM (OpenAI or Ollama)
- Full pipeline: extract, normalize, translate, summarize, keywords, main content, research directions
- Vietnamese-first output for summaries and research directions (configurable via `SUMMARY_OUTPUT_LANG` / `RESEARCH_OUTPUT_LANG`)
- Context-adaptive chunking: chunk sizes derived automatically from the model's context window (`AI_MODEL_CONTEXT_WINDOW * AI_CHUNK_RATIO`)
- Digest assembly endpoint that collects all pipeline outputs into a single structured response with .docx download
- Async task system: each pipeline run creates a job record (`PENDING/IN_PROGRESS/COMPLETED/FAILED`) plus a `Task` row for progress polling. No external queue.
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

# Chunking and output language
AI_MODEL_CONTEXT_WINDOW=128000   # model's token context window
AI_CHUNK_RATIO=0.85              # fraction of context per chunk
SUMMARY_OUTPUT_LANG=vi           # BCP-47 code for summary output
RESEARCH_OUTPUT_LANG=vi          # BCP-47 code for research direction output

# Storage
UPLOAD_DIR=./uploads
LIBREOFFICE_PATH=soffice
```

### 3. Initialize the database

```bash
python scripts/init_db.py
# Creates tables and a default admin account: admin / admin
```

### 4. Start services

**All-in-one (recommended):**
```bash
bash start.sh
```

This starts the LLM pipeline container, the vLLM OCR server (background), waits for dependencies to be ready, then launches the API server on port 8002.

**Or start individually:**

Start the vLLM OCR server:

```bash
bash serve_deepseek_ocr.sh
```

The script starts DeepSeek-OCR-2 on port 8000 with the `NGramPerReqLogitsProcessor` anti-hallucination processor registered server-side.

Start the LLM pipeline container (if using local llama.cpp):
```bash
docker compose -f SETUPS/llms/docker-compose.yml up -d
```

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
| PATCH | `/api/v2/auth/me` | Update profile (`full_name`, `email`) |
| PUT | `/api/v2/auth/me/password` | Change password (requires current password) |
| POST | `/api/v2/auth/approve/{user_id}` | Approve pending user (ADMIN) |
| GET | `/api/v2/auth/users` | List all users (ADMIN) |
| GET | `/api/v2/auth/users?q=...` | Search users by username — partial, case-insensitive (ADMIN) |
| DELETE | `/api/v2/auth/users/{user_id}` | Delete user and their documents (ADMIN; cannot delete self or last admin) |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents` | Upload a document (PDF, DOCX, DOC, image) |
| GET | `/api/v2/documents?page=1&limit=50` | List documents (paginated; each item includes `task_summary`) |
| GET | `/api/v2/documents/{id}` | Get document metadata |
| DELETE | `/api/v2/documents/{id}` | Delete document |
| POST | `/api/v2/documents/{id}/extract` | Run extraction (OCR or native) |
| GET | `/api/v2/documents/{id}/text` | Get OCR and normalized text |
| GET | `/api/v2/documents/{id}/text/download?type=ocr|normalized` | Download text as .docx |
| POST | `/api/v2/documents/{id}/text/upload` | Override normalized text |
| POST | `/api/v2/documents/{id}/tree-index` | Build hierarchical tree index |
| GET | `/api/v2/documents/{id}/tree-index` | Get tree index |

### Pipeline

Every `POST` endpoint below creates a job record up-front (status `PENDING`) and returns a `task_id` plus a `resource_id` (the id of that job row). Clients can either:
- poll `GET /api/v2/tasks/{task_id}` for progress (0–100 + message), or
- poll the resource list/detail endpoint and read the `status` field.

Unified job-status enum: `PENDING → IN_PROGRESS → COMPLETED | FAILED`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents/{id}/translations` | Start translation — body: `{ "target_language": "zh", "domain": "general" }` (returns `resource_id` = translation_id) |
| GET | `/api/v2/documents/{id}/translations` | List translations (incl. status) |
| GET | `/api/v2/documents/{id}/translations/{tid}` | Get one translation (incl. status, content) |
| GET | `/api/v2/documents/{id}/translations/{tid}/download` | Download translation as .docx |
| POST | `/api/v2/documents/{id}/translations/{tid}/upload` | Override translation text |
| POST | `/api/v2/documents/{id}/summaries` | Start summarization |
| GET | `/api/v2/documents/{id}/summaries` | List summaries (incl. status) |
| GET | `/api/v2/documents/{id}/summaries/{sid}` | Get one summary |
| GET | `/api/v2/documents/{id}/summaries/{sid}/download` | Download summary as .docx |
| POST | `/api/v2/documents/{id}/summaries/{sid}/upload` | Override summary text |
| POST | `/api/v2/documents/{id}/main-content` | Extract main content |
| GET | `/api/v2/documents/{id}/main-content` | Get latest main content (incl. status) |
| GET | `/api/v2/documents/{id}/main-content/list` | List all main-content jobs |
| GET | `/api/v2/documents/{id}/main-content/{mid}` | Get one main-content job |
| POST | `/api/v2/documents/{id}/keywords` | Extract keywords |
| GET | `/api/v2/documents/{id}/keywords` | Get current keywords + `latest_extraction` status |
| GET | `/api/v2/documents/{id}/keywords/extractions` | List keyword-extraction jobs |
| GET | `/api/v2/documents/{id}/keywords/extractions/{eid}` | Get one keyword-extraction job |
| POST | `/api/v2/documents/{id}/research-directions` | Extract research directions |
| GET | `/api/v2/documents/{id}/research-directions` | Get current directions + `latest_extraction` status |
| GET | `/api/v2/documents/{id}/research-directions/extractions` | List research-extraction jobs |
| GET | `/api/v2/documents/{id}/research-directions/extractions/{eid}` | Get one research-extraction job |

### Digest

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v2/documents/{id}/digest` | Assemble all pipeline outputs (read-only) |
| GET | `/api/v2/documents/{id}/digest/download` | Download digest as .docx |

### Tasks and Search

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v2/tasks/{task_id}` | Poll task status and progress |
| GET | `/api/v2/tasks?document_id=...` | List tasks for a document |
| GET | `/api/v2/search?q=...` | Full-text search across the library (JWT required) |

**Search query parameters:** `q` (required), `search_in` (optional: `title,content,keywords,translations`), `language` (filter translation hits), `page`, `limit`.

**Search response** uses the same envelope as `GET /api/v2/documents`:

```json
{
  "items": [
    {
      "id": "DOC_001",
      "title": "...",
      "format": "pdf",
      "total_pages": 12,
      "processing_status": "EXTRACTED",
      "source_language": "en",
      "created_at": "...",
      "task_summary": { "EXTRACT": "COMPLETED" },
      "snippet": "...matched excerpt...",
      "match_field": "content"
    }
  ],
  "total": 4,
  "page": 1,
  "limit": 20,
  "total_pages": 1,
  "query": "machine learning"
}
```

Each `items[]` row is a `DocumentListItem`. Search adds `snippet` and `match_field`; list endpoints leave those fields `null`.

### Translation languages

`target_language` accepts BCP-47 codes (`en`, `zh`, `ru`, `vi`, …) and common aliases (`zh-CN`, `ru-RU`, `english`, …). Codes are normalized server-side and mapped to full language names in LLM prompts (e.g. `en` → English, `zh` → Chinese). Source language comes from the document's `source_language` at upload time. Target must differ from source.

## Job Status Flow

When you `POST` to start a pipeline step:
1. The server creates a job row (Translation / Summary / MainContent / KeywordExtraction / ResearchExtraction) with `status="PENDING"`.
2. A `Task` row is also created (used for progress %).
3. The endpoint immediately returns `{ task_id, resource_id, status: "PENDING" }`.
4. The background worker:
   - Updates the job row → `IN_PROGRESS`
   - On success → writes content + sets `COMPLETED`
   - On exception → sets `FAILED` (and on Keywords/Research jobs the `error` column is filled too)

**Frontend strategy**
- Right after POST, the resource record exists in `GET /…/translations` (etc.) so you can immediately render a row in `PENDING` state.
- Poll either `GET /api/v2/tasks/{task_id}` (for progress %) or the resource list/detail endpoint (for `status`).
- When `status == COMPLETED`, GET the detail endpoint to read the content.

> **Note on `Document.processing_status`**
>
> The document’s `processing_status` field tracks **only** the OCR / extraction phase (`INIT → EXTRACT_IN_PROGRESS → EXTRACTED → FAILED`). It does **not** change when translate / summarize / keywords / etc. run — each of those has its own job row with its own status.


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
│   ├── file_download.py         # .docx response builder (build_docx_response, safe_filename)
│   ├── file_upload.py           # Extract text from uploaded .txt/.docx
│   └── soffice.py               # LibreOffice subprocess wrapper
├── config/
│   ├── settings.py              # Pydantic BaseSettings (all env vars)
│   └── spatial_config.py        # Spatial analysis tuning
├── scripts/
│   └── init_db.py               # DB init + default admin seed
├── tests/                       # pytest test suite
├── serve_deepseek_ocr.sh        # vLLM server startup script
├── start.sh                     # All-in-one startup (OCR + LLM + API)
└── CLAUDE.md                    # Developer guide for Claude Code
```

## OCR Anti-Hallucination

DeepSeek OCR-2 can enter repetition loops on dense documents. Three mechanisms are active:

1. **NGramPerReqLogitsProcessor** — hard-bans n-grams (size 20) seen in the last 50 tokens from being generated again. Registered server-side via `--logits-processors` in `serve_deepseek_ocr.sh`; activated per request via `extra_body.logits_processors` in `serving/logic.py`.
2. **Image size cap** — `max_image_size=1344` in `core/constants.py` keeps A4 pages to a 2-tile vLLM layout, reducing tiling-induced hallucination.
3. **Degenerate output guard** — `_is_degenerate()` in `serving/logic.py` scans the last 300 chars of model output for numbered-list loops, excessive blank lines, or repeated phrases, and returns an error event rather than passing corrupt text to downstream services.

## License

MIT
