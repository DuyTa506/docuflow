# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the API server
uvicorn serving.workflow_api:app --port 8002 --reload

# Start the vLLM OCR server (DeepSeek-OCR-2)
bash serve_deepseek_ocr.sh

# Initialize database (creates tables + default admin: admin/admin)
python scripts/init_db.py
python scripts/init_db.py --drop-existing   # wipe and recreate

# Run all tests
pytest

# Run tests without coverage (faster, avoids coverage fail-under)
pytest --override-ini="addopts=" -q

# Run a single test
pytest tests/unit/test_data/test_db_models.py::TestDocument::test_create_document

# Run by marker
pytest -m unit

# Format
black .
isort .
```

## Architecture

### Request lifecycle

```
HTTP request
  → serving/routers/{name}_router.py   (FastAPI route handler)
  → services/{name}_service.py         (business logic, validation)
  → data/repositories/{name}_repo.py   (DB queries, no logic)
  → data/db_models.py                  (SQLAlchemy ORM models)
```

### Background task pattern

All pipeline steps (extract, translate, summarize, keywords, etc.) are async background tasks:

1. Router calls `service.submit(db, document_id, ...)` → returns `task_id` immediately
2. Service calls `task_manager.submit(db, document_id, task_type, coro)` which:
   - Creates a `Task` row in DB (status=PENDING)
   - Launches `asyncio.create_task(coro)`
3. The async coroutine runs, calling `self._progress(task_id, pct, msg)` at each step
4. Client polls `GET /api/v2/tasks/{task_id}` until `status == COMPLETED`
5. Client fetches result via the document-specific GET endpoint

All async pipeline services inherit from `services/base_service.py` (`BaseTaskService`) which provides `_find_task_id()`, `_read_text()`, `_progress()`, and `_extract_json()`.

There is no external queue (no Redis/Celery) — all tasks run in-process via asyncio.

### ID generation

Primary keys are prefixed strings, not integers or UUIDs. `data/id_generator.py` reads/increments a counter in the `id_sequences` table. Format: `DOC_001`, `USR_042`, `TRANSLATE_007`, etc. When writing tests that create `Document` rows directly, you must supply an explicit `id` (e.g. `id=str(uuid.uuid4())`) because the generator only runs in the service layer.

### Key models

| Model | Table | Notes |
|---|---|---|
| `User` | `users` | `group` (TEACHER/LIBRARY) + `role` (MEMBER/ADMIN) + `status` |
| `Document` | `documents` | `processing_status`: INIT → EXTRACT_IN_PROGRESS → EXTRACTED / FAILED |
| `DigitizedText` | `digitized_texts` | `ocr_content` (raw) + `normalized_content` (cleaned, used by all downstream) |
| `Translation` | `translations` | `status`: PENDING → IN_PROGRESS → COMPLETED → PENDING_REVIEW → APPROVED |
| `Task` | `tasks` | `status`: PENDING → RUNNING → COMPLETED / FAILED; `progress` 0–100 |

### Auth

JWT bearer tokens. Payload: `{"sub": user_id, "role": "ADMIN"|"MEMBER", "group": "TEACHER"|"LIBRARY"}`. Dependency injected via `get_current_user()` and `require_role(*roles)` in `api/dependencies.py`. Accounts must have `status == ACTIVE` to log in. TEACHER group self-registers as ACTIVE; LIBRARY group starts as PENDING_APPROVAL.

### LLM clients

Two separate LLM usages:
- **OCR** (`api/dependencies.py` → `get_ocr_client()`): raw `AsyncOpenAI` pointed at a vLLM server (`VLLM_SERVER_URL`). Used by `services/ocr_service.py` for vision-based OCR. The vLLM server must be started with `--logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor` for anti-hallucination to work (see `serve_deepseek_ocr.sh`).
- **Pipeline LLM** (`api/dependencies.py` → `get_llm_client()`): cached `LLMClientFactory` instance supporting OpenAI or Ollama (`AI_PROVIDER`). Used by translation, summarization, keywords, research, main_content services.

### OCR anti-hallucination

`serving/logic.py` activates `NGramPerReqLogitsProcessor` per request via `extra_body.logits_processors`. Parameters: `ngram_size=20`, `window_size=50`, `whitelist_token_ids=[128821, 128822]` (`<td>`/`</td>` tokens kept unbanned for table rendering). The server-side `--logits-processors` flag is required to trust the processor; per-request activation is required to apply it. A `_is_degenerate()` guard in `logic.py` catches any loops that slip through and returns an error event instead of passing bad output downstream.

Images are encoded as JPEG and sent with `data:image/jpeg` MIME type. `max_image_size` is 1344px (capped from the original 2048) to keep A4 pages in a 2-tile layout and reduce hallucination risk.

### Configuration

`config/settings.py` (`Settings` class, Pydantic v2 BaseSettings). All values can be overridden via `.env` file or environment variables. Key groups:
- `VLLM_*` — OCR vLLM server (`VLLM_API_KEY`, `VLLM_SERVER_URL`, `VLLM_MODEL`)
- `OCR_*` — OCR parameters (`OCR_MAX_TOKENS`, `OCR_TEMPERATURE`, `OCR_TARGET_DPI`, `OCR_MAX_IMAGE_SIZE`, `OCR_PROMPT`)
- `AI_*` — pipeline LLM (`AI_PROVIDER`, `AI_MODEL`, `AI_OLLAMA_BASE_URL`, `AI_OPENAI_BASE_URL`)
- `PAGEINDEX_*` — tree indexing LLM and parameters
- `DATABASE_URL` — defaults to `sqlite:///document_store.db`
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
- `UPLOAD_DIR`, `LIBREOFFICE_PATH`, `PDF_TEXT_THRESHOLD`

### File upload override

Users can override auto-generated content by uploading `.txt` or `.docx` files. Shared helper: `utils/file_upload.py` (`extract_text_from_upload()`). Endpoints:
- `POST /api/v2/documents/{id}/text/upload` — overrides `normalized_content`
- `POST /api/v2/documents/{id}/translations/{tid}/upload` — overrides translation, sets status → PENDING_REVIEW
- `POST /api/v2/documents/{id}/summaries/{sid}/upload` — overrides summary content

### Digest

`POST /api/v2/documents/{id}/digest` is a read-only assembly endpoint — it does NOT trigger any pipeline. It collects existing DB rows (summary + main_content + keywords + research_directions) into a single response. Missing sections appear in the `missing` array. `GET /api/v2/documents/{id}/digest/download` returns the same data as a formatted `.docx` file.

### Document extraction

`services/extractors/` contains four extractor implementations selected automatically by file type:
- `pdf_text_extractor.py` — native text extraction for text-layer PDFs (threshold: `PDF_TEXT_THRESHOLD` chars)
- `ocr_extractor.py` — falls back to OCR pipeline for scanned/image PDFs
- `docx_extractor.py` — DOCX extraction via python-docx
- `doc_converter.py` — converts `.doc` to `.docx` via LibreOffice before extraction

### Spatial analysis and PageIndex

`core/spatial/` implements a layout-aware hierarchy pipeline used during tree indexing:
- `filters.py` — strips headers/footers from layout elements
- `zone_classifier.py` — labels zones (title_block, abstract, section_heading, main_text, table, figure, etc.)
- `hierarchy.py` — scores elements for hierarchy level using whitespace, size, label, indent
- `reading_order.py` — topological sort for correct column/multi-column reading order
- `grouping.py` — detects columns, groups lines into blocks

`core/pageindex/` wraps an LLM-based hierarchical indexer that builds a tree from markdown content. It is used by `serving/tree_indexing_service.py`. The LLM client is configured via `PAGEINDEX_LLM_PROVIDER` (openai or ollama) and `PAGEINDEX_MODEL`.

### Search

`services/search_service.py` + `serving/routers/search_router.py`. Endpoint: `GET /api/v2/search?q=...`. Searches across document titles and normalized OCR content.

### Storage

`services/storage_service.py` handles file I/O for uploaded documents. Files are stored under `UPLOAD_DIR` (default `./uploads`). LibreOffice (`LIBREOFFICE_PATH`) is used for `.doc` → `.docx` conversion via `utils/soffice.py`.
