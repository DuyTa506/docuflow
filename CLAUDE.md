# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the API server
uvicorn serving.workflow_api:app --port 8002 --reload

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

All pipeline steps (extract, translate, summarize, keywords, etc.) are **async background tasks**:

1. Router calls `service.submit(db, document_id, ...)` → returns `task_id` immediately
2. Service calls `task_manager.submit(db, document_id, task_type, coro)` which:
   - Creates a `Task` row in DB (status=PENDING)
   - Launches `asyncio.create_task(coro)`
3. The async coroutine runs, calling `self._progress(task_id, pct, msg)` at each step
4. Client polls `GET /api/v2/tasks/{task_id}` until `status == COMPLETED`
5. Client fetches result via the document-specific GET endpoint

All async pipeline services inherit from `services/base_service.py` (`BaseTaskService`) which provides `_find_task_id()`, `_read_text()`, `_progress()`, and `_extract_json()`.

There is **no external queue** (no Redis/Celery) — all tasks run in-process via asyncio.

### ID generation

Primary keys are **prefixed strings**, not integers or UUIDs. `data/id_generator.py` reads/increments a counter in the `id_sequences` table. Format: `DOC_001`, `USR_042`, `TRANSLATE_007`, etc. When writing tests that create `Document` rows directly, you must supply an explicit `id` (e.g. `id=str(uuid.uuid4())`) because the generator only runs in the service layer.

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
- **OCR** (`api/dependencies.py` → `get_ocr_client()`): raw `AsyncOpenAI` pointed at a vLLM server (`VLLM_SERVER_URL`). Used by `services/ocr_service.py` for vision-based OCR.
- **Pipeline LLM** (`api/dependencies.py` → `get_llm_client()`): cached `LLMClientFactory` instance supporting OpenAI or Ollama (`AI_PROVIDER`). Used by translation, summarization, keywords, research, main_content services.

### Configuration

`config/settings.py` (`Settings` class, Pydantic v2 BaseSettings). All values can be overridden via `.env` file or environment variables. Key groups:
- `VLLM_*` — OCR vLLM server
- `AI_*` — pipeline LLM (OpenAI/Ollama)
- `DATABASE_URL` — defaults to `sqlite:///document_store.db`
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
- `UPLOAD_DIR`, `LIBREOFFICE_PATH`

### File upload override

Users can override auto-generated content by uploading `.txt` or `.docx` files. Shared helper: `utils/file_upload.py` (`extract_text_from_upload()`). Endpoints:
- `POST /api/v2/documents/{id}/text/upload` — overrides `normalized_content`
- `POST /api/v2/documents/{id}/translations/{tid}/upload` — overrides translation, sets status → PENDING_REVIEW
- `POST /api/v2/documents/{id}/summaries/{sid}/upload` — overrides summary content

### Digest

`POST /api/v2/documents/{id}/digest` is a **read-only assembly** endpoint — it does NOT trigger any pipeline. It collects existing DB rows (summary + main_content + keywords + research_directions) into a single response. Missing sections appear in the `missing` array. `GET /api/v2/documents/{id}/digest/download` returns the same data as a formatted `.docx` file.
