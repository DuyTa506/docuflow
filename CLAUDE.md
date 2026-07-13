# CLAUDE.md

Agent guide for this repo. For deep implementation detail, read the source files referenced below.

## Commands

```bash
bash start.sh                                    # OCR + LLM container + API
uvicorn serving.workflow_api:app --port 8022 --reload
bash serve_deepseek_ocr.sh                       # vLLM OCR (DeepSeek-OCR-2)
docker compose -f SETUPS/llms/docker-compose.yml up -d
python scripts/init_db.py                        # admin/admin; --drop-existing to wipe
pytest                                           # full suite
pytest --override-ini="addopts=" -q              # fast, no coverage gate
pytest tests/unit/test_data/test_db_models.py::TestDocument::test_create_document
black . && isort .
```

## Architecture

**Request flow:** `serving/routers/{name}_router.py` → `services/{name}_service.py` → `data/repositories/{name}_repo.py` → `data/db_models.py`

**Background tasks:**
- **Single-stage reruns (keywords/summary/…):** in-process asyncio via `task_manager` (no Redis/Celery). Startup sweeps orphaned PENDING/RUNNING rows to FAILED.
- **OCR/extraction:** `ExtractionWorkflow` on queue `docuflow-extraction` (default; `OCR_USE_TEMPORAL=false` for legacy path). Pages persist as they complete; a retry passes `resume=True` and skips stored pages.
- **Translate:** `TranslationWorkflow` on queue `docuflow-translation` (default; `TRANSLATION_USE_TEMPORAL=false` for legacy) — crash/retry resumes via the MinIO unit cache (`services/translators/_cache.py`, keyed sha1(kind, target_lang, text) under `documents/{doc}/translations/{tid}/cache/`). Cancel: `DELETE .../translations/{tid}`.
- **Full digest (Tổng thuật):** Temporal workflow `DigestPipelineWorkflow` on queue `docuflow-digest`. Non-critical stage failures degrade to quality-report warnings (`CRITICAL_STAGES` in `services/pipeline/constants.py`); summarize clusters leaf nodes above `SUMMARIZE_CLUSTER_THRESHOLD` and checkpoints node summaries for resume.
  1. `POST /api/v2/documents/{id}/analysis` → starts workflow + parent `DIGEST_PIPELINE` task
  2. Worker: `bash scripts/start_temporal_worker.sh` (separate from uvicorn)
  3. Poll `GET /api/v2/documents/{id}/pipeline-status` (or parent task via `/tasks/{id}`)
  4. DAG: `BUILD_TREE` → parallel(biblio, keywords, research, usage) → parallel(summarize, main_content) → finalize

```bash
docker compose up -d                    # postgres, minio, temporal, temporal-ui (:8088)
bash scripts/start_temporal_worker.sh   # digest pipeline worker
python scripts/migrate_pipeline_fields.py
```

Pipeline services inherit `services/base_service.py` (`BaseTaskService`: `_find_task_id`, `_read_text`, `_progress`, `_extract_json`).

**IDs:** prefixed strings via `data/id_generator.py` (`DOC_001`, `USR_042`, …). Tests creating rows directly must set explicit `id` — generator runs in service layer only.

**Key models:** `User` (group TEACHER/LIBRARY, role MEMBER/ADMIN, status) · `Document` (processing_status INIT→EXTRACTED/FAILED) · `DigitizedText` (`ocr_content` raw, `normalized_content` for downstream) · `Translation` (PENDING→APPROVED) · `Task` (PENDING→RUNNING→COMPLETED/FAILED, progress 0–100)

## Auth

JWT bearer (`api/dependencies.py`: `get_current_user`, `require_role`, `get_authorized_document`). Payload: `sub`, `role`, `group`. ACTIVE required to login; LIBRARY starts PENDING_APPROVAL. Document endpoints: owner or ADMIN. Routes/schemas: `serving/routers/auth_router.py`, `api/schemas.py`, `services/auth_service.py`.

## LLM & OCR

| Use | Client | Config |
|-----|--------|--------|
| OCR | `get_ocr_client()` → vLLM AsyncOpenAI | `VLLM_*`, `OCR_*` |
| Pipeline | `get_llm_client()` → LLMClientFactory | `AI_*`, `AI_PROVIDER` |

OCR: `services/extractors/ocr_extractor.py` + `serving/logic.py`. vLLM must start with `--logits-processors ...NGramPerReqLogitsProcessor` (`serve_deepseek_ocr.sh`); per-request activation + `_is_degenerate()` guard. JPEG, max 1344px.

**Critical:** keep `OCR_PROMPT` short (`<image>\n<|grounding|>Convert the document to markdown.`). Do NOT add LaTeX/formatting instructions — breaks DeepSeek grounding. Math export uses layout labels → `$$...$$` → Pandoc OMML (`utils/math_omml.py`).

## Config

All in `config/settings.py` (Pydantic BaseSettings, `.env` override). Key groups: `VLLM_*`, `OCR_*`, `AI_*` (`ai_chunk_tokens` = context × chunk ratio), `PAGEINDEX_*`, `DATABASE_URL`, `MINIO_*`, `JWT_*`, `UPLOAD_DIR`. Pipeline output language: Vietnamese via `pipeline_output_lang_clause()`; keywords use source language; translation uses `lang_name()`.

## Extraction

Hybrid in `services/extractors/`:
- Text-layer PDF → `DoclingLayoutExtractor` (Docling)
- Scoring → `OcrExtractor` (DeepSeek vLLM)
- DOCX/DOC → `docx_extractor.py` / LibreOffice

Page routing: `classify_pages()` in `docling_pdf_extractor.py` (`PDF_TEXT_THRESHOLD`).

## Translation routing

See `services/translation_service.py`:

| Input | Mode | Output |
|-------|------|--------|
| DOCX/DOC | `docx_inplace` | Translated DOCX |
| PDF text-layer + overlay enabled | `pdf_overlay` | Translated PDF (`core/pdf_overlay/`) |
| PDF/scan, >500 elements | `block_based` | Spatial DOCX / layout PDF |
| PDF/scan, ≤500 elements | `element_based` | Spatial DOCX / layout PDF |
| No elements, has tree | `tree` | Flat DOCX |
| Fallback | `flat` | Chunked flat DOCX |

Block merge: `utils/translation_blocks.py` + `core/spatial/grouping.py`, parallel via `TRANSLATION_PARALLELISM`.

## Export & download

Helpers: `utils/file_download.py`, `utils/markdown_docx.py` (spatial DOCX), `utils/layout_pdf.py` (layout PDF), `utils/markdown_pandoc.py` (OMML).

- Text: `GET .../text/download?type=ocr|normalized&format=docx|pdf&mode=...`
- Translation: `GET .../translations/{tid}/download?format=docx|pdf`
- `DOCX_EXPORT_ENGINE=auto|pandoc|python`; spatial rebuilds from `layout_elements` bbox
- Layout PDF: OCR=`text_overlay=skip`, translation=`text_overlay=replace` (white mask + redraw)
- Text upload override (`text_overridden=True`) skips spatial export → `utils/file_upload.py`

Renderer details (tables, images, inline math): `utils/markdown_docx.py`.

## Storage (3-layer)

- **PostgreSQL:** metadata, structure, tasks, searchable text
- **MinIO:** originals, page/crop images, large blobs via `*_key` columns
- **Local:** `UPLOAD_DIR/<doc_id>/` via `services/storage_service.py`

Helpers: `utils/storage_keys.py`, `utils/content_storage.py`, `utils/text_assembly.py`, `utils/tree_payload.py`. Migrations: `scripts/migrate_*.py`, `alembic upgrade head`.

Structure-preserving (never bloat): `layout_elements`, `translated_elements`, `tree_data`.

## Other endpoints

- **Digest:** `POST/GET .../digest` — read-only assembly, no pipeline trigger
- **Tree index:** `POST/GET .../tree-index` — async; feeds summarization + keywords (`core/pageindex/`, `core/spatial/`)
- **Search:** `GET /api/v2/search?q=...` — titles + normalized content
- **Upload overrides:** `POST .../text/upload`, `.../translations/{tid}/upload`, `.../summaries/{sid}/upload`

## Prompts

Pattern: ROLE → TASK → CONSTRAINTS → OUTPUT FORMAT. Anti-hallucination: source-grounded claims only; preserve numbers/names/dates verbatim. Token truncation: `BaseEnricher.truncate_to_tokens()` in `core/pageindex/enrichment/base.py` (use `settings.ai_chunk_tokens - 1000`).

## Deploy & auto-restart

Two tiers — use **systemd first**, then **Docker** when packaging is required.

**Tier 1 — systemd (host API + worker):**
```bash
bash deploy/install-autostart.sh          # infra → backend → temporal-worker + PM2 FE
bash deploy/check-backend.sh
sudo systemctl status docuflow-infra docuflow-backend docuflow-temporal-worker
journalctl -u docuflow-temporal-worker -f
bash deploy/uninstall-autostart.sh
```
Units: `deploy/docuflow-infra.service` (docker: postgres/minio/temporal), `docuflow-backend.service` (`start.sh`), `docuflow-temporal-worker.service` (`Restart=always`).

**Tier 2 — Docker (API + worker in containers):**
```bash
# Merge deploy/docker.env.example into .env (host GPU via host.docker.internal)
docker compose --profile app up -d --build   # infra + api + worker; restart: unless-stopped
bash deploy/install-docker-autostart.sh      # systemd → compose --profile app on boot
bash deploy/check-backend.sh --docker
```
GPU OCR (vLLM :8000) and pipeline LLM (llama :5011) stay on the **host** in both tiers.
