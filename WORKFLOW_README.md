# DocuFlow Pipeline Workflow

End-to-end guide for processing a document through the API: upload → extract → enrichment (translate, summarize, keywords, research, main content) → digest.

For setup / install / configuration see the main [README.md](./README.md). All endpoints below assume the server is running at `http://localhost:8002` and you have a JWT in `$TOKEN`.

---

## Job Status Lifecycle

Every async step creates a **job record** in its own table plus a **Task row** for progress.

```
PENDING ─▶ IN_PROGRESS ─▶ COMPLETED
                      └─▶ FAILED
```

| Resource | List endpoint | Detail endpoint |
|----------|---------------|-----------------|
| Translation | `GET /api/v2/documents/{id}/translations` | `GET …/translations/{tid}` |
| Summary | `GET /api/v2/documents/{id}/summaries` | `GET …/summaries/{sid}` |
| Main Content | `GET /api/v2/documents/{id}/main-content` | `GET …/main-content/{mid}` |
| Keywords | `GET /api/v2/documents/{id}/keywords` (current + `latest_extraction`) | `GET …/keywords/extractions/{eid}` |
| Research | `GET /api/v2/documents/{id}/research-directions` (current + `latest_extraction`) | `GET …/research-directions/extractions/{eid}` |

> `Document.processing_status` only tracks OCR/extraction (`INIT → EXTRACT_IN_PROGRESS → EXTRACTED → FAILED`). It does **not** change for translate/summary/keywords/etc.

---

## End-to-End Workflow

### 1. Login

```bash
TOKEN=$(curl -s -X POST http://localhost:8002/api/v2/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' \
  | jq -r .access_token)
```

### 2. Upload a document

```bash
curl -X POST http://localhost:8002/api/v2/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@research_paper.pdf \
  -F source_language=en
# → { "document_id": "DOC_001", "processing_status": "INIT", ... }
```

### 3. Run extraction (OCR / native)

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/extract \
  -H "Authorization: Bearer $TOKEN"
# → { "task_id": "EXTRACT_001", "status": "PENDING" }
```

Poll either:
- `GET /api/v2/tasks/EXTRACT_001` (progress 0–100, message)
- `GET /api/v2/documents/DOC_001` (`processing_status` flips to `EXTRACTED`)

### 4. (Optional) Override extracted text

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/text/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@corrected.txt
```

Download extracted text as .docx:
```bash
curl -OJ http://localhost:8002/api/v2/documents/DOC_001/text/download?type=ocr \
  -H "Authorization: Bearer $TOKEN"
# type=ocr (default) or type=normalized
```

### 5. (Optional) Build tree index

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/tree-index \
  -H "Authorization: Bearer $TOKEN"
# Enables hierarchical summarization (bottom-up tree walk).
# Keywords also use per-node candidates when tree exists.
```

### 6. Translate

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/translations \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"target_language":"vi","domain":"general"}'
# → { "task_id": "TRANSLATE_002", "resource_id": "<translation_id>", "status": "PENDING" }
```

`GET /api/v2/documents/DOC_001/translations` immediately shows the row with `status: "PENDING"` → `IN_PROGRESS` → `COMPLETED`.

### 7. Summarize

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/summaries \
  -H "Authorization: Bearer $TOKEN"
# → { "task_id": "HIERARCHICAL_SUMMARIZE_003", "resource_id": "<summary_id>", "status": "PENDING" }
```

### 8. Extract keywords

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/keywords \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"max_keywords":20}'
# → { "task_id": "KEYWORDS_004", "resource_id": "<extraction_id>", ... }
```

`GET …/keywords` returns current keywords plus `latest_extraction.status`. List historical jobs at `GET …/keywords/extractions`.

### 9. Identify research directions

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/research-directions \
  -H "Authorization: Bearer $TOKEN"
# → { "task_id": "RESEARCH_DIRECTIONS_005", "resource_id": "<extraction_id>", ... }
```

### 10. Extract main content

```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/main-content \
  -H "Authorization: Bearer $TOKEN"
# → { "task_id": "MAIN_CONTENT_006", "resource_id": "<main_content_id>", ... }
```

### 11. Assemble the digest

JSON preview:
```bash
curl -X POST http://localhost:8002/api/v2/documents/DOC_001/digest \
  -H "Authorization: Bearer $TOKEN"
# Response includes: title, abstract, main_content, keywords, research_directions, missing[]
```

Download as `.docx` (official Tổng thuật template):
```bash
curl -OJ http://localhost:8002/api/v2/documents/DOC_001/digest/download \
  -H "Authorization: Bearer $TOKEN"
```

The digest reads only `COMPLETED` rows for Summary and MainContent. Sections not yet processed are listed under `missing` (e.g. `["abstract", "keywords"]`).

---

## Polling Task Progress

`GET /api/v2/tasks/{task_id}` returns:

```json
{
  "task_id": "TRANSLATE_002",
  "document_id": "DOC_001",
  "task_type": "TRANSLATE",
  "status": "RUNNING",
  "progress": 47,
  "message": "Chunk 4/9",
  "result": null,
  "error": null,
  "created_at": "2026-04-25T17:10:11Z",
  "updated_at": "2026-04-25T17:10:18Z"
}
```

**When to poll which?**

| Use case | Endpoint |
|----------|----------|
| Want progress % / live status bar | `GET /api/v2/tasks/{task_id}` |
| Want list of all jobs for a doc | `GET /api/v2/tasks?document_id=…` or the resource list endpoint |
| Want to render the result content | resource detail endpoint (`/translations/{tid}`, `/summaries/{sid}`, …) |
| Want to know if any pipeline step is running | resource list — check `status` field |

---

## File Upload Override

Three endpoints accept user-uploaded `.txt` / `.docx` to replace auto-generated content:

| Endpoint | Replaces |
|----------|----------|
| `POST /api/v2/documents/{id}/text/upload` | OCR / extracted text (`normalized_content`) |
| `POST /api/v2/documents/{id}/translations/{tid}/upload` | Translation content |
| `POST /api/v2/documents/{id}/summaries/{sid}/upload` | Summary content |

The body is `multipart/form-data` with a single `file` field.

---

## Frontend Integration Tips

- **Optimistic rendering** — right after POST, GET the resource list. Newly created `PENDING` row appears immediately, so you can render a row with a spinner without local state.
- **Polling cadence** — 2–3 seconds against `GET /api/v2/tasks/{task_id}` is enough for most pipelines.
- **Status color map** — `PENDING` (gray), `IN_PROGRESS` (blue), `COMPLETED` (green), `FAILED` (red). Same enum across every pipeline.
- **Failure** — for `FAILED` rows, fetch the corresponding `Task` row to read `error` (full stack trace). Keywords/Research jobs also store a short `error` string on the extraction row itself.
- **Document list** — `GET /api/v2/documents` returns each row with a `task_summary` dict (`{EXTRACT: COMPLETED, TRANSLATE: RUNNING, …}`) so you can render the dashboard without N+1 calls.
