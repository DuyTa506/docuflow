"""
Configuration management using Pydantic Settings.

Environment variables:
- VLLM_API_KEY: API key for vLLM server
- VLLM_SERVER_URL: Base URL for vLLM server
- DATABASE_URL: SQLAlchemy database URL
- OCR_MAX_TOKENS: Maximum tokens for OCR
- OCR_TEMPERATURE: Temperature for OCR model
- JWT_SECRET_KEY: Secret key for JWT tokens
- AI_PROVIDER / AI_MODEL: LLM backend for AI services
- AI_MODEL_CONTEXT_WINDOW: Token context window of the pipeline LLM
- AI_CHUNK_RATIO: Fraction of context window used per chunk (0–1)
- SUMMARY_OUTPUT_LANG / RESEARCH_OUTPUT_LANG: legacy env vars (pipeline output is always vi)
"""

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from core.constants import OCR_PROMPTS

# Populate os.environ from .env — needed by modules that read os.getenv()
# directly (e.g. core/pageindex/llm/openai_client.py), in addition to this
# Settings class's own `env_file` loading below.
load_dotenv()

# Mandatory output language for pipeline LLM tasks (not translation / OCR / keywords).
PIPELINE_OUTPUT_LANG = "vi"

# Priority translation pair: English source → Chinese / Russian targets (codes stored as-is).
TRANSLATION_PRIORITY_LANGS: tuple[str, ...] = ("en", "zh", "ru")

LANG_NAME_MAP: dict[str, str] = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "ja": "Japanese",
    "fr": "French",
    "de": "German",
}

# Aliases → canonical codes used in DB and prompts.
LANG_CODE_ALIASES: dict[str, str] = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-tw": "zh",
    "zh-hant": "zh",
    "cn": "zh",
    "russian": "ru",
    "ru-ru": "ru",
}


def normalize_lang_code(code: str, *, default: str = "en") -> str:
    """Map BCP-47 / free-text input to a canonical language code (en, zh, ru, …)."""
    c = (code or "").strip().lower()
    if not c or c == "auto":
        return default
    if c in LANG_CODE_ALIASES:
        return LANG_CODE_ALIASES[c]
    if c in LANG_NAME_MAP:
        return c
    primary = c.split("-")[0]
    return LANG_CODE_ALIASES.get(primary, primary)


def lang_name(code: str) -> str:
    """Return a human-readable language name for a BCP-47 code."""
    canonical = normalize_lang_code(code, default="")
    if not canonical:
        return "the source language"
    return LANG_NAME_MAP.get(canonical, canonical)


def pipeline_output_lang_clause(*, json_values: bool = False) -> str:
    """Mandatory Vietnamese output for summarization, research, main content, tree summaries."""
    lang = lang_name(PIPELINE_OUTPUT_LANG)
    if json_values:
        return (
            f"OUTPUT LANGUAGE: All human-readable string values in your response MUST be "
            f"written in {lang}. Proper nouns, acronyms, numbers, and technical terms "
            f"quoted verbatim from the source document may remain in their original form.\n\n"
        )
    return (
        f"OUTPUT LANGUAGE: You MUST respond entirely in {lang}. "
        f"Do not use any other language for generated prose.\n\n"
    )


def pipeline_keyword_lang_clause() -> str:
    """Keywords must stay in the document source language — never translate."""
    return (
        "OUTPUT LANGUAGE: Each `keyword` string MUST appear verbatim in the same language "
        "as the source document. Do NOT translate keywords to Vietnamese or any other language.\n\n"
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── vLLM Configuration ──────────────────────────────────────────
    vllm_api_key: str = Field(default="123", env="VLLM_API_KEY")
    vllm_server_url: str = Field(
        default="http://localhost:8000/v1",
        env="VLLM_SERVER_URL",
    )
    vllm_model: str = Field(default="deepseek-ai/DeepSeek-OCR-2", env="VLLM_MODEL")

    # ── Database Configuration ──────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg2://docuflow:docuflow@localhost:5433/docuflow",
        env="DATABASE_URL",
    )

    # ── OCR Parameters ──────────────────────────────────────────────
    ocr_max_tokens: int = Field(default=4096, env="OCR_MAX_TOKENS")
    ocr_temperature: float = Field(default=0.0, env="OCR_TEMPERATURE")
    ocr_target_dpi: int = Field(default=200, env="OCR_TARGET_DPI")
    ocr_max_image_size: int = Field(default=2048, env="OCR_MAX_IMAGE_SIZE")
    ocr_prompt: str = Field(
        default=OCR_PROMPTS["markdown"],
        env="OCR_PROMPT",
        description=(
            "DeepSeek-OCR grounding prompt. Must include <image> and <|grounding|>. "
            "Do not add LaTeX instructions here — equation labels are converted to "
            "Word equations (OMML) at DOCX/PDF export time."
        ),
    )

    # ── API Configuration ───────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8022, env="API_PORT")

    # ── Spatial Analysis ────────────────────────────────────────────
    spatial_vertical_weight: float = Field(default=0.2, env="SPATIAL_VERTICAL_WEIGHT")
    spatial_size_weight: float = Field(default=0.3, env="SPATIAL_SIZE_WEIGHT")
    spatial_label_weight: float = Field(default=0.4, env="SPATIAL_LABEL_WEIGHT")
    spatial_indent_weight: float = Field(default=0.1, env="SPATIAL_INDENT_WEIGHT")

    # ── PageIndex Configuration ─────────────────────────────────────
    pageindex_llm_provider: str = Field(default="openai", env="PAGEINDEX_LLM_PROVIDER")
    pageindex_model: str = Field(default="gpt-4o-2024-11-20", env="PAGEINDEX_MODEL")
    pageindex_ollama_base_url: str = Field(
        default="http://localhost:11434", env="PAGEINDEX_OLLAMA_BASE_URL"
    )
    pageindex_ollama_timeout: int = Field(default=300, env="PAGEINDEX_OLLAMA_TIMEOUT")

    # Tree Building Parameters
    pageindex_if_thinning: bool = Field(default=True, env="PAGEINDEX_IF_THINNING")
    pageindex_min_token_threshold: int = Field(default=5000, env="PAGEINDEX_MIN_TOKEN_THRESHOLD")
    pageindex_if_add_node_summary: str = Field(default="yes", env="PAGEINDEX_IF_ADD_NODE_SUMMARY")
    pageindex_summary_token_threshold: int = Field(
        default=200, env="PAGEINDEX_SUMMARY_TOKEN_THRESHOLD"
    )
    pageindex_if_add_doc_description: str = Field(
        default="no", env="PAGEINDEX_IF_ADD_DOC_DESCRIPTION"
    )
    pageindex_if_add_node_text: str = Field(default="no", env="PAGEINDEX_IF_ADD_NODE_TEXT")
    pageindex_if_add_node_id: str = Field(default="yes", env="PAGEINDEX_IF_ADD_NODE_ID")

    # ── JWT / Authentication ────────────────────────────────────────
    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=480, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # ── AI Service (for translation, summarization, etc.) ───────────
    ai_provider: str = Field(default="openai", env="AI_PROVIDER")
    ai_model: str = Field(default="gpt-4o-2024-11-20", env="AI_MODEL")
    ai_ollama_base_url: str = Field(default="http://localhost:11434", env="AI_OLLAMA_BASE_URL")
    # Optional base URL for OpenAI-compatible endpoints (e.g. Alibaba DashScope).
    # Leave empty to use the default OpenAI API endpoint.
    ai_openai_base_url: str = Field(default="", env="AI_OPENAI_BASE_URL")

    # ── AI Chunking & Output Language ───────────────────────────────
    ai_model_context_window: int = Field(default=128000, env="AI_MODEL_CONTEXT_WINDOW")
    ai_chunk_ratio: float = Field(default=0.85, env="AI_CHUNK_RATIO")
    ai_output_reserve_tokens: int = Field(default=3000, env="AI_OUTPUT_RESERVE_TOKENS")
    # Default 8 matches llama.cpp `--parallel 8`. Benchmarked 2026-07-13 on
    # qwen3.5 (24 mixed requests): 4-concurrent = 47 tok/s, 8-concurrent =
    # 82 tok/s (+74%), p95 latency 7.7s → 10.1s. NOTE: the API process and
    # the Temporal worker each hold their own semaphore — split the 8 slots
    # across processes via env (.env_example) to avoid oversubscription.
    ai_max_concurrent_requests: int = Field(default=8, env="AI_MAX_CONCURRENT_REQUESTS")
    ai_request_timeout_seconds: int = Field(
        default=600,
        env="AI_REQUEST_TIMEOUT_SECONDS",
        description="HTTP timeout per LLM request — generous to allow queueing "
        "at the server, but bounded so a stalled backend can't hang a worker",
    )
    summary_output_lang: str = Field(default="vi", env="SUMMARY_OUTPUT_LANG")
    research_output_lang: str = Field(default="vi", env="RESEARCH_OUTPUT_LANG")

    @property
    def ai_chunk_tokens(self) -> int:
        """Per-chunk token budget = context_window * chunk_ratio."""
        return max(1, int(self.ai_model_context_window * self.ai_chunk_ratio))

    @property
    def ai_input_budget_tokens(self) -> int:
        """Input-token budget after reserving room for prompt overhead + output."""
        return max(1, self.ai_chunk_tokens - self.ai_output_reserve_tokens)

    # ── Adaptive summarization (large trees) ─────────────────────────
    summarize_cluster_threshold: int = Field(
        default=800,
        env="SUMMARIZE_CLUSTER_THRESHOLD",
        description="Above this many tree nodes, leaf siblings are batched "
        "into one prompt per cluster instead of one LLM call per node",
    )
    summarize_cluster_max_nodes: int = Field(default=10, env="SUMMARIZE_CLUSTER_MAX_NODES")
    summarize_checkpoint_nodes: int = Field(
        default=50,
        env="SUMMARIZE_CHECKPOINT_NODES",
        description="Persist node summaries back to the tree every N nodes "
        "so a retry resumes instead of redoing the whole stage",
    )
    # ── Digest §2.2 unit selection ──────────────────────────────────
    main_content_max_units: int = Field(
        default=0,
        env="MAIN_CONTENT_MAX_UNITS",
        description="Ceiling on §2.2 entries. 0 (default) scales it with document length "
        "(~25 entries for a 400-page book, sub-linear, clamped to 10..40). Set a positive "
        "value to pin the ceiling for a work that genuinely has more chapters. Exists "
        "because a tree levelled by percentile yields hundreds of pseudo-chapters "
        "(N4.11.160: 265 entries, 76 pages).",
    )
    main_content_min_unit_chars: int = Field(
        default=1500,
        env="MAIN_CONTENT_MIN_UNIT_CHARS",
        description="A §2.2 unit owning less than this much text is folded into its "
        "neighbour — the operational definition of 'not fragmented per heading'.",
    )
    main_content_unit_coverage_min: float = Field(
        default=0.60,
        env="MAIN_CONTENT_UNIT_COVERAGE_MIN",
        description="Reject a unit-selection tier whose first anchor covers less than this "
        "share of the document: it matched a body mention, not the structure.",
    )
    main_content_chapter_sample_tokens: int = Field(
        default=6000,
        env="MAIN_CONTENT_CHAPTER_SAMPLE_TOKENS",
        description="Budget for the stratified sample fed to each §2.2 chapter summary. "
        "Head truncation at 4000 chars showed the model ~1%% of a 90-page chapter.",
    )
    main_content_chapter_max_tokens: int = Field(
        default=700,
        env="MAIN_CONTENT_CHAPTER_MAX_TOKENS",
        description="Output cap per §2.2 chapter summary (prompt also constrains length, "
        "because the Ollama client drops max_tokens).",
    )
    main_content_prefer_node_summaries: bool = Field(
        default=True,
        env="MAIN_CONTENT_PREFER_NODE_SUMMARIES",
        description="Reuse per-node summaries already checkpointed into the tree by the "
        "summarize stage instead of re-reading raw text. No-op on a first run.",
    )
    summarize_node_content_tokens: int = Field(default=1500, env="SUMMARIZE_NODE_CONTENT_TOKENS")

    # ── Digest §3 — CTĐT catalog & research directions ───────────────
    ctdt_catalog_path: str = Field(
        default="",
        env="CTDT_CATALOG_PATH",
        description="Uploaded CTĐT catalog file. Empty (default) resolves to "
        "UPLOAD_DIR/catalog/ctdt_catalog.json, and falls back to the bundled "
        "config/ctdt_catalog.json when no upload exists. Not every deployment has "
        "the Academy's programme list — §3 degrades instead of inventing programmes.",
    )
    research_directions_max_items: int = Field(
        default=12,
        env="RESEARCH_DIRECTIONS_MAX_ITEMS",
        description="Directions requested per document. 20 items each carrying a "
        "Vietnamese `reasoning` field overran the output window and truncated the JSON.",
    )
    research_directions_max_tokens: int = Field(
        default=2000,
        env="RESEARCH_DIRECTIONS_MAX_TOKENS",
        description="Output cap for the research-directions call, which previously "
        "passed none at all and got cut mid-JSON.",
    )

    translation_prompt_overhead_tokens: int = Field(
        default=600,
        env="TRANSLATION_PROMPT_OVERHEAD_TOKENS",
        description="Token cost of the translation prompt scaffold "
        "(terminology/structure constraints), reserved out of the slot budget",
    )

    @property
    def translation_chunk_tokens(self) -> int:
        """Per-chunk token budget for translation.

        A translation is roughly the size of its source, so one model slot
        must hold prompt overhead + source chunk + translated output —
        hence half the window after overhead, not the full input budget.
        """
        return max(
            512,
            (self.ai_chunk_tokens - self.translation_prompt_overhead_tokens) // 2,
        )

    # Spatial OCR export is O(n) in layout elements; cap keeps download fast on large docs.
    ocr_download_spatial_max_elements: int = Field(
        default=2500,
        env="OCR_DOWNLOAD_SPATIAL_MAX_ELEMENTS",
    )
    ocr_download_spatial_max_pages: int = Field(
        default=200,
        env="OCR_DOWNLOAD_SPATIAL_MAX_PAGES",
        description="Above this page count, DOCX download uses markdown path (faster)",
    )
    export_spatial_embed_images_max_elements: int = Field(
        default=800,
        env="EXPORT_SPATIAL_EMBED_IMAGES_MAX_ELEMENTS",
        description="Above this, spatial DOCX skips MinIO image embed (text only)",
    )
    export_pandoc_max_chars: int = Field(
        default=300_000,
        env="EXPORT_PANDOC_MAX_CHARS",
        description="Skip pandoc above this size; use python-docx renderer",
    )

    # ── Structure-preserving export ─────────────────────────────────
    docx_export_engine: str = Field(
        default="auto",
        env="DOCX_EXPORT_ENGINE",
        description="auto|pandoc|python|spatial — pandoc converts LaTeX to OMML",
    )
    enable_pdf_overlay: bool = Field(default=False, env="ENABLE_PDF_OVERLAY")
    pdf_render_engine: str = Field(
        default="hybrid",
        env="PDF_RENDER_ENGINE",
        description="hybrid=layout renderer (default); overlay=legacy pdf2zh path (rollback)",
    )
    pdf_overlay_threads: int = Field(default=4, env="PDF_OVERLAY_THREADS")
    pdf_overlay_merge_max_chars: int = Field(default=800, env="PDF_OVERLAY_MERGE_MAX_CHARS")
    pdf_overlay_max_pages: int = Field(
        default=0,
        env="PDF_OVERLAY_MAX_PAGES",
        description="0 = all pages; set to N for dev/test limit",
    )
    pdf_overlay_max_scanned_ratio: float = Field(
        default=0.02,
        env="PDF_OVERLAY_MAX_SCANNED_RATIO",
        description="Max share of scanned pages still allowed to use the overlay path. "
        "A scanned page has no text layer to translate and passes through untouched, "
        "so a few of them must not demote a whole book to the flat tree path.",
    )
    pdf_overlay_text_mask: bool = Field(default=True, env="PDF_OVERLAY_TEXT_MASK")
    layout_pdf_text_overlay_pad: float = Field(default=1.5, env="LAYOUT_PDF_TEXT_OVERLAY_PAD")
    layout_pdf_text_expand_ratio: float = Field(default=0.8, env="LAYOUT_PDF_TEXT_EXPAND_RATIO")
    layout_pdf_export_dpi: int = Field(
        default=150,
        env="LAYOUT_PDF_EXPORT_DPI",
        description="DPI for re-rendering page backgrounds at export time, independent of the OCR model's low-res input image. "
        "150 was chosen after measuring real output: 200dpi/quality=95 produced ~1MB/page (30MB for an 18-page doc); "
        "150dpi/quality=85 (~470KB/page) is still a clear resolution upgrade over the OCR model's 1344px-capped image "
        "while keeping export file size reasonable.",
    )
    layout_pdf_export_max_size: int = Field(
        default=3000,
        env="LAYOUT_PDF_EXPORT_MAX_SIZE",
        description="Max pixel dimension for export-time page background renders (higher than the OCR model's 1344px cap).",
    )
    layout_pdf_export_jpeg_quality: int = Field(
        default=85,
        env="LAYOUT_PDF_EXPORT_JPEG_QUALITY",
        description="JPEG quality for export-time page backgrounds. Kept separate from the OCR model's own image "
        "quality (95) since export renders are much larger pixel dimensions -- 95 there would multi-MB-bloat every page.",
    )
    translation_block_merge: bool = Field(default=True, env="TRANSLATION_BLOCK_MERGE")
    translation_element_max: int = Field(default=500, env="TRANSLATION_ELEMENT_MAX")
    # Runaway guard only — NOT the download-speed cap. Routing translation
    # through the spatial path must survive large books (BlockTranslator is
    # built for them); falling to tree/flat permanently strips figures from
    # the translated artifact.
    translation_spatial_max_elements: int = Field(
        default=20000,
        env="TRANSLATION_SPATIAL_MAX_ELEMENTS",
    )
    translation_parallelism: int = Field(default=4, env="TRANSLATION_PARALLELISM")
    doclayout_model_path: str = Field(default="", env="DOCLAYOUT_MODEL_PATH")
    pdf_overlay_vi_font_path: str = Field(
        default="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        env="PDF_OVERLAY_VI_FONT_PATH",
        description="Font used for pdf_overlay translations targeting Vietnamese specifically. "
        "Confirmed this file has full coverage of all Vietnamese precomposed tone-marked "
        "characters and is ~322KB vs. GoNotoKurrent's ~15MB (unsubsetted) -- matches the "
        "Times New Roman convention already used for DOCX exports. Falls back to the "
        "multi-script GoNotoKurrent font (needed for zh/ja/ko/ar/hi/th/etc.) if this path "
        "doesn't exist on disk.",
    )
    max_concurrent_tasks: int = Field(default=4, env="MAX_CONCURRENT_TASKS")

    # ── Structured ETA estimation ────────────────────────────────────
    eta_enabled: bool = Field(default=True, env="ETA_ENABLED")
    eta_shadow_mode: bool = Field(
        default=True,
        env="ETA_SHADOW_MODE",
        description="Calculate/log predictions but withhold numeric ranges from clients",
    )
    eta_public_profile_keys: str = Field(
        default="",
        env="ETA_PUBLIC_PROFILE_KEYS",
        description="Comma-separated validated profile keys allowed to publish; '*' enables all",
    )
    eta_live_sample_threshold: int = Field(default=3, env="ETA_LIVE_SAMPLE_THRESHOLD")
    eta_ema_alpha: float = Field(default=0.25, env="ETA_EMA_ALPHA")
    eta_profile_min_samples: int = Field(default=20, env="ETA_PROFILE_MIN_SAMPLES")
    eta_profile_max_observations: int = Field(default=200, env="ETA_PROFILE_MAX_OBSERVATIONS")
    eta_hysteresis_ratio: float = Field(default=0.10, env="ETA_HYSTERESIS_RATIO")
    eta_hysteresis_seconds: int = Field(default=60, env="ETA_HYSTERESIS_SECONDS")
    eta_max_step_ratio: float = Field(default=0.25, env="ETA_MAX_STEP_RATIO")
    eta_stall_p90_multiplier: float = Field(default=3.0, env="ETA_STALL_P90_MULTIPLIER")
    eta_stall_min_seconds: int = Field(default=120, env="ETA_STALL_MIN_SECONDS")

    # ── Temporal (digest pipeline) ────────────────────────────────────
    temporal_host: str = Field(default="localhost:7233", env="TEMPORAL_HOST")
    temporal_namespace: str = Field(default="default", env="TEMPORAL_NAMESPACE")
    temporal_task_queue: str = Field(default="docuflow-digest", env="TEMPORAL_TASK_QUEUE")
    temporal_translation_task_queue: str = Field(
        default="docuflow-translation", env="TEMPORAL_TRANSLATION_TASK_QUEUE"
    )
    translation_use_temporal: bool = Field(
        default=True,
        env="TRANSLATION_USE_TEMPORAL",
        description="Route new translations through TranslationWorkflow "
        "(durable, resumable) instead of the in-process task manager",
    )
    temporal_extraction_task_queue: str = Field(
        default="docuflow-extraction", env="TEMPORAL_EXTRACTION_TASK_QUEUE"
    )
    ocr_use_temporal: bool = Field(
        default=True,
        env="OCR_USE_TEMPORAL",
        description="Route OCR/extraction through ExtractionWorkflow — "
        "survives API restarts; retries resume from already-extracted pages",
    )
    temporal_retention_hours: float = Field(
        default=24.0,
        env="TEMPORAL_RETENTION_HOURS",
        description="Temporal's WorkflowExecutionRetentionTtl. Past this a "
        "closed workflow is gone from history, so 'not found' only proves a "
        "run is dead once the Task row is older than this",
    )
    temporal_stage_task_queue: str = Field(
        default="docuflow-stage", env="TEMPORAL_STAGE_TASK_QUEUE"
    )
    stage_rerun_use_temporal: bool = Field(
        default=True,
        env="STAGE_RERUN_USE_TEMPORAL",
        description="Route standalone single-stage reruns (keywords, summary, "
        "main content, tree…) through StageRerunWorkflow. On book-length "
        "documents these run for hours; in-process they had no timeout, "
        "heartbeat, retry or resume and died with every API restart",
    )
    auto_extract_on_upload: bool = Field(
        default=True,
        env="AUTO_EXTRACT_ON_UPLOAD",
        description="Kick off OCR/extraction immediately after upload — no "
        "second click needed; failure to submit never fails the upload",
    )
    extraction_max_concurrent: int = Field(
        default=1,
        env="EXTRACTION_MAX_CONCURRENT",
        description="Concurrent extraction workflows per worker. OCR is "
        "GPU-bound on the shared vLLM server — per-page fan-out inside one "
        "document already saturates it; raise to 2 to let a small doc "
        "overlap a big book at the cost of slowing both",
    )
    max_concurrent_pipelines: int = Field(
        default=2,
        env="MAX_CONCURRENT_PIPELINES",
        description="Submit-layer cap on OPEN digest (and heavy stage-rerun) tasks",
    )
    max_concurrent_translations: int = Field(
        default=2,
        env="MAX_CONCURRENT_TRANSLATIONS",
        description="Submit-layer cap on OPEN translation tasks",
    )
    max_concurrent_jobs_per_user: int = Field(
        default=3,
        env="MAX_CONCURRENT_JOBS_PER_USER",
        description="Per-user fairness cap across digest/extract/translate",
    )
    digest_group_a_parallelism: int = Field(
        default=2,
        env="DIGEST_GROUP_A_PARALLELISM",
        description="Max concurrent Group A digest stages on this host (biblio/"
        "keywords/research/usage). 4 saturates a single-GPU llama.cpp slot pool",
    )
    digest_group_b_parallel: bool = Field(
        default=True,
        env="DIGEST_GROUP_B_PARALLEL",
        description="Run summarize and main-content as a pair. Set false to "
        "serialize them on a contended GPU",
    )
    gpu_lease_dir: str = Field(default="", env="GPU_LEASE_DIR")
    gpu_docling_slots: int = Field(default=1, env="GPU_DOCLING_SLOTS")
    gpu_lease_ttl_seconds: int = Field(default=90, env="GPU_LEASE_TTL_SECONDS")
    gpu_lease_wait_seconds: int = Field(
        default=600,
        env="GPU_LEASE_WAIT_SECONDS",
        description="How long an extraction waits for the Docling GPU lease",
    )
    worker_graceful_shutdown_seconds: int = Field(
        default=300,
        env="WORKER_GRACEFUL_SHUTDOWN_SECONDS",
        description="Temporal worker drain window on SIGTERM before activities "
        "are interrupted; systemd TimeoutStopSec must be larger",
    )
    temporal_max_concurrent_activities: int = Field(
        default=8,
        env="TEMPORAL_MAX_CONCURRENT_ACTIVITIES",
        description="Worker-wide cap on concurrently executing activities "
        "(NOT pipelines) — must be >= the widest parallel stage group",
    )
    tree_index_max_age_hours: int = Field(
        default=168,
        env="TREE_INDEX_MAX_AGE_HOURS",
        description="Skip BUILD_TREE in digest workflow if index is newer than this",
    )

    # ── MinIO object storage ─────────────────────────────────────────
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="docuflow", env="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")

    # Default preview page count for OCR / translation text endpoints
    export_preview_pages: int = Field(default=2, env="EXPORT_PREVIEW_PAGES")

    # Offload very large text blobs to MinIO (chars); 0 disables offload
    text_offload_threshold_chars: int = Field(
        default=500_000,
        env="TEXT_OFFLOAD_THRESHOLD_CHARS",
    )

    # ── Upload settings ─────────────────────────────────────────────
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_upload_bytes: int = Field(
        default=524_288_000,
        env="MAX_UPLOAD_BYTES",
        description="Reject uploads larger than this (default 500 MiB)",
    )
    max_documents_per_user: int = Field(
        default=200,
        env="MAX_DOCUMENTS_PER_USER",
        description="Per-user document quota; 0 disables",
    )
    require_registration_approval: bool = Field(
        default=True,
        env="REQUIRE_REGISTRATION_APPROVAL",
        description="All self-registered accounts start PENDING_APPROVAL",
    )
    cors_allow_origins: str = Field(
        default="*",
        env="CORS_ALLOW_ORIGINS",
        description="Comma-separated origins, or * for any (dev / same-LAN SPA)",
    )
    admin_password: str = Field(
        default="admin",
        env="ADMIN_PASSWORD",
        description="Bootstrap password for the first admin user (init_db)",
    )

    # ── Document extraction settings ────────────────────────────────
    libreoffice_path: str = Field(default="soffice", env="LIBREOFFICE_PATH")
    pdf_text_threshold: int = Field(default=50, env="PDF_TEXT_THRESHOLD")
    ocr_page_parallelism: int = Field(
        default=4,
        env="OCR_PAGE_PARALLELISM",
        description="Concurrent scanned-page OCR requests during extraction",
    )
    docling_do_ocr: bool = Field(
        default=False,
        env="DOCLING_DO_OCR",
        description="Run OCR inside Docling for text-layer PDFs (default: off)",
    )
    docling_table_structure: bool = Field(
        default=True,
        env="DOCLING_TABLE_STRUCTURE",
    )
    docling_artifacts_path: str = Field(
        default="",
        env="DOCLING_ARTIFACTS_PATH",
        description="Cache dir for Docling model weights (empty = default)",
    )
    docling_generate_picture_images: bool = Field(
        default=True,
        env="DOCLING_GENERATE_PICTURE_IMAGES",
        description="Extract embedded PDF figures as pixels for spatial export",
    )
    docling_images_scale: float = Field(
        default=2.5,
        env="DOCLING_IMAGES_SCALE",
        description="Resolution multiplier for extracted figure/chart images (Docling's own default of "
        "1.0 is ~72dpi-equivalent -- confirmed live: a chart with data labels extracted at only "
        "405x354px, visibly blurry once embedded in DOCX/PDF exports; 2.5 gives ~1011x885px).",
    )
    docling_do_formula_enrichment: bool = Field(
        default=True,
        env="DOCLING_DO_FORMULA_ENRICHMENT",
        description="Run Docling VLM for LaTeX formula conversion. Confirmed live: without "
        "this, standalone equations extract as garbled flat Unicode (no sub/superscripts); "
        "with it, proper LaTeX macros. Adds VLM inference time per document with equations.",
    )
    docling_min_picture_px: int = Field(
        default=40,
        env="DOCLING_MIN_PICTURE_PX",
        description="Skip decorative picture crops smaller than this on both sides",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # silently ignore unknown env vars (e.g. OPENAI_API_KEY read by openai lib directly)

    # ── Validators ──────────────────────────────────────────────────

    @field_validator("ocr_prompt")
    @classmethod
    def ensure_ocr_grounding_prefix(cls, v: str) -> str:
        """DeepSeek-OCR-2 requires <image> + <|grounding|>; repair common misconfig."""
        text = (v or "").strip()
        if not text:
            return OCR_PROMPTS["markdown"]
        if "<image>" not in text:
            text = f"<image>\n{text}"
        if "<|grounding|>" not in text:
            if text.startswith("<image>"):
                rest = text[len("<image>") :].lstrip("\n")
                text = f"<image>\n<|grounding|>{rest}"
            else:
                return OCR_PROMPTS["markdown"]
        return text

    @model_validator(mode="after")
    def warn_if_default_jwt_in_prod(self) -> "Settings":
        """Fail-fast in production when default secrets are still in place.

        LAN-first: this is durability (anyone on Wi-Fi must not be able to
        forge JWTs or wipe MinIO), not an enterprise control. Dev keeps the
        warning so local `.env.example` still boots.
        """
        import os
        import warnings

        prod = os.environ.get("DOCUFLOW_PROD", "").strip().lower() in ("1", "true", "yes")
        problems: list[str] = []
        if self.jwt_secret_key == "change-me-in-production":
            problems.append("JWT_SECRET_KEY")
        if self.minio_access_key == "minioadmin" or self.minio_secret_key == "minioadmin":
            problems.append("MINIO_ACCESS_KEY/MINIO_SECRET_KEY")
        if "docuflow:docuflow@" in (self.database_url or ""):
            problems.append("DATABASE_URL password")
        if self.admin_password == "admin":
            problems.append("ADMIN_PASSWORD")

        if not problems:
            return self

        msg = (
            "Insecure default credentials still set: "
            + ", ".join(problems)
            + ". Override them in .env before production use."
        )
        if prod:
            raise ValueError(f"PRODUCTION STARTUP REFUSED: {msg}")
        if "JWT_SECRET_KEY" in problems:
            warnings.warn(
                "JWT_SECRET_KEY is using the insecure default! "
                "Set JWT_SECRET_KEY via environment or .env before deploying.",
                UserWarning,
                stacklevel=2,
            )
        return self


# Global settings instance
settings = Settings()
