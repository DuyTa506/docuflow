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
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from core.constants import OCR_PROMPTS

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
        default="sqlite:///document_store.db",
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
    api_port: int = Field(default=8002, env="API_PORT")

    # ── Spatial Analysis ────────────────────────────────────────────
    spatial_vertical_weight: float = Field(default=0.2, env="SPATIAL_VERTICAL_WEIGHT")
    spatial_size_weight: float = Field(default=0.3, env="SPATIAL_SIZE_WEIGHT")
    spatial_label_weight: float = Field(default=0.4, env="SPATIAL_LABEL_WEIGHT")
    spatial_indent_weight: float = Field(default=0.1, env="SPATIAL_INDENT_WEIGHT")

    # ── PageIndex Configuration ─────────────────────────────────────
    pageindex_llm_provider: str = Field(default="openai", env="PAGEINDEX_LLM_PROVIDER")
    pageindex_model: str = Field(default="gpt-4o-2024-11-20", env="PAGEINDEX_MODEL")
    pageindex_ollama_base_url: str = Field(default="http://localhost:11434", env="PAGEINDEX_OLLAMA_BASE_URL")
    pageindex_ollama_timeout: int = Field(default=300, env="PAGEINDEX_OLLAMA_TIMEOUT")

    # Tree Building Parameters
    pageindex_if_thinning: bool = Field(default=True, env="PAGEINDEX_IF_THINNING")
    pageindex_min_token_threshold: int = Field(default=5000, env="PAGEINDEX_MIN_TOKEN_THRESHOLD")
    pageindex_if_add_node_summary: str = Field(default="yes", env="PAGEINDEX_IF_ADD_NODE_SUMMARY")
    pageindex_summary_token_threshold: int = Field(default=200, env="PAGEINDEX_SUMMARY_TOKEN_THRESHOLD")
    pageindex_if_add_doc_description: str = Field(default="no", env="PAGEINDEX_IF_ADD_DOC_DESCRIPTION")
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
    summary_output_lang: str = Field(default="vi", env="SUMMARY_OUTPUT_LANG")
    research_output_lang: str = Field(default="vi", env="RESEARCH_OUTPUT_LANG")

    @property
    def ai_chunk_tokens(self) -> int:
        """Per-chunk token budget = context_window * chunk_ratio."""
        return max(1, int(self.ai_model_context_window * self.ai_chunk_ratio))

    # Spatial OCR export is O(n) in layout elements; high cap (91k elems ~4s render).
    ocr_download_spatial_max_elements: int = Field(
        default=500000,
        env="OCR_DOWNLOAD_SPATIAL_MAX_ELEMENTS",
    )

    # ── Structure-preserving export ─────────────────────────────────
    docx_export_engine: str = Field(
        default="auto",
        env="DOCX_EXPORT_ENGINE",
        description="auto|pandoc|python|spatial — pandoc converts LaTeX to OMML",
    )
    enable_pdf_overlay: bool = Field(default=False, env="ENABLE_PDF_OVERLAY")
    pdf_overlay_threads: int = Field(default=4, env="PDF_OVERLAY_THREADS")
    doclayout_model_path: str = Field(default="", env="DOCLAYOUT_MODEL_PATH")
    max_concurrent_tasks: int = Field(default=4, env="MAX_CONCURRENT_TASKS")

    # ── Upload settings ─────────────────────────────────────────────
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")

    # ── Document extraction settings ────────────────────────────────
    libreoffice_path: str = Field(default="soffice", env="LIBREOFFICE_PATH")
    pdf_text_threshold: int = Field(default=50, env="PDF_TEXT_THRESHOLD")

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
        """Warn loudly when DOCUFLOW_PROD=1 but JWT secret is still the default."""
        import os
        import warnings

        if self.jwt_secret_key != "change-me-in-production":
            return self

        prod = os.environ.get("DOCUFLOW_PROD", "").strip().lower() in ("1", "true", "yes")
        msg = (
            "JWT_SECRET_KEY is using the insecure default! "
            "Set JWT_SECRET_KEY via environment or .env before deploying."
        )
        if prod:
            warnings.warn(f"PRODUCTION STARTUP: {msg}", UserWarning, stacklevel=2)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return self

    # ── Helper methods ──────────────────────────────────────────────

    def get_spatial_weights(self) -> dict:
        """Get spatial weights as dictionary."""
        return {
            "vertical": self.spatial_vertical_weight,
            "size": self.spatial_size_weight,
            "label": self.spatial_label_weight,
            "indent": self.spatial_indent_weight,
        }

    def get_pageindex_config(self) -> dict:
        """Get PageIndex configuration as dictionary."""
        return {
            "llm_provider": self.pageindex_llm_provider,
            "model": self.pageindex_model,
            "ollama_base_url": self.pageindex_ollama_base_url,
            "ollama_timeout": self.pageindex_ollama_timeout,
            "if_thinning": self.pageindex_if_thinning,
            "min_token_threshold": self.pageindex_min_token_threshold,
            "if_add_node_summary": self.pageindex_if_add_node_summary,
            "summary_token_threshold": self.pageindex_summary_token_threshold,
            "if_add_doc_description": self.pageindex_if_add_doc_description,
            "if_add_node_text": self.pageindex_if_add_node_text,
            "if_add_node_id": self.pageindex_if_add_node_id,
        }


# Global settings instance
settings = Settings()
