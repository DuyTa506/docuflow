"""
Configuration management using Pydantic Settings.

Environment variables:
- VLLM_API_KEY: API key for vLLM server
- VLLM_SERVER_URL: Base URL for vLLM server  
- DATABASE_URL: SQLAlchemy database URL
- OCR_MAX_TOKENS: Maximum tokens for OCR
- OCR_TEMPERATURE: Temperature for OCR model
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # vLLM Configuration
    vllm_api_key: str = Field(default="123", env="VLLM_API_KEY")
    vllm_server_url: str = Field(
        default="http://localhost:8000/v1",
        env="VLLM_SERVER_URL"
    )
    vllm_model: str = Field(default="ocr", env="VLLM_MODEL")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///document_store.db",
        env="DATABASE_URL"
    )
    
    # OCR Parameters
    ocr_max_tokens: int = Field(default=4096, env="OCR_MAX_TOKENS")
    ocr_temperature: float = Field(default=0.0, env="OCR_TEMPERATURE")
    ocr_target_dpi: int = Field(default=200, env="OCR_TARGET_DPI")
    ocr_max_image_size: int = Field(default=2048, env="OCR_MAX_IMAGE_SIZE")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8002, env="API_PORT")
    
    # Spatial Analysis
    spatial_vertical_weight: float = Field(default=0.2, env="SPATIAL_VERTICAL_WEIGHT")
    spatial_size_weight: float = Field(default=0.3, env="SPATIAL_SIZE_WEIGHT")
    spatial_label_weight: float = Field(default=0.4, env="SPATIAL_LABEL_WEIGHT")
    spatial_indent_weight: float = Field(default=0.1, env="SPATIAL_INDENT_WEIGHT")
    
    # PageIndex Configuration
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_spatial_weights(self) -> dict:
        """Get spatial weights as dictionary."""
        return {
            'vertical': self.spatial_vertical_weight,
            'size': self.spatial_size_weight,
            'label': self.spatial_label_weight,
            'indent': self.spatial_indent_weight
        }
    
    def get_pageindex_config(self) -> dict:
        """Get PageIndex configuration as dictionary."""
        return {
            'llm_provider': self.pageindex_llm_provider,
            'model': self.pageindex_model,
            'ollama_base_url': self.pageindex_ollama_base_url,
            'ollama_timeout': self.pageindex_ollama_timeout,
            'if_thinning': self.pageindex_if_thinning,
            'min_token_threshold': self.pageindex_min_token_threshold,
            'if_add_node_summary': self.pageindex_if_add_node_summary,
            'summary_token_threshold': self.pageindex_summary_token_threshold,
            'if_add_doc_description': self.pageindex_if_add_doc_description,
            'if_add_node_text': self.pageindex_if_add_node_text,
            'if_add_node_id': self.pageindex_if_add_node_id,
        }


# Global settings instance
settings = Settings()
