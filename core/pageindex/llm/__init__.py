"""
LLM Client abstraction layer for PageIndex.

This module provides a unified interface for different LLM providers (OpenAI, Ollama, etc.)
using the Strategy Pattern.
"""

from .client_factory import LLMClientFactory
from .llm_client_base import BaseLLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "OllamaClient",
    "LLMClientFactory",
]
