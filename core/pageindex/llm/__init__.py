"""
LLM Client abstraction layer for PageIndex.

This module provides a unified interface for different LLM providers (OpenAI, Ollama, etc.)
using the Strategy Pattern.
"""

from .llm_client_base import BaseLLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from .client_factory import LLMClientFactory

__all__ = [
    'BaseLLMClient',
    'OpenAIClient',
    'OllamaClient',
    'LLMClientFactory',
]
