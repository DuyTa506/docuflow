"""
Factory for creating LLM clients.

This provides a centralized way to instantiate the correct LLM client
based on the provider configuration.
"""

from typing import Optional

from .llm_client_base import BaseLLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient


class LLMClientFactory:
    """
    Factory class for creating LLM clients.
    """

    @staticmethod
    def create_client(provider: str, model: str, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client based on the provider.

        Args:
            provider: Provider name ('openai' or 'ollama')
            model: Model name/identifier
            **kwargs: Provider-specific configuration
                For OpenAI / OpenAI-compatible (Alibaba DashScope, vLLM, etc.):
                    - api_key: Optional API key
                    - openai_base_url: Optional base URL override.
                      Leave empty for the default OpenAI endpoint.
                      Set to a DashScope URL to use Alibaba Cloud Qwen models.
                For Ollama:
                    - ollama_base_url: Ollama server URL (default: http://localhost:11434)
                    - ollama_timeout: Request timeout in seconds (default: 300)

        Returns:
            Configured LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower().strip()

        if provider == "openai":
            return OpenAIClient(
                model=model,
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("openai_base_url") or None,
                **kwargs,
            )
        elif provider == "ollama":
            return OllamaClient(
                model=model,
                base_url=kwargs.get("ollama_base_url", "http://localhost:11434"),
                timeout=kwargs.get("ollama_timeout", 300),
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported providers: 'openai', 'ollama'"
            )
