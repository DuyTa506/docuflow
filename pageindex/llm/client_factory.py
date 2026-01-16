"""
Factory for creating LLM clients.

This provides a centralized way to instantiate the correct LLM client
based on the provider configuration.
"""

from typing import Optional
from .llm_client_base import BaseLLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient


class LLMClientFactory:
    """
    Factory class for creating LLM clients.
    """
    
    @staticmethod
    def create_client(
        provider: str,
        model: str,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client based on the provider.
        
        Args:
            provider: Provider name ('openai' or 'ollama')
            model: Model name/identifier
            **kwargs: Provider-specific configuration
                For OpenAI:
                    - api_key: Optional API key
                For Ollama:
                    - base_url: Ollama server URL (default: http://localhost:11434)
                    - timeout: Request timeout in seconds (default: 300)
        
        Returns:
            Configured LLM client instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower().strip()
        
        if provider == 'openai':
            return OpenAIClient(
                model=model,
                api_key=kwargs.get('api_key'),
                **kwargs
            )
        elif provider == 'ollama':
            return OllamaClient(
                model=model,
                base_url=kwargs.get('ollama_base_url', 'http://localhost:11434'),
                timeout=kwargs.get('ollama_timeout', 300),
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported providers: 'openai', 'ollama'"
            )
    
    @staticmethod
    def get_supported_providers():
        """
        Get list of supported providers.
        
        Returns:
            List of provider names
        """
        return ['openai', 'ollama']
