"""
LLM wrapper module for PageIndex.

This provides a unified interface for LLM operations that wraps the new 
LLM client abstraction layer while maintaining backward compatibility.
"""

import asyncio
from typing import Optional, List, Dict, Tuple
from .llm import BaseLLMClient, LLMClientFactory


class LLMWrapper:
    """
    Wrapper class that provides both sync and async LLM operations.
    
    This maintains backward compatibility with the old ChatGPT_API functions
    while using the new LLM client abstraction layer underneath.
    """
    
    def __init__(self, client: BaseLLMClient):
        """
        Initialize the wrapper with an LLM client.
        
        Args:
            client: An instance of BaseLLMClient (OpenAIClient or OllamaClient)
        """
        self.client = client
    
    async def chat_completion_async(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Async chat completion.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            
        Returns:
            Model response
        """
        return await self.client.chat_completion(prompt, chat_history)
    
    def chat_completion(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Synchronous chat completion (runs async method in event loop).
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            
        Returns:
            Model response
        """
        return asyncio.run(self.client.chat_completion(prompt, chat_history))
    
    async def chat_completion_with_finish_reason_async(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str]:
        """
        Async chat completion with finish reason.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            
        Returns:
            Tuple of (response, finish_reason)
        """
        return await self.client.chat_completion_with_finish_reason(prompt, chat_history)
    
    def chat_completion_with_finish_reason(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str]:
        """
        Synchronous chat completion with finish reason.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            
        Returns:
            Tuple of (response, finish_reason)
        """
        return asyncio.run(
            self.client.chat_completion_with_finish_reason(prompt, chat_history)
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Number of tokens
        """
        return self.client.count_tokens(text)
    
    def extract_json(self, content: str) -> Dict:
        """
        Extract JSON from LLM response.
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed JSON dictionary
        """
        return self.client.extract_json(content)
    
    def get_json_content(self, response: str) -> str:
        """
        Extract JSON content from markdown code blocks.
        
        Args:
            response: Response text
            
        Returns:
            Extracted JSON string
        """
        return self.client.get_json_content(response)


def create_llm_wrapper_from_config(config) -> LLMWrapper:
    """
    Create an LLM wrapper from configuration.
    
    Args:
        config: Configuration object with llm_provider, model, etc.
        
    Returns:
        Configured LLM wrapper instance
    """
    provider = getattr(config, 'llm_provider', 'openai')
    model = getattr(config, 'model', 'gpt-4o-2024-11-20')
    
    # Prepare kwargs based on provider
    kwargs = {}
    if provider == 'ollama':
        kwargs['ollama_base_url'] = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        kwargs['ollama_timeout'] = getattr(config, 'ollama_timeout', 300)
    
    # Create client
    client = LLMClientFactory.create_client(
        provider=provider,
        model=model,
        **kwargs
    )
    
    return LLMWrapper(client)
