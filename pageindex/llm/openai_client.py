"""
OpenAI client implementation.

This wraps the OpenAI API and implements the BaseLLMClient interface.
"""

import os
from typing import Optional, Tuple, Dict, List
import tiktoken
from openai import AsyncOpenAI
from .llm_client_base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    LLM client for OpenAI API.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: OpenAI model name (e.g., 'gpt-4o-2024-11-20')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('CHATGPT_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def chat_completion(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Call OpenAI chat completion API.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Model response text
        """
        # Build messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        return response.choices[0].message.content.strip()
    
    async def chat_completion_with_finish_reason(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Call OpenAI API and return response with finish reason.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_text, finish_reason)
        """
        # Build messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        content = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        
        # Map OpenAI finish reasons to our standard format
        if finish_reason == 'stop':
            finish_reason = 'finished'
        
        return content, finish_reason
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
