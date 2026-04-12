"""
Ollama client implementation.

This wraps the Ollama API and implements the BaseLLMClient interface.
"""

import httpx
import json
from typing import Optional, Tuple, Dict, List
from .llm_client_base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    LLM client for Ollama (local LLM server).
    """
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name (e.g., 'qwen3:30b', 'llama3:latest')
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 300)
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout)
        )
    
    async def chat_completion(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Call Ollama chat completion API.
        
        Args:
            prompt: User prompt
            chat_history: Optional conversation history
            **kwargs: Additional parameters (temperature, etc.)
            
        Returns:
            Model response text
            
        Raises:
            httpx.ConnectError: If cannot connect to Ollama server
            httpx.HTTPStatusError: If server returns error
        """
        # Build messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        # Add optional parameters
        options = {}
        if 'temperature' in kwargs:
            options['temperature'] = kwargs['temperature']
        if options:
            payload['options'] = options
        
        try:
            # Call Ollama API
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result['message']['content'].strip()
            
        except httpx.ConnectError as e:
            raise Exception(
                f"Could not connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running (e.g., 'ollama serve') and "
                f"you have pulled the model (e.g., 'ollama pull {self.model}'). "
                f"Error: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"Ollama server returned error: {e.response.status_code} - {e.response.text}"
            )
        except json.JSONDecodeError as e:
            raise Exception(
                f"Invalid JSON response from Ollama: {response.text}"
            )
    
    async def chat_completion_with_finish_reason(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Call Ollama API and return response with finish reason.
        
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
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        # Add optional parameters
        options = {}
        if 'temperature' in kwargs:
            options['temperature'] = kwargs['temperature']
        if options:
            payload['options'] = options
        
        try:
            # Call Ollama API
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content = result['message']['content'].strip()
            
            # Ollama uses 'done_reason' field
            # Map to our standard format
            done_reason = result.get('done_reason', 'stop')
            if done_reason == 'stop' or result.get('done', False):
                finish_reason = 'finished'
            elif done_reason == 'length':
                finish_reason = 'length'
            else:
                finish_reason = 'finished'  # Default
            
            return content, finish_reason
            
        except httpx.ConnectError as e:
            raise Exception(
                f"Could not connect to Ollama server at {self.base_url}. "
                f"Error: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"Ollama server returned error: {e.response.status_code} - {e.response.text}"
            )
        except json.JSONDecodeError as e:
            raise Exception(
                f"Invalid JSON response from Ollama: {response.text}"
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Ollama models.
        
        Since Ollama doesn't provide a tokenizer API, we use a simple heuristic:
        approximately 4 characters per token (similar to OpenAI's GPT models).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple heuristic: ~4 chars per token
        return len(text) // 4
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
