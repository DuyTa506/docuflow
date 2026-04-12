"""
Base abstract class for LLM clients.

This defines the interface that all LLM provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import json
import re


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM provider implementations (OpenAI, Ollama, etc.) must inherit from this
    class and implement the abstract methods.
    """
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name/identifier
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    async def chat_completion(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Perform a chat completion request.
        
        Args:
            prompt: The user prompt/message
            chat_history: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            The model's response as a string
            
        Raises:
            Exception: If the API call fails
        """
        pass
    
    @abstractmethod
    async def chat_completion_with_finish_reason(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Perform a chat completion request and return the finish reason.
        
        Args:
            prompt: The user prompt/message
            chat_history: Optional conversation history
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_text, finish_reason)
            finish_reason can be: 'finished', 'length', 'stop', etc.
            
        Raises:
            Exception: If the API call fails
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def extract_json(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response.
        
        This handles common cases like:
        - JSON wrapped in markdown code blocks
        - JSON with extra text before/after
        
        Args:
            content: Raw response content from LLM
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array in the content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise json.JSONDecodeError(
            f"Could not extract valid JSON from content: {content[:200]}...",
            content,
            0
        )
    
    def get_json_content(self, response: str) -> str:
        """
        Extract JSON content from markdown code blocks.
        
        Args:
            response: Response text that may contain ```json ... ``` blocks
            
        Returns:
            Extracted JSON string
        """
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        elif response.startswith('```'):
            response = response.replace('```', '').strip()
        return response
