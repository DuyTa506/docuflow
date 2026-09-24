"""
Base abstract class for LLM clients.

This defines the interface that all LLM provider implementations must follow.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
            **kwargs: Additional provider-specific configuration.
                max_concurrent: bounds concurrent requests against the backing
                    server (shared across every caller of this client instance
                    — see api.dependencies.get_llm_client, which caches one
                    instance per provider/model). Defaults to 4.
        """
        self.model = model
        self.config = kwargs
        max_concurrent = kwargs.get("max_concurrent") or 4
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    @abstractmethod
    async def chat_completion(
        self, prompt: str, chat_history: Optional[List[Dict[str, str]]] = None, **kwargs
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
        self, prompt: str, chat_history: Optional[List[Dict[str, str]]] = None, **kwargs
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

    def extract_json(self, content: str, expected_root: Optional[str] = None) -> Any:
        """
        Extract JSON from LLM response.

        When *expected_root* is ``"list"`` or ``"dict"``, only a complete value
        of that root type is returned. A truncated outer array must not resume
        scanning at an inner ``{``.

        Args:
            content: Raw response content from LLM
            expected_root: Optional ``"list"`` or ``"dict"`` contract

        Returns:
            Parsed JSON value

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try direct JSON parse first
        try:
            value = json.loads(content)
            if self._root_matches(value, expected_root):
                return value
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks
        json_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            try:
                value = json.loads(json_match.group(1))
                if self._root_matches(value, expected_root):
                    return value
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        allowed = self._scan_chars(expected_root, content)
        for idx, ch in enumerate(content):
            if ch not in allowed:
                continue
            try:
                value, _ = decoder.raw_decode(content, idx)
                if self._root_matches(value, expected_root):
                    return value
            except json.JSONDecodeError:
                continue

        logger.warning(
            "extract_json found no valid JSON in LLM response | expected_root=%s | snippet: %r",
            expected_root,
            content[:200],
        )
        raise json.JSONDecodeError(
            f"Could not extract valid JSON from content: {content[:200]}...", content, 0
        )

    @staticmethod
    def _root_matches(value: Any, expected_root: Optional[str]) -> bool:
        if expected_root == "list":
            return isinstance(value, list)
        if expected_root == "dict":
            return isinstance(value, dict)
        return True

    @staticmethod
    def _scan_chars(expected_root: Optional[str], content: str) -> str:
        if expected_root == "list":
            return "["
        if expected_root == "dict":
            return "{"
        return "{["

    def get_json_content(self, response: str) -> str:
        """
        Extract JSON content from markdown code blocks.

        Args:
            response: Response text that may contain ```json ... ``` blocks

        Returns:
            Extracted JSON string
        """
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        elif response.startswith("```"):
            response = response.replace("```", "").strip()
        return response
