"""
Base class for document enrichment operations.

Provides common utilities for enrichment features like translation,
summarization, keyword extraction, etc.
"""

from typing import Dict, List, Optional
from ..llm.llm_client_base import BaseLLMClient


class BaseEnricher:
    """
    Base class for document enrichment operations.
    
    All enrichers (translator, summarizer, etc.) should inherit from this.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize enricher with LLM client.
        
        Args:
            llm_client: LLM client instance for making completions
        """
        self.llm_client = llm_client
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using LLM client's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.llm_client.count_tokens(text)
    
    def chunk_text(
        self, 
        text: str, 
        max_tokens: int = 2000,
        overlap: int = 100
    ) -> List[str]:
        """
        Split long text into chunks that fit within token limit.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple implementation - split by sentences
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Finish current chunk
                chunks.append('. '.join(current_chunk) + '.')
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 0:
                    # Keep last sentence for context
                    current_chunk = [current_chunk[-1], sentence]
                    current_tokens = self.count_tokens(current_chunk[-1]) + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    async def process_with_retry(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> str:
        """
        Make LLM completion with retry logic.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retries
            
        Returns:
            LLM response
        """
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat_completion(prompt)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{max_retries} due to error: {e}")
                continue
        
        raise Exception("Max retries exceeded")
