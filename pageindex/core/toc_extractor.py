"""
TOC Extraction Component.

Responsible for extracting and parsing table of contents content from pages.
"""

import re
from typing import Dict, List
from ..llm.llm_client_base import BaseLLMClient


class TOCExtractor:
    """
    Extracts table of contents content from document pages.
    
    Handles raw text extraction and initial processing of TOC content.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize TOC extractor.
        
        Args:
            llm_client: LLM client instance for making API calls
        """
        self.llm = llm_client
    
    async def extract_content(self, content: str) -> str:
        """
        Extract TOC content from page text using LLM.
        
        Args:
            content: Raw page text
            
        Returns:
            Extracted and cleaned TOC content
        """
        prompt = f"""
        Your job is to extract the full table of contents from the given text, replace ... with :

        Given text: {content}

        Directly return the full table of contents content. Do not output anything else."""

        response, finish_reason = await self.llm.chat_completion_with_finish_reason(prompt)
        
        if_complete = await self._check_extraction_complete(content, response)
        if if_complete == "yes" and finish_reason == "finished":
            return response
        
        # Handle incomplete extraction
        chat_history = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response},    
        ]
        continuation_prompt = """please continue the generation of table of contents , directly output the remaining part of the structure"""
        
        new_response, finish_reason = await self.llm.chat_completion_with_finish_reason(
            continuation_prompt,
            chat_history=chat_history
        )
        response = response + new_response
        if_complete = await self._check_extraction_complete(content, response)
        
        retry_count = 0
        while not (if_complete == "yes" and finish_reason == "finished"):
            chat_history = [
                {"role": "user", "content": continuation_prompt}, 
                {"role": "assistant", "content": response},    
            ]
            new_response, finish_reason = await self.llm.chat_completion_with_finish_reason(
                continuation_prompt,
                chat_history=chat_history
            )
            response = response + new_response
            if_complete = await self._check_extraction_complete(content, response)
            
            retry_count += 1
            if retry_count > 5:
                raise Exception('Failed to complete table of contents extraction after maximum retries')
        
        return response
    
    async def _check_extraction_complete(self, content: str, toc: str) -> str:
        """
        Check if TOC extraction is complete.
        
        Args:
            content: Original document content
            toc: Extracted TOC
            
        Returns:
            'yes' if complete, 'no' otherwise
        """
        prompt = f"""
        You are given a raw table of contents and a  table of contents.
        Your job is to check if the  table of contents is complete.

        Reply format:
        {{
            "thinking": <why do you think the cleaned table of contents is complete or not>
            "completed": "yes" or "no"
        }}
        Directly return the final JSON structure. Do not output anything else."""

        prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
        response = await self.llm.chat_completion(prompt)
        json_content = self.llm.extract_json(response)
        return json_content['completed']
    
    async def detect_page_numbers(self, toc_content: str) -> str:
        """
        Detect if TOC content contains page numbers.
        
        Args:
            toc_content: Extracted TOC content
            
        Returns:
            'yes' if page numbers detected, 'no' otherwise
        """
        print('start detect_page_index')
        prompt = f"""
        You will be given a table of contents.

        Your job is to detect if there are page numbers/indices given within the table of contents.

        Given text: {toc_content}

        Reply format:
        {{
            "thinking": <why do you think there are page numbers/indices given within the table of contents>
            "page_index_given_in_toc": "<yes or no>"
        }}
        Directly return the final JSON structure. Do not output anything else."""

        response = await self.llm.chat_completion(prompt)
        json_content = self.llm.extract_json(response)
        return json_content['page_index_given_in_toc']
    
    @staticmethod
    def _transform_dots_to_colon(text: str) -> str:
        """Transform sequences of dots to colons for easier parsing."""
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    async def extract_from_pages(
        self,
        page_list: List[tuple],
        toc_page_list: List[int]
    ) -> Dict[str, str]:
        """
        Extract TOC from multiple pages and detect page numbers.
        
        Args:
            page_list: List of (page_text, token_count) tuples
            toc_page_list: Indices of pages containing TOC
            
        Returns:
            Dictionary with 'toc_content' and 'page_index_given_in_toc'
        """
        toc_content = ""
        for page_index in toc_page_list:
            toc_content += page_list[page_index][0]
        
        toc_content = self._transform_dots_to_colon(toc_content)
        has_page_index = await self.detect_page_numbers(toc_content)
        
        return {
            "toc_content": toc_content,
            "page_index_given_in_toc": has_page_index
        }
