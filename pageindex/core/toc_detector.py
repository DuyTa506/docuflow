"""
TOC Detection Component.

Responsible for detecting whether pages contain table of contents.
"""

from typing import List, Optional
from ..llm.llm_client_base import BaseLLMClient


class TOCDetector:
    """
    Detects Table of Contents in document pages using LLM.
    
    This class encapsulates all logic related to identifying which pages
    in a document contain the table of contents.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize TOC detector.
        
        Args:
            llm_client: LLM client instance for making API calls
        """
        self.llm = llm_client
    
    async def detect_single_page(self, content: str) -> str:
        """
        Detect if a single page contains TOC.
        
        Args:
            content: Text content of the page
            
        Returns:
            'yes' if TOC detected, 'no' otherwise
        """
        prompt = f"""
        Your job is to detect if there is a table of content provided in the given text.

        Given text: {content}

        return the following JSON format:
        {{
            "thinking": <why do you think there is a table of content in the given text>
            "toc_detected": "<yes or no>",
        }}

        Directly return the final JSON structure. Do not output anything else.
        Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

        response = await self.llm.chat_completion(prompt)
        json_content = self.llm.extract_json(response)
        return json_content['toc_detected']
    
    def find_toc_pages(
        self,
        page_list: List[tuple],
        start_index: int = 0,
        max_pages: int = 20,
        logger = None
    ) -> List[int]:
        """
        Find all pages that contain TOC.
        
        Args:
            page_list: List of (page_text, token_count) tuples
            start_index: Index to start searching from
            max_pages: Maximum number of pages to check
            logger: Optional logger instance
            
        Returns:
            List of page indices that contain TOC
        """
        import asyncio
        
        print('start find_toc_pages')
        last_page_is_yes = False
        toc_page_list = []
        i = start_index
        
        while i < len(page_list):
            # Only check beyond max_pages if we're still finding TOC pages
            if i >= max_pages and not last_page_is_yes:
                break
            
            detected_result = asyncio.run(
                self.detect_single_page(page_list[i][0])
            )
            
            if detected_result == 'yes':
                if logger:
                    logger.info(f'Page {i} has toc')
                toc_page_list.append(i)
                last_page_is_yes = True
            elif detected_result == 'no' and last_page_is_yes:
                if logger:
                    logger.info(f'Found the last page with toc: {i-1}')
                break
            i += 1
        
        if not toc_page_list and logger:
            logger.info('No toc found')
            
        return toc_page_list
