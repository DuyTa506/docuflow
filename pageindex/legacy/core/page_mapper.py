"""
Page Mapping Component.

Responsible for mapping logical page numbers to physical page indices in PDF.
"""

from typing import List, Dict, Optional
from ..llm.llm_client_base import BaseLLMClient
from .. import utils


class PageMapper:
    """
    Maps logical page numbers (from TOC) to physical page indices (in PDF file).
    
    Handles the offset calculation and application for accurate page mapping.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize page mapper.
        
        Args:
            llm_client: LLM client instance for making API calls
        """
        self.llm = llm_client
    
    async def find_physical_indices(
        self,
        toc: List[Dict],
        content: str
    ) -> List[Dict]:
        """
        Find physical page indices for TOC items using LLM.
        
        Args:
            toc: TOC items without physical indices
            content: Document content with physical_index tags
            
        Returns:
            TOC items with physical_index added
        """
        print('start toc_index_extractor')
        prompt = """
        You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

        The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

        The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

        The response should be in the following JSON format: 
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "physical_index": "<physical_index_X>" (keep the format)
            },
            ...
        ]

        Only add the physical_index to the sections that are in the provided pages.
        If the section is not in the provided pages, do not add the physical_index to it.
        Directly return the final JSON structure. Do not output anything else."""

        prompt = prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
        response = await self.llm.chat_completion(prompt)
        json_content = self.llm.extract_json(response)
        return json_content
    
    @staticmethod
    def extract_matching_pairs(
        toc_page: List[Dict],
        toc_physical: List[Dict],
        start_page_index: int
    ) -> List[Dict]:
        """
        Extract matching pairs of logical and physical page numbers.
        
        Args:
            toc_page: TOC with logical page numbers
            toc_physical: TOC with physical indices
            start_page_index: Starting page index
            
        Returns:
            List of matching pairs with title, page, and physical_index
        """
        pairs = []
        for phy_item in toc_physical:
            for page_item in toc_page:
                if phy_item.get('title') == page_item.get('title'):
                    physical_index = phy_item.get('physical_index')
                    if physical_index is not None and int(physical_index) >= start_page_index:
                        pairs.append({
                            'title': phy_item.get('title'),
                            'page': page_item.get('page'),
                            'physical_index': physical_index
                        })
        return pairs
    
    @staticmethod
    def calculate_offset(pairs: List[Dict]) -> Optional[int]:
        """
        Calculate page offset from matching pairs.
        
        Args:
            pairs: List of {page, physical_index} pairs
            
        Returns:
            Most common offset, or None if no valid pairs
        """
        differences = []
        for pair in pairs:
            try:
                physical_index = pair['physical_index']
                page_number = pair['page']
                difference = physical_index - page_number
                differences.append(difference)
            except (KeyError, TypeError):
                continue
        
        if not differences:
            return None
        
        # Find most common offset
        difference_counts = {}
        for diff in differences:
            difference_counts[diff] = difference_counts.get(diff, 0) + 1
        
        most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
        return most_common
    
    @staticmethod
    def apply_offset(toc_data: List[Dict], offset: int) -> List[Dict]:
        """
        Apply calculated offset to TOC data.
        
        Args:
            toc_data: TOC with logical page numbers
            offset: Offset to apply
            
        Returns:
            TOC with physical_index = page + offset
        """
        for i in range(len(toc_data)):
            if toc_data[i].get('page') is not None and isinstance(toc_data[i]['page'], int):
                toc_data[i]['physical_index'] = toc_data[i]['page'] + offset
                del toc_data[i]['page']
        
        return toc_data
