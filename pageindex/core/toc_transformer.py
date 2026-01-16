"""
TOC Transformation Component.

Responsible for transforming raw TOC text into structured JSON format.
"""

import json
from typing import List, Dict
from ..llm.llm_client_base import BaseLLMClient
from .. import utils


class TOCTransformer:
    """
    Transforms raw table of contents into structured JSON format.
    
    Uses LLM to parse and structure TOC data with proper hierarchy.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize TOC transformer.
        
        Args:
            llm_client: LLM client instance for making API calls
        """
        self.llm = llm_client
    
    async def transform_to_json(self, toc_content: str) -> List[Dict]:
        """
        Transform raw TOC content to structured JSON.
        
        Args:
            toc_content: Raw TOC text
            
        Returns:
            List of TOC items with structure, title, and page information
        """
        print('start toc_transformer')
        init_prompt = """
        You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

        structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

        The response should be in the following JSON format: 
        {
        table_of_contents: [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "page": <page number or None>,
            },
            ...
            ],
        }
        You should transform the full table of contents in one go.
        Directly return the final JSON structure, do not output anything else. """

        prompt = init_prompt + '\n Given table of contents\n:' + toc_content
        last_complete, finish_reason = await self.llm.chat_completion_with_finish_reason(prompt)
        
        if_complete = await self._check_transformation_complete(toc_content, last_complete)
        if if_complete == "yes" and finish_reason == "finished":
            last_complete = self.llm.extract_json(last_complete)
            cleaned_response = utils.convert_page_to_int(last_complete['table_of_contents'])
            return cleaned_response
        
        last_complete = self.llm.get_json_content(last_complete)
        while not (if_complete == "yes" and finish_reason == "finished"):
            position = last_complete.rfind('}')
            if position != -1:
                last_complete = last_complete[:position+2]
            
            continuation_prompt = f"""
            Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
            The response should be in the following JSON format: 

            The raw table of contents json structure is:
            {toc_content}

            The incomplete transformed table of contents json structure is:
            {last_complete}

            Please continue the json structure, directly output the remaining part of the json structure."""

            new_complete, finish_reason = await self.llm.chat_completion_with_finish_reason(continuation_prompt)

            if new_complete.startswith('```json'):
                new_complete = self.llm.get_json_content(new_complete)
                last_complete = last_complete + new_complete

            if_complete = await self._check_transformation_complete(toc_content, last_complete)

        last_complete = json.loads(last_complete)
        cleaned_response = utils.convert_page_to_int(last_complete['table_of_contents'])
        return cleaned_response
    
    async def _check_transformation_complete(self, content: str, toc: str) -> str:
        """
        Check if TOC transformation is complete.
        
        Args:
            content: Original TOC content
            toc: Transformed TOC
            
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
