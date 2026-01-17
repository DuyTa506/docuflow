"""
TOC Verification Component.

Responsible for verifying TOC accuracy and fixing incorrect entries.
"""

import asyncio
import random
from typing import List, Dict, Tuple
from ..llm.llm_client_base import BaseLLMClient
from .. import utils


class TOCVerifier:
    """
    Verifies and fixes table of contents entries.
    
    Uses LLM to check if section titles actually appear on claimed pages
    and attempts to fix incorrect mappings.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize TOC verifier.
        
        Args:
            llm_client: LLM client instance for making API calls
        """
        self.llm = llm_client
    
    async def check_title_appearance(
        self,
        item: Dict,
        page_list: List[tuple],
        start_index: int = 1
    ) -> Dict:
        """
        Check if a title appears on its claimed page.
        
        Args:
            item: TOC item with title and physical_index
            page_list: List of (page_text, token_count) tuples
            start_index: Starting page index
            
        Returns:
            Dict with answer ('yes'/'no'), title, and page_number
        """
        title = item['title']
        if 'physical_index' not in item or item['physical_index'] is None:
            return {
                'list_index': item.get('list_index'),
                'answer': 'no',
                'title': title,
                'page_number': None
            }
        
        page_number = item['physical_index']
        page_text = page_list[page_number - start_index][0]
        
        prompt = f"""
        Your job is to check if the given section appears or starts in the given page_text.

        Note: do fuzzy matching, ignore any space inconsistency in the page_text.

        The given section title is {title}.
        The given page_text is {page_text}.
        
        Reply format:
        {{
            "thinking": <why do you think the section appears or starts in the page_text>
            "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
        }}
        Directly return the final JSON structure. Do not output anything else."""

        response = await self.llm.chat_completion(prompt)
        response = self.llm.extract_json(response)
        
        answer = response.get('answer', 'no')
        return {
            'list_index': item['list_index'],
            'answer': answer,
            'title': title,
            'page_number': page_number
        }
    
    async def verify_toc(
        self,
        page_list: List[tuple],
        toc_items: List[Dict],
        start_index: int = 1,
        sample_size: int = None
    ) -> Tuple[float, List[Dict]]:
        """
        Verify TOC accuracy by checking a sample of items.
        
        Args:
            page_list: List of (page_text, token_count) tuples
            toc_items: TOC items to verify
            start_index: Starting page index
            sample_size: Number of items to check (None = check all)
            
        Returns:
            Tuple of (accuracy_score, list_of_incorrect_items)
        """
        print('start verify_toc')
        
        # Find last valid physical_index
        last_physical_index = None
        for item in reversed(toc_items):
            if item.get('physical_index') is not None:
                last_physical_index = item['physical_index']
                break
        
        # Early return if not enough data
        if last_physical_index is None or last_physical_index < len(page_list) / 2:
            return 0, []
        
        # Determine which items to check
        if sample_size is None:
            print('check all items')
            sample_indices = range(0, len(toc_items))
        else:
            sample_size = min(sample_size, len(toc_items))
            print(f'check {sample_size} items')
            sample_indices = random.sample(range(0, len(toc_items)), sample_size)
        
        # Prepare items for checking
        indexed_sample_list = []
        for idx in sample_indices:
            item = toc_items[idx]
            if item.get('physical_index') is not None:
                item_with_index = item.copy()
                item_with_index['list_index'] = idx
                indexed_sample_list.append(item_with_index)
        
        # Run checks concurrently
        tasks = [
            self.check_title_appearance(item, page_list, start_index)
            for item in indexed_sample_list
        ]
        results = await asyncio.gather(*tasks)
        
        # Process results
        correct_count = 0
        incorrect_results = []
        for result in results:
            if result['answer'] == 'yes':
                correct_count += 1
            else:
                incorrect_results.append(result)
        
        # Calculate accuracy
        checked_count = len(results)
        accuracy = correct_count / checked_count if checked_count > 0 else 0
        print(f"accuracy: {accuracy*100:.2f}%")
        
        return accuracy, incorrect_results
    
    async def fix_single_item(
        self,
        section_title: str,
        content: str
    ) -> int:
        """
        Fix a single TOC item by finding its correct page.
        
        Args:
            section_title: Title of the section
            content: Document content with physical_index tags
            
        Returns:
            Correct physical_index as integer
        """
        prompt = """
        You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

        The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

        Reply in a JSON format:
        {
            "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
            "physical_index": "<physical_index_X>" (keep the format)
        }
        Directly return the final JSON structure. Do not output anything else."""

        prompt = prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
        response = await self.llm.chat_completion(prompt)
        json_content = self.llm.extract_json(response)
        return utils.convert_physical_index_to_int(json_content['physical_index'])
    
    async def fix_incorrect_items(
        self,
        toc: List[Dict],
        page_list: List[tuple],
        incorrect_items: List[Dict],
        start_index: int = 1,
        logger = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Attempt to fix incorrect TOC items.
        
        Args:
            toc: Full TOC list
            page_list: List of (page_text, token_count) tuples
            incorrect_items: Items identified as incorrect
            start_index: Starting page index
            logger: Optional logger
            
        Returns:
            Tuple of (updated_toc, still_incorrect_items)
        """
        print(f'start fix_incorrect_toc with {len(incorrect_items)} incorrect results')
        incorrect_indices = {result['list_index'] for result in incorrect_items}
        end_index = len(page_list) + start_index - 1
        
        async def process_and_check_item(incorrect_item):
            list_index = incorrect_item['list_index']
            
            # Validate index
            if list_index < 0 or list_index >= len(toc):
                return {
                    'list_index': list_index,
                    'title': incorrect_item['title'],
                    'physical_index': incorrect_item.get('physical_index'),
                    'is_valid': False
                }
            
            # Find previous correct item
            prev_correct = start_index - 1
            for i in range(list_index - 1, -1, -1):
                if i not in incorrect_indices and i < len(toc):
                    physical_index = toc[i].get('physical_index')
                    if physical_index is not None:
                        prev_correct = physical_index
                        break
            
            # Find next correct item
            next_correct = end_index
            for i in range(list_index + 1, len(toc)):
                if i not in incorrect_indices and i < len(toc):
                    physical_index = toc[i].get('physical_index')
                    if physical_index is not None:
                        next_correct = physical_index
                        break
            
            # Build content range
            page_contents = []
            for page_index in range(prev_correct, next_correct + 1):
                list_idx = page_index - start_index
                if 0 <= list_idx < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_idx][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
            
            content_range = ''.join(page_contents)
            
            # Fix the item
            physical_index_int = await self.fix_single_item(incorrect_item['title'], content_range)
            
            # Verify the fix
            check_item = incorrect_item.copy()
            check_item['physical_index'] = physical_index_int
            check_result = await self.check_title_appearance(check_item, page_list, start_index)
            
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': physical_index_int,
                'is_valid': check_result['answer'] == 'yes'
            }
        
        # Process all incorrect items concurrently
        tasks = [process_and_check_item(item) for item in incorrect_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        results = [r for r in results if not isinstance(r, Exception)]
        
        # Update TOC and collect still-incorrect items
        invalid_results = []
        for result in results:
            if result['is_valid']:
                list_idx = result['list_index']
                if 0 <= list_idx < len(toc):
                    toc[list_idx]['physical_index'] = result['physical_index']
            else:
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        
        if logger:
            logger.info(f'invalid_results: {invalid_results}')
        
        return toc, invalid_results
    
    async def fix_with_retries(
        self,
        toc: List[Dict],
        page_list: List[tuple],
        incorrect_items: List[Dict],
        start_index: int = 1,
        max_attempts: int = 3,
        logger = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Fix incorrect items with retry logic.
        
        Args:
            toc: Full TOC list
            page_list: List of (page_text, token_count) tuples
            incorrect_items: Items to fix
            start_index: Starting page index
            max_attempts: Maximum retry attempts
            logger: Optional logger
            
        Returns:
            Tuple of (updated_toc, still_incorrect_after_retries)
        """
        print('start fix_incorrect_toc with retries')
        current_toc = toc
        current_incorrect = incorrect_items
        
        for attempt in range(max_attempts):
            if not current_incorrect:
                break
            
            print(f"Fixing {len(current_incorrect)} incorrect results (attempt {attempt + 1})")
            current_toc, current_incorrect = await self.fix_incorrect_items(
                current_toc, page_list, current_incorrect, start_index, logger
            )
        
        if logger:
            logger.info(f"Maximum fix attempts reached" if current_incorrect else "All items fixed")
        
        return current_toc, current_incorrect
