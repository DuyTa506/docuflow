"""
PDF Document Processor.

High-level orchestrator for PDF document processing pipeline.
"""

import asyncio
from typing import Dict, List, Optional
from ..llm.llm_client_base import BaseLLMClient
from ..core import (
    TOCDetector,
    TOCExtractor,
    TOCTransformer,
    PageMapper,
    TOCVerifier,
    TreeBuilder
)
from .. import utils


class PDFProcessor:
    """
    Orchestrates the complete PDF processing pipeline.
    
    Coordinates TOC detection, extraction, transformation, verification,
    and tree building to produce structured document representation.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config):
        """
        Initialize PDF processor.
        
        Args:
            llm_client: LLM client instance
            config: Configuration object
        """
        self.llm = llm_client
        self.config = config
        
        # Initialize components
        self.detector = TOCDetector(llm_client)
        self.extractor = TOCExtractor(llm_client)
        self.transformer = TOCTransformer(llm_client)
        self.mapper = PageMapper(llm_client)
        self.verifier = TOCVerifier(llm_client)
        self.tree_builder = TreeBuilder()
    
    async def process_toc_with_page_numbers(
        self,
        toc_content: str,
        toc_page_list: List[int],
        page_list: List[tuple],
        logger = None
    ) -> List[Dict]:
        """
        Process TOC that includes page numbers.
        
        Args:
            toc_content: Extracted TOC text
            toc_page_list: Pages containing TOC
            page_list: All document pages
            logger: Optional logger
            
        Returns:
            TOC with physical indices mapped
        """
        # Transform to JSON
        toc_json = await self.transformer.transform_to_json(toc_content)
        
        # Get sample pages for mapping
        toc_check_pages = getattr(self.config, 'toc_check_page_num', 20)
        sample_pages = utils.get_text_of_pdf_pages_with_labels(page_list, 1, min(toc_check_pages, len(page_list)))
        
        # Find physical indices for sample
        toc_with_physical = await self.mapper.find_physical_indices(toc_json, sample_pages)
        toc_with_physical = [item for item in toc_with_physical if item.get('physical_index') is not None]
        
        # Convert to int
        for item in toc_with_physical:
            item['physical_index'] = utils.convert_physical_index_to_int(item['physical_index'])
        
        # Extract matching pairs and calculate offset
        pairs = self.mapper.extract_matching_pairs(toc_json, toc_with_physical, 1)
        offset = self.mapper.calculate_offset(pairs)
        
        if offset is None:
            raise Exception('No valid page offset found')
        
        # Apply offset to all items
        toc_with_page_number = self.mapper.apply_offset(toc_json, offset)
        
        return toc_with_page_number
    
    async def process_toc_no_page_numbers(
        self,
        toc_content: str,
        page_list: List[tuple],
        logger = None
    ) -> List[Dict]:
        """
        Process TOC without page numbers.
        
        Args:
            toc_content: Extracted TOC text
            page_list: All document pages
            logger: Optional logger
            
        Returns:
            TOC with physical indices found via search
        """
        # Transform to JSON  
        toc_json = await self.transformer.transform_to_json(toc_content)
        
        # Build content with physical tags
        content = utils.get_text_of_pdf_pages_with_labels(page_list, 1, len(page_list))
        
        # Find indices using LLM
        toc_with_physical = await self.mapper.find_physical_indices(toc_json, content)
        
        # Convert to int
        for item in toc_with_physical:
            if item.get('physical_index') is not None:
                item['physical_index'] = utils.convert_physical_index_to_int(item['physical_index'])
        
        return toc_with_physical
    
    async def meta_processor(
        self,
        page_list: List[tuple],
        mode: str = None,
        toc_content: str = None,
        toc_page_list: List[int] = None,
        start_index: int = 1,
        logger = None
    ) -> List[Dict]:
        """
        Meta-processor with fallback strategies.
        
        Tries different processing modes with verification and fixing.
        
        Args:
            page_list: All document pages
            mode: Processing mode
            toc_content: TOC text
            toc_page_list: Pages with TOC
            start_index: Starting page index
            logger: Optional logger
            
        Returns:
            Verified and fixed TOC
        """
        print(f'Processing mode: {mode}')
        print(f'Start index: {start_index}')
        
        # Process based on mode
        if mode == 'process_toc_with_page_numbers':
            toc_with_page_number = await self.process_toc_with_page_numbers(
                toc_content, toc_page_list, page_list, logger
            )
        elif mode == 'process_toc_no_page_numbers':
            toc_with_page_number = await self.process_toc_no_page_numbers(
                toc_content, page_list, logger
            )
        else:
            # No TOC found - would need to generate from scratch
            # This is complex logic that could be implemented later
            raise NotImplementedError('No TOC processing mode not yet implemented in new architecture')
        
        # Filter out None indices
        toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]
        
        # Validate indices
        toc_with_page_number = self._validate_and_truncate_indices(
            toc_with_page_number, len(page_list), start_index, logger
        )
        
        # Verify
        accuracy, incorrect_results = await self.verifier.verify_toc(
            page_list, toc_with_page_number, start_index
        )
        
        if logger:
            logger.info({
                'mode': mode,
                'accuracy': accuracy,
                'incorrect_count': len(incorrect_results)
            })
        
        # Return if perfect
        if accuracy == 1.0 and len(incorrect_results) == 0:
            return toc_with_page_number
        
        # Try to fix if accuracy is decent
        if accuracy > 0.6 and len(incorrect_results) > 0:
            toc_with_page_number, _ = await self.verifier.fix_with_retries(
                toc_with_page_number, page_list, incorrect_results,
                start_index, max_attempts=3, logger=logger
            )
            return toc_with_page_number
        
        # Fallback to next mode
        if mode == 'process_toc_with_page_numbers':
            return await self.meta_processor(
                page_list, 'process_toc_no_page_numbers',
                toc_content, toc_page_list, start_index, logger
            )
        else:
            raise Exception('All processing modes failed')
    
    @staticmethod
    def _validate_and_truncate_indices(
        toc_items: List[Dict],
        total_pages: int,
        start_index: int,
        logger = None
    ) -> List[Dict]:
        """Validate and fix out-of-range indices."""
        result = []
        for item in toc_items:
            if item.get('physical_index') is not None:
                idx = item['physical_index']
                if idx >= start_index and idx <= total_pages:
                    result.append(item)
                else:
                    if logger:
                        logger.info(f"Removed out-of-range item: {item}")
        return result
    
    async def process(
        self,
        pdf_path: str,
        opt = None,
        logger = None
    ) -> Dict:
        """
        Main processing entry point.
        
        Args:
            pdf_path: Path to PDF file
            opt: Configuration options
            logger: Optional logger
            
        Returns:
            Structured document tree
        """
        # Get pages with tokens
        page_list = utils.get_page_tokens(pdf_path, model=self.config.model)
        
        # Detect TOC
        toc_page_list = self.detector.find_toc_pages(
            page_list,
            start_index=0,
            max_pages=getattr(self.config, 'toc_check_page_num', 20),
            logger=logger
        )
        
        # Determine mode
        if toc_page_list:
            # Extract TOC
            extraction_result = await self.extractor.extract_from_pages(page_list, toc_page_list)
            toc_content = extraction_result['toc_content']
            has_page_numbers = extraction_result['page_index_given_in_toc']
            
            mode = ('process_toc_with_page_numbers' if has_page_numbers == 'yes' 
                   else 'process_toc_no_page_numbers')
        else:
            mode = 'process_no_toc'
            toc_content = None
        
        # Process
        toc_result = await self.meta_processor(
            page_list, mode, toc_content, toc_page_list, 1, logger
        )
        
        # Build tree
        toc_result = self.tree_builder.add_preface_if_needed(toc_result)
        tree = self.tree_builder.build_from_flat_list(toc_result, len(page_list))
        
        return {
            'doc_name': utils.get_pdf_name(pdf_path),
            'structure': tree
        }
