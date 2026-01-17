"""
Tree Indexing Service

Handles building and storing tree indices for documents using PageIndex.
Supports both standard markdown-based and spatial-enhanced indexing.
"""
import tempfile
import os
from typing import Optional, Dict
from sqlalchemy.orm import Session

from data.db_models import TreeIndex, TreeNode, Page
from .storage_service import DocumentStorageService


class TreeIndexingService:
    """Service for building and storing PageIndex tree structures."""
    
    def __init__(
        self,
        session: Session,
        llm_provider: str = "openai",
        model: str = "gpt-4o-2024-11-20"
    ):
        """
        Initialize tree indexing service.
        
        Args:
            session: Database session
            llm_provider: LLM provider ('openai' or 'ollama')
            model: Model name
        """
        self.session = session
        self.storage = DocumentStorageService(session)
        self.llm_provider = llm_provider
        self.model = model
    
    async def build_tree_index(
        self,
        document_id: str,
        if_thinning: bool = True,
        min_token_threshold: int = 5000,
        if_add_node_summary: str = "yes",
        summary_token_threshold: int = 200,
        if_add_doc_description: str = "no",
        if_add_node_text: str = "no",
        if_add_node_id: str = "yes",
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: int = 300,
        **kwargs
    ) -> Dict:
        """
        Build PageIndex tree structure from stored document.
        
        Args:
            document_id: ID of document to index
            if_thinning: Whether to apply tree thinning
            min_token_threshold: Minimum token threshold for thinning
            if_add_node_summary: Whether to add node summaries
            summary_token_threshold: Token threshold for summaries
            if_add_doc_description: Whether to add document description
            if_add_node_text: Whether to add node text
            if_add_node_id: Whether to add node IDs
            ollama_base_url: Ollama server URL (if using Ollama)
            ollama_timeout: Ollama timeout in seconds
            **kwargs: Additional PageIndex parameters
        
        Returns:
            Dictionary with tree_index_id, node_count, max_depth
        """
        # Get document
        document = self.storage.get_document(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Get markdown content
        markdown = self.storage.get_document_markdown(document_id)
        
        # Create temporary markdown file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.md', text=True)
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            # Import PageIndex async function directly
            from pageindex.entry_points import _md_to_tree_async
            
            # Build tree using PageIndex (await async function)
            tree_result = await _md_to_tree_async(
                md_path=temp_path,
                if_thinning=if_thinning,
                min_token_threshold=min_token_threshold,
                if_add_node_summary=if_add_node_summary,
                summary_token_threshold=summary_token_threshold,
                model=self.model,
                if_add_doc_description=if_add_doc_description,
                if_add_node_text=if_add_node_text,
                if_add_node_id=if_add_node_id,
                llm_provider=self.llm_provider,
                ollama_base_url=ollama_base_url,
                ollama_timeout=ollama_timeout
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Store tree configuration
        config = {
            'llm_provider': self.llm_provider,
            'model': self.model,
            'if_thinning': if_thinning,
            'min_token_threshold': min_token_threshold,
            'if_add_node_summary': if_add_node_summary,
            'summary_token_threshold': summary_token_threshold,
            'if_add_doc_description': if_add_doc_description,
            'if_add_node_text': if_add_node_text,
            'if_add_node_id': if_add_node_id
        }
        if self.llm_provider == 'ollama':
            config['ollama_base_url'] = ollama_base_url
            config['ollama_timeout'] = ollama_timeout
        
        # Save tree index to database
        tree_index = self.storage.save_tree_index(
            document_id=document_id,
            tree_data=tree_result,
            config=config
        )
        
        # Calculate statistics
        node_count = len(tree_index.tree_nodes)
        max_depth = self._calculate_tree_depth(tree_result)
        
        return {
            'tree_index_id': tree_index.id,
            'document_id': document_id,
            'node_count': node_count,
            'max_depth': max_depth,
            'config': config
        }
    
    def _calculate_tree_depth(self, node: Dict, current_depth: int = 0) -> int:
        """Recursively calculate tree depth."""
        children = node.get('children', node.get('child_nodes', []))
        if not children:
            return current_depth
        
        max_child_depth = current_depth
        for child in children:
            child_depth = self._calculate_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def get_tree_index(self, document_id: str) -> Optional[Dict]:
        """
        Get tree index for a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            Dictionary with tree data or None
        """
        tree_index = self.storage.get_tree_index(document_id)
        if not tree_index:
            return None
        
        return {
            'tree_index_id': tree_index.id,
            'document_id': tree_index.document_id,
            'tree_data': tree_index.tree_data,
            'config': tree_index.config,
            'created_at': tree_index.created_at.isoformat(),
            'node_count': len(tree_index.tree_nodes)
        }
    
    async def _apply_pageindex_llm_processing(
        self,
        tree: Dict,
        markdown: str,
        if_add_node_summary: str = "no",
        summary_token_threshold: int = 200,
        if_add_doc_description: str = "no",
        if_add_node_text: str = "no",
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: int = 300
    ) -> Dict:
        """
        Apply PageIndex LLM post-processing to spatial tree.
        
        Adds:
        - Node summaries (LLM-generated)
        - Document description (LLM-generated)
        - Node text (extracted from markdown)
        
        Uses spatial thinning, NOT PageIndex thinning.
        """
        # Initialize LLM client
        if self.llm_provider == "ollama":
            from ollama import AsyncClient
            llm_client = AsyncClient(host=ollama_base_url, timeout=ollama_timeout)
        else:
            from openai import AsyncOpenAI
            import os
            llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Recursively process nodes
        async def process_node(node: Dict):
            """Process a single node and its children."""
            
            # Add node text if requested
            if if_add_node_text == "yes":
                node_content = node.get('content', '')
                if node_content:
                    node['text'] = node_content
            
            # Add node summary if requested
            if if_add_node_summary == "yes" and node.get('content'):
                content = node['content']
                
                # Skip if content too short
                if len(content) < 50:
                    node['summary'] = content
                else:
                    # Generate summary with LLM
                    try:
                        summary_prompt = f"""Summarize in under {summary_token_threshold} tokens:

{content[:2000]}

Summary:"""
                        
                        if self.llm_provider == "ollama":
                            response = await llm_client.chat(
                                model=self.model,
                                messages=[{"role": "user", "content": summary_prompt}],
                                options={"num_predict": summary_token_threshold}
                            )
                            summary = response['message']['content']
                        else:
                            response = await llm_client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": summary_prompt}],
                                max_tokens=summary_token_threshold
                            )
                            summary = response.choices[0].message.content
                        
                        node['summary'] = summary.strip()
                    except Exception as e:
                        node['summary'] = content[:200] + "..."
            
            # Process children recursively
            if 'children' in node and node['children']:
                for child in node['children']:
                    await process_node(child)
        
        # Process tree
        await process_node(tree)
        
        # Add document description if requested
        if if_add_doc_description == "yes":
            def collect_summaries(node: Dict, summaries: list):
                if node.get('title'):
                    summaries.append(f"- {node['title']}: {node.get('summary', 'N/A')}")
                if node.get('children'):
                    for child in node['children']:
                        collect_summaries(child, summaries)
            
            all_summaries = []
            collect_summaries(tree, all_summaries)
            
            if all_summaries:
                desc_prompt = f"""Write a brief overview (1-2 paragraphs):

{chr(10).join(all_summaries[:20])}

Overview:"""
                
                try:
                    if self.llm_provider == "ollama":
                        response = await llm_client.chat(
                            model=self.model,
                            messages=[{"role": "user", "content": desc_prompt}],
                            options={"num_predict": 300}
                        )
                        description = response['message']['content']
                    else:
                        response = await llm_client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": desc_prompt}],
                            max_tokens=300
                        )
                        description = response.choices[0].message.content
                    
                    tree['document_description'] = description.strip()
                except Exception as e:
                    tree['document_description'] = "Document overview unavailable"
        
        return tree
    
    async def build_enhanced_tree_index(
        self,
        document_id: str,
        use_spatial_metadata: bool = True,
        discover_implicit_sections: bool = True,
        spatial_weights: Optional[Dict[str, float]] = None,
        if_thinning: bool = False,
        if_add_node_summary: str = "no",
        **kwargs
    ) -> Dict:
        """
        Build tree index using both markdown and spatial metadata.
        
        This is an enhanced version that uses bounding boxes and grounding labels
        to improve hierarchy detection.
        
        Args:
            document_id: ID of document to index
            use_spatial_metadata: Whether to use spatial metadata for hierarchy
            discover_implicit_sections: Whether to find sections without markdown headers
            spatial_weights: Custom weights for spatial scoring
            if_thinning: Whether to apply tree thinning (PageIndex)
            if_add_node_summary: Whether to add node summaries (PageIndex)
            **kwargs: Additional PageIndex parameters
        
        Returns:
            Dictionary with tree_index_id, node_count, max_depth
        """
        # Get document
        document = self.storage.get_document(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Get markdown content
        markdown = self.storage.get_document_markdown(document_id)
        
        # Get layout elements with spatial metadata FROM DATABASE
        layout_elements_db = self.storage.get_document_elements(document_id)
        
        if use_spatial_metadata and layout_elements_db:
            # Re-parse markdown to get FULL text content using V2 parser
            # (Database elements don't have text_full field)
            from utils import extract_layout_coordinates_v2
            
            # Get first page to extract image dims (needed for V2 parser)
            first_page = self.session.query(Page).filter(
                Page.document_id == document_id
            ).first()
            
            if not first_page:
                raise ValueError("No pages found for document")
            
            img_width = first_page.image_width or 1600
            img_height = first_page.image_height or 2400
            
            # Re-parse to get elements with text_content AND text_full
            # Need raw OCR output - get from pages
            pages = self.session.query(Page).filter(
                Page.document_id == document_id
            ).order_by(Page.page_number).all()
            
            elements_list = []
            for page in pages:
                # Parse markdown content for this page with V2
                # (V2 extracts text from grounding tags in markdown)
                page_elements = extract_layout_coordinates_v2(
                    page.markdown_content,  # Raw markdown with grounding tags
                    img_width,
                    img_height,
                    page_number=page.page_number
                )
                elements_list.extend(page_elements)
            
            # Use NEW spatial-first tree builder WITH spatial thinning
            from spatial import build_spatial_tree
            
            tree_result = build_spatial_tree(
                layout_elements=elements_list,
                use_filters=True,
                use_zone_classification=True,
                use_reading_order=True,
                use_markdown_validation=discover_implicit_sections,
                use_adaptive_thresholds=True,
                use_thinning=if_thinning,  # Use spatial thinning (NOT PageIndex thinning)
                thinning_gap_multiplier=2.0,
                spatial_weights=spatial_weights
            )
            
            # Apply PageIndex LLM post-processing (summaries, descriptions)
            # Skip PageIndex thinning - already done by spatial thinning
            if if_add_node_summary == "yes" or if_add_doc_description == "yes":
                tree_result = await self._apply_pageindex_llm_processing(
                    tree=tree_result,
                    markdown=markdown,
                    if_add_node_summary=if_add_node_summary,
                    summary_token_threshold=kwargs.get('summary_token_threshold', 200),
                    if_add_doc_description=if_add_doc_description,
                    if_add_node_text=kwargs.get('if_add_node_text', 'no'),
                    ollama_base_url=kwargs.get('ollama_base_url', 'http://localhost:11434'),
                    ollama_timeout=kwargs.get('ollama_timeout', 300)
                )
            
            method = "spatial_first_with_llm" if (if_add_node_summary == "yes" or if_add_doc_description == "yes") else "spatial_first"
        else:
            # Fall back to standard PageIndex
            # Create temporary markdown file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.md', text=True)
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                
                # Import PageIndex async function
                from pageindex.entry_points import _md_to_tree_async
                
                # Build tree using PageIndex (await async function)
                tree_result = await _md_to_tree_async(
                    md_path=temp_path,
                    if_thinning=if_thinning,
                    if_add_node_summary=if_add_node_summary,
                    model=self.model,
                    llm_provider=self.llm_provider,
                    **kwargs
                )
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            method = "pageindex_standard"
        
        # Store tree configuration
        config = {
            'method': method,
            'use_spatial_metadata': use_spatial_metadata,
            'discover_implicit_sections': discover_implicit_sections,
            'spatial_weights': spatial_weights,
            'llm_provider': self.llm_provider,
            'model': self.model,
            'if_thinning': if_thinning,
            'if_add_node_summary': if_add_node_summary
        }
        
        # Save tree index to database
        tree_index = self.storage.save_tree_index(
            document_id=document_id,
            tree_data=tree_result,
            config=config
        )
        
        # Calculate statistics
        node_count = len(tree_index.tree_nodes)
        max_depth = self._calculate_tree_depth(tree_result)
        
        return {
            'tree_index_id': tree_index.id,
            'document_id': document_id,
            'node_count': node_count,
            'max_depth': max_depth,
            'method': method,
            'config': config
        }

