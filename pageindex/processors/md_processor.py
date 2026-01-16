"""
Markdown Document Processor.

High-level orchestrator for Markdown document processing pipeline.
"""

import asyncio
from typing import Dict, Optional
from ..llm.llm_client_base import BaseLLMClient
from ..core import MarkdownParser, TreeOptimizer, MarkdownTreeBuilder


class MarkdownProcessor:
    """
    Orchestrates the complete Markdown processing pipeline.
    
    Coordinates parsing, optimization (thinning), tree building, and
    optional summarization to produce structured document representation.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config):
        """
        Initialize Markdown processor.
        
        Args:
            llm_client: LLM client instance
            config: Configuration object
        """
        self.llm = llm_client
        self.config = config
        
        # Initialize components
        self.parser = MarkdownParser(llm_client)
        self.optimizer = TreeOptimizer(llm_client)
        self.tree_builder = MarkdownTreeBuilder()
    
    async def process(
        self,
        md_path: str,
        if_thinning: bool = False,
        min_token_threshold: int = 5000,
        if_add_node_summary: bool = False,
        if_add_node_text: bool = True
    ) -> Dict:
        """
        Main processing entry point.
        
        Args:
            md_path: Path to markdown file
            if_thinning: Whether to apply tree thinning
            min_token_threshold: Minimum tokens per node for thinning
            if_add_node_summary: Whether to generate summaries
            if_add_node_text: Whether to include text in output
            
        Returns:
            Structured document tree
        """
        # Read markdown file
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Extract nodes
        node_list, markdown_lines = self.parser.extract_nodes(md_content)
        
        # Extract text content
        node_list = self.parser.extract_text_content(node_list, markdown_lines)
        
        # Count tokens
        node_list = self.parser.count_tokens_for_nodes(node_list)
        
        # Optional: Apply tree thinning
        if if_thinning:
            node_list = self.optimizer.thin_tree(node_list, min_token_threshold)
        
        # Build tree
        tree = self.tree_builder.build_from_nodes(node_list)
        
        # Optional: Generate summaries
        if if_add_node_summary:
            tree = await self._generate_summaries(tree)
        
        # Optional: Remove text
        if not if_add_node_text:
            tree = self._remove_text_field(tree)
        
        # Clean tree
        tree = self.tree_builder.clean_for_output(tree)
        
        # Get document name
        import os
        doc_name = os.path.basename(md_path).replace('.md', '')
        
        return {
            'doc_name': doc_name,
            'structure': tree
        }
    
    async def _generate_summaries(self, tree: list) -> list:
        """
        Generate summaries for all nodes in tree.
        
        Args:
            tree: Tree structure
            
        Returns:
            Tree with summaries added
        """
        async def generate_node_summary(node):
            """Generate summary for a single node."""
            prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

            Partial Document Text: {node['text']}
            
            Directly return the description, do not include any other text.
            """
            response = await self.llm.chat_completion(prompt)
            return response
        
        async def process_node(node):
            """Recursively process node and children."""
            # Generate summary for this node
            if 'text' in node:
                node['summary'] = await generate_node_summary(node)
            
            # Process children
            if 'nodes' in node and node['nodes']:
                tasks = [process_node(child) for child in node['nodes']]
                await asyncio.gather(*tasks)
            
            return node
        
        # Process all root nodes
        tasks = [process_node(node) for node in tree]
        await asyncio.gather(*tasks)
        
        return tree
    
    @staticmethod
    def _remove_text_field(tree: list) -> list:
        """
        Remove text field from all nodes.
        
        Args:
            tree: Tree structure
            
        Returns:
            Tree without text fields
        """
        def remove_from_node(node):
            node.pop('text', None)
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    remove_from_node(child)
            return node
        
        return [remove_from_node(node) for node in tree]
