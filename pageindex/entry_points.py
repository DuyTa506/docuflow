"""
Entry points for PageIndex using new architecture.

Provides backward-compatible API while using clean modular components.
"""

import asyncio
from typing import Dict, Optional
from types import SimpleNamespace

from .llm import LLMClientFactory
from .processors import PDFProcessor, MarkdownProcessor
from .utils import ConfigLoader, get_page_tokens, get_pdf_name
from .core import TreeOptimizer, MarkdownParser, MarkdownTreeBuilder

# Re-export config for backward compatibility
from types import SimpleNamespace as config


async def _page_index_main_async(pdf_path: str, opt) -> Dict:
    """
    Async implementation of PDF processing using new architecture.
    
    Args:
        pdf_path: Path to PDF file
        opt: Configuration object
        
    Returns:
        Document structure dictionary
    """
    # Initialize LLM client
    provider = getattr(opt, 'llm_provider', 'openai')
    model = getattr(opt, 'model', 'gpt-4o-2024-11-20')
    
    kwargs = {}
    if provider == 'ollama':
        kwargs['ollama_base_url'] = getattr(opt, 'ollama_base_url', 'http://localhost:11434')
        kwargs['ollama_timeout'] = getattr(opt, 'ollama_timeout', 300)
    
    llm_client = LLMClientFactory.create_client(
        provider=provider,
        model=model,
        **kwargs
    )
    
    # Create processor and process
    processor = PDFProcessor(llm_client, opt)
    result = await processor.process(pdf_path, opt=opt)
    
    return result


def page_index_main(pdf_path: str, opt) -> Dict:
    """
    Main entry point for PDF processing.
    
    Maintains backward compatibility with legacy API while using new architecture.
    
    Args:
        pdf_path: Path to PDF file
        opt: Configuration object with settings
        
    Returns:
        Document structure dictionary with 'doc_name' and 'structure'
    """
    return asyncio.run(_page_index_main_async(pdf_path, opt))


async def _md_to_tree_async(
    md_path: str,
    if_thinning: bool = False,
    min_token_threshold: int = 5000,
    if_add_node_summary: str = "yes",
    summary_token_threshold: int = 200,
    model: str = "gpt-4o-2024-11-20",
    if_add_doc_description: str = "no",
    if_add_node_text: str = "no",
    if_add_node_id: str = "yes",
    llm_provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: int = 300
) -> Dict:
    """
    Async implementation of Markdown processing using new architecture.
    
    Args:
        md_path: Path to markdown file
        if_thinning: Whether to apply tree thinning
        min_token_threshold: Minimum token threshold for thinning
        if_add_node_summary: Whether to add summaries
        summary_token_threshold: Token threshold for summaries
        model: Model name
        if_add_doc_description: Whether to add document description
        if_add_node_text: Whether to add node text
        if_add_node_id: Whether to add node IDs
        llm_provider: LLM provider ('openai' or 'ollama')
        ollama_base_url: Ollama server URL
        ollama_timeout: Ollama timeout in seconds
        
    Returns:
        Document structure dictionary
    """
    # Initialize LLM client
    kwargs = {}
    if llm_provider == 'ollama':
        kwargs['ollama_base_url'] = ollama_base_url
        kwargs['ollama_timeout'] = ollama_timeout
    
    llm_client = LLMClientFactory.create_client(
        provider=llm_provider,
        model=model,
        **kwargs
    )
    
    # Create processor
    processor = MarkdownProcessor(llm_client, model)
    
    # Process markdown - only pass supported parameters
    # Note: MarkdownProcessor doesn't support summary_token_threshold, 
    # if_add_doc_description, or if_add_node_id yet
    result = await processor.process(
        md_path=md_path,
        if_thinning=if_thinning,
        min_token_threshold=min_token_threshold,
        if_add_node_summary=(if_add_node_summary == "yes"),
        if_add_node_text=(if_add_node_text == "yes")
    )

    
    return result


def md_to_tree(
    md_path: str,
    if_thinning: bool = False,
    min_token_threshold: int = 5000,
    if_add_node_summary: str = "yes",
    summary_token_threshold: int = 200,
    model: str = "gpt-4o-2024-11-20",
    if_add_doc_description: str = "no",
    if_add_node_text: str = "no",
    if_add_node_id: str = "yes",
    llm_provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: int = 300
) -> Dict:
    """
    Process markdown file to tree structure.
    
    Maintains backward compatibility with legacy API while using new architecture.
    
    Args:
        md_path: Path to markdown file
        if_thinning: Whether to apply tree thinning
        min_token_threshold: Minimum token threshold for thinning
        if_add_node_summary: Whether to add summaries
        summary_token_threshold: Token threshold for summaries
        model: Model name
        if_add_doc_description: Whether to add document description
        if_add_node_text: Whether to add node text
        if_add_node_id: Whether to add node IDs
        llm_provider: LLM provider ('openai' or 'ollama')
        ollama_base_url: Ollama server URL
        ollama_timeout: Ollama timeout in seconds
        
    Returns:
        Document structure dictionary
    """
    return asyncio.run(_md_to_tree_async(
        md_path=md_path,
        if_thinning=if_thinning,
        min_token_threshold=min_token_threshold,
        if_add_node_summary=if_add_node_summary,
        summary_token_threshold=summary_token_threshold,
        model=model,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
        if_add_node_id=if_add_node_id,
        llm_provider=llm_provider,
        ollama_base_url=ollama_base_url,
        ollama_timeout=ollama_timeout
    ))


def page_index(
    doc,
    model: Optional[str] = None,
    toc_check_page_num: Optional[int] = None,
    max_page_num_each_node: Optional[int] = None,
    max_token_num_each_node: Optional[int] = None,
    if_add_node_id: Optional[str] = None,
    if_add_node_summary: Optional[str] = None,
    if_add_doc_description: Optional[str] = None,
    if_add_node_text: Optional[str] = None,
    llm_provider: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_timeout: Optional[int] = None
) -> Dict:
    """
    Convenient wrapper function for PDF processing.
    
    Maintains backward compatibility with legacy API.
    
    Args:
        doc: Path to PDF file
        model: Model name
        toc_check_page_num: Number of pages to check for TOC
        max_page_num_each_node: Max pages per node
        max_token_num_each_node: Max tokens per node
        if_add_node_id: Whether to add node IDs
        if_add_node_summary: Whether to add summaries
        if_add_doc_description: Whether to add document description
        if_add_node_text: Whether to add node text
        llm_provider: LLM provider ('openai' or 'ollama')
        ollama_base_url: Ollama server URL
        ollama_timeout: Ollama timeout in seconds
        
    Returns:
        Document structure dictionary
    """
    # Build user options dict
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    
    # Load config with defaults
    opt = ConfigLoader().load(user_opt)
    
    return page_index_main(doc, opt)
