"""
Helper method for PageIndex LLM processing integration.
Add this to TreeIndexingService class.
"""

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
    
    Args:
        tree: Tree structure from build_spatial_tree
        markdown: Full markdown content
        if_add_node_summary: Whether to add summaries
        summary_token_threshold: Max tokens for summaries
        if_add_doc_description: Whether to add doc description
        if_add_node_text: Whether to extract full text
        ollama_base_url: Ollama server URL
        ollama_timeout: Timeout for LLM calls
    
    Returns:
        Enhanced tree with LLM metadata
    """
    # Import PageIndex utilities
    from pageindex.core import markdown_tree_utils
    
    # Initialize LLM client
    if self.llm_provider == "ollama":
        from ollama import AsyncClient
        llm_client = AsyncClient(host=ollama_base_url, timeout=ollama_timeout)
    else:
        from openai import AsyncOpenAI
        import os
        llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Recursively process nodes
    async def process_node(node: Dict, depth: int = 0):
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
                    summary_prompt = f"""Summarize the following content in under {summary_token_threshold} tokens. Be concise and capture key points:

{content[:2000]}  # Limit context to avoid token overload

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
                    print(f"Warning: Failed to generate summary: {e}")
                    node['summary'] = content[:200] + "..."
        
        # Process children recursively
        if 'children' in node and node['children']:
            for child in node['children']:
                await process_node(child, depth + 1)
        
        return node
    
    # Process tree
    await process_node(tree)
    
    # Add document description if requested
    if if_add_doc_description == "yes":
        # Collect all node titles/summaries
        def collect_summaries(node: Dict, summaries: list):
            if node.get('title'):
                summaries.append(f"- {node['title']}: {node.get('summary', 'N/A')}")
            if node.get('children'):
                for child in node['children']:
                    collect_summaries(child, summaries)
        
        all_summaries = []
        collect_summaries(tree, all_summaries)
        
        if all_summaries:
            desc_prompt = f"""Based on the following document structure, write a brief overview (1-2 paragraphs) of what this document is about:

{chr(10).join(all_summaries[:20])}  # Limit to first 20 nodes

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
                print(f"Warning: Failed to generate document description: {e}")
                tree['document_description'] = "Document overview unavailable"
    
    return tree
