"""
Tree Optimization Component.

Responsible for optimizing tree structure through thinning operations.
"""

from typing import List, Dict
from ..llm.llm_client_base import BaseLLMClient


class TreeOptimizer:
    """
    Optimizes tree structure by merging small nodes.
    
    Implements tree thinning to create meaningful chunks for RAG applications.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize tree optimizer.
        
        Args:
            llm_client: LLM client for token counting
        """
        self.llm = llm_client
    
    def thin_tree(self, node_list: List[Dict], min_token_threshold: int) -> List[Dict]:
        """
        Thin the tree by merging small nodes into their parents.
        
        Args:
            node_list: Flat list of nodes with text_token_count
            min_token_threshold: Minimum tokens per node
            
        Returns:
            Optimized node list with small nodes merged
        """
        def find_all_children(parent_index, parent_level, nodes):
            """Find all children indices."""
            children_indices = []
            
            for i in range(parent_index + 1, len(nodes)):
                current_level = nodes[i]['level']
                
                if current_level <= parent_level:
                    break
                
                children_indices.append(i)
            
            return children_indices
        
        result_list = node_list.copy()
        nodes_to_remove = set()
        
        # Process from bottom up
        for i in range(len(result_list) - 1, -1, -1):
            if i in nodes_to_remove:
                continue
            
            current_node = result_list[i]
            current_level = current_node['level']
            
            total_tokens = current_node.get('text_token_count', 0)
            
            # If node is too small, merge children into it
            if total_tokens < min_token_threshold:
                children_indices = find_all_children(i, current_level, result_list)
                
                children_texts = []
                for child_index in sorted(children_indices):
                    if child_index not in nodes_to_remove:
                        child_text = result_list[child_index].get('text', '')
                        if child_text.strip():
                            children_texts.append(child_text)
                        nodes_to_remove.add(child_index)
                
                # Merge children text into parent
                if children_texts:
                    parent_text = current_node.get('text', '')
                    merged_text = parent_text
                    for child_text in children_texts:
                        if merged_text and not merged_text.endswith('\n'):
                            merged_text += '\n\n'
                        merged_text += child_text
                    
                    result_list[i]['text'] = merged_text
                    result_list[i]['text_token_count'] = self.llm.count_tokens(merged_text)
        
        # Remove marked nodes
        for index in sorted(nodes_to_remove, reverse=True):
            result_list.pop(index)
        
        return result_list
