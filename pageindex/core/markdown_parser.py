"""
Markdown Parser Component.

Responsible for parsing markdown structure and extracting nodes with token counts.
"""

import re
from typing import List, Dict, Tuple
from ..llm.llm_client_base import BaseLLMClient


class MarkdownParser:
    """
    Parses markdown files to extract hierarchical structure.
    
    Identifies headers, extracts content, and calculates token counts.
    """
    
    def __init__(self, llm_client: BaseLLMClient = None):
        """
        Initialize markdown parser.
        
        Args:
            llm_client: Optional LLM client for token counting
        """
        self.llm = llm_client
    
    @staticmethod
    def extract_nodes(markdown_content: str) -> Tuple[List[Dict], List[str]]:
        """
        Extract header nodes from markdown content.
        
        Args:
            markdown_content: Raw markdown text
            
        Returns:
            Tuple of (node_list, lines)
        """
        header_pattern = r'^(#{1,6})\s+(.+)$'
        code_block_pattern = r'^```'
        node_list = []
        
        lines = markdown_content.split('\n')
        in_code_block = False
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Check for code block delimiters
            if re.match(code_block_pattern, stripped_line):
                in_code_block = not in_code_block
                continue
            
            # Skip empty lines
            if not stripped_line:
                continue
            
            # Only look for headers when not inside a code block
            if not in_code_block:
                match = re.match(header_pattern, stripped_line)
                if match:
                    title = match.group(2).strip()
                    node_list.append({'node_title': title, 'line_num': line_num})
        
        return node_list, lines
    
    @staticmethod
    def extract_text_content(node_list: List[Dict], markdown_lines: List[str]) -> List[Dict]:
        """
        Extract text content for each node.
        
        Args:
            node_list: List of nodes with line numbers
            markdown_lines: Original markdown lines
            
        Returns:
            Nodes with text content and level added
        """
        all_nodes = []
        for node in node_list:
            line_content = markdown_lines[node['line_num'] - 1]
            header_match = re.match(r'^(#{1,6})', line_content)
            
            if header_match is None:
                print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
                continue
            
            processed_node = {
                'title': node['node_title'],
                'line_num': node['line_num'],
                'level': len(header_match.group(1))
            }
            all_nodes.append(processed_node)
        
        # Extract text for each node
        for i, node in enumerate(all_nodes):
            start_line = node['line_num'] - 1
            if i + 1 < len(all_nodes):
                end_line = all_nodes[i + 1]['line_num'] - 1
            else:
                end_line = len(markdown_lines)
            
            node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
        
        return all_nodes
    
    def count_tokens_for_nodes(self, node_list: List[Dict]) -> List[Dict]:
        """
        Count tokens for each node including all children.
        
        Args:
            node_list: Nodes with text content
            
        Returns:
            Nodes with text_token_count added
        """
        def find_all_children(parent_index, parent_level, nodes):
            """Find all direct and indirect children of a parent node."""
            children_indices = []
            
            for i in range(parent_index + 1, len(nodes)):
                current_level = nodes[i]['level']
                
                # If we hit a node at same or higher level than parent, stop
                if current_level <= parent_level:
                    break
                
                # This is a descendant
                children_indices.append(i)
            
            return children_indices
        
        # Make a copy to avoid modifying original
        result_list = node_list.copy()
        
        # Process nodes from end to beginning to ensure children are processed before parents
        for i in range(len(result_list) - 1, -1, -1):
            current_node = result_list[i]
            current_level = current_node['level']
            
            # Get all children of this node
            children_indices = find_all_children(i, current_level, result_list)
            
            # Start with the node's own text
            node_text = current_node.get('text', '')
            total_text = node_text
            
            # Add all children's text
            for child_index in children_indices:
                child_text = result_list[child_index].get('text', '')
                if child_text:
                    total_text += '\n' + child_text
            
            # Calculate token count for combined text
            if self.llm:
                result_list[i]['text_token_count'] = self.llm.count_tokens(total_text)
            else:
                # Fallback: estimate 4 chars per token
                result_list[i]['text_token_count'] = len(total_text) // 4
        
        return result_list
