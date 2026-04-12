"""
Markdown Tree Building Component.

Responsible for building hierarchical trees from markdown nodes.
"""

from typing import List, Dict


class MarkdownTreeBuilder:
    """
    Builds hierarchical tree structures from markdown nodes.
    
    Converts flat list of headers into nested tree based on header levels.
    """
    
    @staticmethod
    def build_from_nodes(node_list: List[Dict]) -> List[Dict]:
        """
        Build tree from flat list of nodes.
        
        Args:
            node_list: Flat list with title, level, text fields
            
        Returns:
            Hierarchical tree with nested 'nodes'
        """
        if not node_list:
            return []
        
        stack = []
        root_nodes = []
        node_counter = 1
        
        for node in node_list:
            current_level = node['level']
            
            tree_node = {
                'title': node['title'],
                'node_id': str(node_counter).zfill(4),
                'text': node['text'],
                'line_num': node['line_num'],
                'nodes': []
            }
            node_counter += 1
            
            # Pop stack until we find the parent level
            while stack and stack[-1][1] >= current_level:
                stack.pop()
            
            # Add to appropriate parent or root
            if not stack:
                root_nodes.append(tree_node)
            else:
                parent_node, parent_level = stack[-1]
                parent_node['nodes'].append(tree_node)
            
            # Push current node to stack
            stack.append((tree_node, current_level))
        
        return root_nodes
    
    @staticmethod
    def clean_for_output(tree_nodes: List[Dict]) -> List[Dict]:
        """
        Clean tree for output by removing empty nodes fields.
        
        Args:
            tree_nodes: Tree with nested nodes
            
        Returns:
            Cleaned tree
        """
        cleaned_nodes = []
        
        for node in tree_nodes:
            cleaned_node = {
                'title': node['title'],
                'node_id': node['node_id'],
                'line_num': node['line_num']
            }
            
            # Add text if present
            if 'text' in node:
                cleaned_node['text'] = node['text']
            
            # Add summary if present
            if 'summary' in node:
                cleaned_node['summary'] = node['summary']
            
            # Recursively clean children
            if node['nodes']:
                cleaned_node['nodes'] = MarkdownTreeBuilder.clean_for_output(node['nodes'])
            
            cleaned_nodes.append(cleaned_node)
        
        return cleaned_nodes

