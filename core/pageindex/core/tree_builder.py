"""
Tree Building Component.

Responsible for converting flat TOC lists into hierarchical tree structures.
"""

from typing import List, Dict
import copy


class TreeBuilder:
    """
    Builds hierarchical tree structures from flat TOC lists.
    
    Handles structure indexing, parent-child relationships, and tree cleanup.
    """
    
    @staticmethod
    def build_from_flat_list(toc_items: List[Dict], end_physical_index: int) -> List[Dict]:
        """
        Build tree from flat list with proper start/end indices.
        
        Args:
            toc_items: Flat list of TOC items with physical_index
            end_physical_index: Last page index in document
            
        Returns:
            Hierarchical tree structure
        """
        # First convert physical_index to start_index
        for i, item in enumerate(toc_items):
            item['start_index'] = item.get('physical_index')
            if i < len(toc_items) - 1:
                if toc_items[i + 1].get('appear_start') == 'yes':
                    item['end_index'] = toc_items[i + 1]['physical_index'] - 1
                else:
                    item['end_index'] = toc_items[i + 1]['physical_index']
            else:
                item['end_index'] = end_physical_index
        
        tree = TreeBuilder.list_to_tree(toc_items)
        
        if len(tree) != 0:
            return tree
        else:
            # Remove temporary fields if tree building failed
            for node in toc_items:
                node.pop('appear_start', None)
                node.pop('physical_index', None)
            return toc_items
    
    @staticmethod
    def list_to_tree(data: List[Dict]) -> List[Dict]:
        """
        Convert flat list with structure indices to hierarchical tree.
        
        Args:
            data: List of items with 'structure' field like "1", "1.1", "1.1.1"
            
        Returns:
            Hierarchical tree with nested 'nodes'
        """
        def get_parent_structure(structure):
            """Get parent structure code."""
            if not structure:
                return None
            parts = str(structure).split('.')
            return '.'.join(parts[:-1]) if len(parts) > 1 else None
        
        # Create nodes and track parent-child relationships
        nodes = {}
        root_nodes = []
        
        for item in data:
            structure = item.get('structure')
            node = {
                'title': item.get('title'),
                'start_index': item.get('start_index'),
                'end_index': item.get('end_index'),
                'nodes': []
            }
            
            nodes[structure] = node
            
            # Find parent
            parent_structure = get_parent_structure(structure)
            
            if parent_structure:
                # Add as child to parent if parent exists
                if parent_structure in nodes:
                    nodes[parent_structure]['nodes'].append(node)
                else:
                    root_nodes.append(node)
            else:
                # No parent, this is a root node
                root_nodes.append(node)
        
        # Clean empty children arrays
        def clean_node(node):
            if not node['nodes']:
                del node['nodes']
            else:
                for child in node['nodes']:
                    clean_node(child)
            return node
        
        return [clean_node(node) for node in root_nodes]
    
    @staticmethod
    def add_preface_if_needed(toc_items: List[Dict]) -> List[Dict]:
        """
        Add preface node if document doesn't start at page 1.
        
        Args:
            toc_items: List of TOC items
            
        Returns:
            TOC items with preface added if needed
        """
        if not isinstance(toc_items, list) or not toc_items:
            return toc_items

        if toc_items[0]['physical_index'] is not None and toc_items[0]['physical_index'] > 1:
            preface_node = {
                "structure": "0",
                "title": "Preface",
                "physical_index": 1,
            }
            toc_items.insert(0, preface_node)
        
        return toc_items
