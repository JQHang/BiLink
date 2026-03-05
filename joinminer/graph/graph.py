"""
Graph Configuration Parser

This module provides the Graph class for parsing and validating simplified
graph configurations based on element tables.

Configuration Format:
- element_table_dir: Base directory for all element tables (with URI scheme)
- partition_columns: List of partition column names
- nodes: Dict of node type configurations
- edges: Dict of edge type configurations

Each node configuration includes:
- table_name: Element table name
- id_column: Single ID column name

Each edge configuration includes:
- table_name: Element table name
- u_node: Source node {type, id_column}
- v_node: Target node {type, id_column}

Feature metadata is automatically loaded from _element_info.json files.
"""

import logging
from typing import Dict, Any

from joinminer.fileio import FileIO

logger = logging.getLogger(__name__)


class Graph:
    """
    Graph configuration parser and metadata manager.

    This class parses simplified graph configurations and loads feature metadata
    from element tables. It validates node/edge references and provides summary
    information about the graph structure.

    Attributes:
        element_table_dir (str): Base directory for element tables (with URI scheme)
        partition_columns (list): List of partition column names
        fileio (FileIO): FileIO instance for reading metadata files
        nodes (dict): Parsed node configurations with metadata (includes 'type_index')
        edges (dict): Parsed edge configurations with metadata (includes 'type_index')
        node_index_to_type (dict): Mapping from node type index to type name
        edge_index_to_type (dict): Mapping from edge type index to type name
    """

    def __init__(self, config: Dict[str, Any], fileio: FileIO):
        """
        Initialize graph from configuration.

        Args:
            config: Graph configuration dictionary with:
                - element_table_dir: Base directory path (with URI scheme)
                - partition_columns: List of partition column names
                - nodes: Dict of node configurations
                - edges: Dict of edge configurations
            fileio: FileIO instance for reading metadata files

        Raises:
            ValueError: If required configuration fields are missing
            ValueError: If node type references in edges don't exist
        """
        # Validate required fields
        if 'element_table_dir' not in config:
            raise ValueError("Configuration must include 'element_table_dir'")
        if 'partition_columns' not in config:
            raise ValueError("Configuration must include 'partition_columns'")
        if 'nodes' not in config:
            raise ValueError("Configuration must include 'nodes' section")
        if 'edges' not in config:
            raise ValueError("Configuration must include 'edges' section")

        # Store global settings
        self.element_table_dir = config['element_table_dir']
        self.partition_columns = config['partition_columns']
        self.fileio = fileio

        # Initialize storage
        self.nodes = {}
        self.edges = {}

        # Parse configuration
        logger.info(f"Parsing graph configuration from: {self.element_table_dir}")
        self._parse_nodes(config['nodes'])
        self._parse_edges(config['edges'])

        # Build index to type name mappings
        self.node_index_to_type = {node_info['type_index']: node_type
                                   for node_type, node_info in self.nodes.items()}
        self.edge_index_to_type = {edge_info['type_index']: edge_type
                                   for edge_type, edge_info in self.edges.items()}

        # Display summary
        self.show_brief_summary()

    def _parse_nodes(self, nodes_config: Dict[str, Any]) -> None:
        """
        Parse node configurations.

        For each node type, this method:
        1. Validates required fields (table_name, id_column)
        2. Constructs table path
        3. Loads feature metadata from _element_info.json
        4. Stores parsed configuration

        Args:
            nodes_config: Dict mapping node type names to configurations

        Raises:
            ValueError: If required fields are missing
        """
        for idx, (node_type, node_config) in enumerate(nodes_config.items()):
            # Validate required fields
            if 'table_name' not in node_config:
                raise ValueError(f"Node '{node_type}' missing required field 'table_name'")
            if 'id_column' not in node_config:
                raise ValueError(f"Node '{node_type}' missing required field 'id_column'")

            table_name = node_config['table_name']
            id_column = node_config['id_column']

            # Construct paths
            table_path = f"{self.element_table_dir}/{table_name}"
            info_path = f"{table_path}/_element_info.json"

            # Load feature metadata
            feature_info = self.fileio.read_json(info_path)

            # Store parsed node
            self.nodes[node_type] = {
                'table_name': table_name,
                'id_column': id_column,
                'table_path': table_path,
                'feature_count': len(feature_info.get('features', [])),
                'feature_metadata': feature_info,
                'type_index': idx
            }

            logger.debug(f"Parsed node '{node_type}': {table_name} "
                        f"({self.nodes[node_type]['feature_count']} features)")

    def _parse_edges(self, edges_config: Dict[str, Any]) -> None:
        """
        Parse edge configurations.

        For each edge type, this method:
        1. Validates required fields (table_name, u_node, v_node)
        2. Validates node type references exist
        3. Constructs table path
        4. Loads feature metadata from _element_info.json
        5. Stores parsed configuration

        Args:
            edges_config: Dict mapping edge type names to configurations

        Raises:
            ValueError: If required fields are missing
            ValueError: If referenced node types don't exist
        """
        for idx, (edge_type, edge_config) in enumerate(edges_config.items()):
            # Validate required fields
            if 'table_name' not in edge_config:
                raise ValueError(f"Edge '{edge_type}' missing required field 'table_name'")
            if 'u_node' not in edge_config:
                raise ValueError(f"Edge '{edge_type}' missing required field 'u_node'")
            if 'v_node' not in edge_config:
                raise ValueError(f"Edge '{edge_type}' missing required field 'v_node'")

            table_name = edge_config['table_name']
            u_node = edge_config['u_node']
            v_node = edge_config['v_node']

            # Validate u_node and v_node structure
            if 'type' not in u_node or 'id_column' not in u_node:
                raise ValueError(f"Edge '{edge_type}': u_node must have 'type' and 'id_column'")
            if 'type' not in v_node or 'id_column' not in v_node:
                raise ValueError(f"Edge '{edge_type}': v_node must have 'type' and 'id_column'")

            # Validate node type references
            if u_node['type'] not in self.nodes:
                raise ValueError(f"Edge '{edge_type}': unknown u_node type '{u_node['type']}'")
            if v_node['type'] not in self.nodes:
                raise ValueError(f"Edge '{edge_type}': unknown v_node type '{v_node['type']}'")

            # Construct paths
            table_path = f"{self.element_table_dir}/{table_name}"
            info_path = f"{table_path}/_element_info.json"

            # Load feature metadata
            feature_info = self.fileio.read_json(info_path)

            # Store parsed edge
            self.edges[edge_type] = {
                'table_name': table_name,
                'table_path': table_path,
                'u_node': u_node,  # {'type': 'work', 'id_column': 'work_id'}
                'v_node': v_node,  # {'type': 'author', 'id_column': 'author_id'}
                'feature_count': len(feature_info.get('features', [])),
                'feature_metadata': feature_info,
                'type_index': idx
            }

            logger.debug(f"Parsed edge '{edge_type}': {table_name} "
                        f"({u_node['type']} <-> {v_node['type']}, "
                        f"{self.edges[edge_type]['feature_count']} features)")

    def show_brief_summary(self) -> None:
        """
        Display a summary of the graph structure.

        This method logs:
        - Number of node types with feature counts and table names
        - Number of edge types with feature counts, connections, and table names

        The summary is displayed at INFO level for visibility during graph initialization.
        """
        logger.info("=" * 70)
        logger.info(f"Graph Summary: {len(self.nodes)} node types, {len(self.edges)} edge types")
        logger.info("=" * 70)

        logger.info(f"\nNode Types ({len(self.nodes)}):")
        for node_type, node_info in self.nodes.items():
            feat_count = node_info['feature_count']
            table_name = node_info['table_name']
            id_column = node_info['id_column']
            type_idx = node_info['type_index']
            logger.info(f"  • [{type_idx}] {node_type:15s} | {feat_count:4d} features | "
                       f"ID: {id_column:20s} | Table: {table_name}")

        logger.info(f"\nEdge Types ({len(self.edges)}):")
        for edge_type, edge_info in self.edges.items():
            feat_count = edge_info['feature_count']
            table_name = edge_info['table_name']
            u_type = edge_info['u_node']['type']
            v_type = edge_info['v_node']['type']
            u_col = edge_info['u_node']['id_column']
            v_col = edge_info['v_node']['id_column']
            type_idx = edge_info['type_index']

            # Get node type indices
            u_idx = self.nodes[u_type]['type_index']
            v_idx = self.nodes[v_type]['type_index']

            connection = f"{u_type}[{u_idx}] <-> {v_type}[{v_idx}]"
            id_cols = f"{u_col}, {v_col}"
            logger.info(f"  • [{type_idx}] {edge_type:20s} | {feat_count:4d} features | "
                       f"{connection:30s} | IDs: {id_cols}")
            logger.info(f"    Table: {table_name}")

        logger.info("=" * 70)
