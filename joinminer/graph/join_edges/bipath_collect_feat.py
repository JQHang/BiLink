"""
Bipath Feature Preparation

This module provides functions for preparing feature ID mappings and extracting
path elements with feature information for bipath collection.

Key Functions:
- create_feat_id_mappings: Create in-memory feat_id mappings for nodes and edges
- extract_path_elements: Extract path elements from exploration paths and map to feat_ids
"""

import logging
from typing import Set
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

from joinminer.spark.io import read_table
from joinminer.graph.join_edges.bipath_types import get_forward_paths, get_backward_paths
from joinminer.graph.join_edges.add_path import _get_path_type_columns

# Module-level logger
logger = logging.getLogger("joinminer")


def create_feat_id_mappings(
    spark_ctx,
    graph,
    output_path: str,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> tuple[DataFrame, DataFrame]:
    """
    Create feat_id mappings for nodes and edges.

    Reads node and edge feature tables, adds type information, unions them,
    and persists the mapping DataFrames.

    Args:
        spark_ctx: Spark context with persist_manager
        graph: Graph instance with node/edge type mappings and partition columns
        output_path: Base output path (used to locate features directory)
        path_node_type_indices: Set of node type indices to include
        path_edge_type_indices: Set of edge type indices to include

    Returns:
        (nodes_feat_id_df, edges_feat_id_df) - Persisted mapping DataFrames

    Persisted DataFrames:
        - nodes_feat_id_df: (node_id, node_type, feat_id, partition_columns)
        - edges_feat_id_df: (u_node_id, v_node_id, edge_type, feat_id, partition_columns)
        - release_point='merge_bipaths_done'
    """
    logger.info("\n" + "="*70)
    logger.info("Creating feat_id mapping for nodes and edges")
    logger.info("="*70)

    features_base_path = f"{output_path}/features"
    node_dfs = []

    # Step 1: Read and process node features
    logger.info("\nStep 1: Reading and processing node features")
    logger.info("-"*70)

    for node_type_index in sorted(path_node_type_indices):
        node_type_name = graph.node_index_to_type[node_type_index]
        node_feat_path = f"{features_base_path}/nodes/{node_type_name}"

        logger.info(f"  Reading node type: {node_type_name} (index: {node_type_index})")
        logger.info(f"    Path: {node_feat_path}")

        node_df = read_table(spark_ctx, node_feat_path)

        # Select only needed columns and add node_type using F.lit
        node_df = node_df.select(
            F.col('node_id'),
            F.lit(node_type_index).alias('node_type'),
            F.col('feat_id'),
            *[F.col(c) for c in graph.partition_columns]
        )

        node_dfs.append(node_df)
        logger.info(f"    ✓ Loaded and processed {node_type_name}")

    logger.info(f"\nProcessed {len(node_dfs)} node type(s)")

    # Union all node DataFrames
    if not node_dfs:
        raise ValueError("No node features found")

    nodes_feat_id_df = node_dfs[0]
    for df in node_dfs[1:]:
        nodes_feat_id_df = nodes_feat_id_df.unionByName(df)

    logger.info(f"  ✓ Unioned {len(node_dfs)} node type(s)")

    # Persist nodes_feat_id_df
    nodes_feat_id_df = spark_ctx.persist_manager.persist(
        nodes_feat_id_df,
        release_point='merge_bipaths_done',
        name='nodes_feat_id_df'
    )
    node_count = nodes_feat_id_df.count()
    logger.info(f"  ✓ Persisted nodes_feat_id_df: {node_count} records")

    # Step 2: Read and process edge features
    logger.info("\nStep 2: Reading and processing edge features")
    logger.info("-"*70)

    edge_dfs = []

    for edge_type_index in sorted(path_edge_type_indices):
        edge_type_name = graph.edge_index_to_type[edge_type_index]
        edge_feat_path = f"{features_base_path}/edges/{edge_type_name}"

        logger.info(f"  Reading edge type: {edge_type_name} (index: {edge_type_index})")
        logger.info(f"    Path: {edge_feat_path}")

        edge_df = read_table(spark_ctx, edge_feat_path)

        # Select only needed columns and add edge_type using F.lit
        edge_df = edge_df.select(
            F.col('u_node_id'),
            F.col('v_node_id'),
            F.lit(edge_type_index).alias('edge_type'),
            F.col('feat_id'),
            *[F.col(c) for c in graph.partition_columns]
        )

        edge_dfs.append(edge_df)
        logger.info(f"    ✓ Loaded and processed {edge_type_name}")

    logger.info(f"\nProcessed {len(edge_dfs)} edge type(s)")

    # Union all edge DataFrames
    if not edge_dfs:
        raise ValueError("No edge features found")

    edges_feat_id_df = edge_dfs[0]
    for df in edge_dfs[1:]:
        edges_feat_id_df = edges_feat_id_df.unionByName(df)

    logger.info(f"  ✓ Unioned {len(edge_dfs)} edge type(s)")

    # Persist edges_feat_id_df
    edges_feat_id_df = spark_ctx.persist_manager.persist(
        edges_feat_id_df,
        release_point='merge_bipaths_done',
        name='edges_feat_id_df'
    )
    edge_count = edges_feat_id_df.count()
    logger.info(f"  ✓ Persisted edges_feat_id_df: {edge_count} records")

    logger.info(f"\nFeat ID mapping created: {node_count} nodes, {edge_count} edges")

    return nodes_feat_id_df, edges_feat_id_df


def extract_path_elements(
    spark_ctx,
    graph,
    output_path: str,
    target_paths_df: DataFrame,
    max_hop: int,
    nodes_feat_id_df: DataFrame,
    edges_feat_id_df: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """
    Extract path elements (nodes and edges) from exploration paths and map to feat_ids.

    Reads exploration paths from different hops, filters by target path types,
    unions them, adds path_id, extracts individual nodes and edges, and maps
    them to feat_ids.

    Args:
        spark_ctx: Spark context with persist_manager
        graph: Graph instance with partition columns
        output_path: Base output path (used to locate path_exploration directory)
        target_paths_df: DataFrame with target path types for filtering
        max_hop: Maximum hop number in bipaths (determines explore_hop)
        nodes_feat_id_df: Node feat_id mapping (node_id, node_type, feat_id, partition_columns)
        edges_feat_id_df: Edge feat_id mapping (u_node_id, v_node_id, edge_type, feat_id, partition_columns)

    Returns:
        (path_nodes, path_edges)

    Output schemas:
        path_nodes: (node_0_id, seed_node_side, path_id, hop_k, pos,
                    node_type, feat_id, partition_columns)
        path_edges: (node_0_id, seed_node_side, path_id, hop_k, pos,
                    edge_type, u_index, v_index, feat_id, partition_columns)

    Persisted DataFrames:
        - all_paths_df: release_point='seed_node_paths_done'
        - path_nodes: release_point='seed_node_paths_done'
        - path_edges: release_point='seed_node_paths_done'
    """
    logger.info("\n" + "="*70)
    logger.info("Extracting path elements from exploration paths")
    logger.info("="*70)

    explore_hop = (max_hop + 1) // 2
    path_df_list = []
    exploration_path = f"{output_path}/path_exploration"

    # Step 1: Read exploration paths and filter by target path types
    logger.info("\nStep 1: Reading and filtering exploration paths")
    logger.info("-"*70)

    # Prepare combined path types (forward + backward)
    forward_path_types = get_forward_paths(spark_ctx.spark, target_paths_df)
    backward_path_types = get_backward_paths(spark_ctx.spark, target_paths_df)

    forward_path_types = forward_path_types.withColumn('seed_node_side', F.lit(0))
    backward_path_types = backward_path_types.withColumn('seed_node_side', F.lit(1))

    combined_path_types = forward_path_types.unionByName(backward_path_types)
    combined_path_types = spark_ctx.persist_manager.persist(
        combined_path_types,
        release_point='seed_node_paths_done',
        name='combined_path_types'
    )
    logger.info(f"Forward path types: {forward_path_types.count()} types")
    logger.info(f"Backward path types: {backward_path_types.count()} types")

    # Process hop_1 to explore_hop (skip hop_0)
    for hop_k in range(1, explore_hop + 1):
        hop_path = f"{exploration_path}/hop_{hop_k}"
        logger.info(f"Reading hop_{hop_k} from: {hop_path}")
        path_df = read_table(spark_ctx, hop_path)

        # Filter paths matching target types
        path_type_cols = _get_path_type_columns(hop_k)
        join_cols = path_type_cols + ['seed_node_side']

        hop_path_types = combined_path_types.filter(F.col('hop_k') == hop_k).select(join_cols).distinct()
        path_df = path_df.join(F.broadcast(hop_path_types), on=join_cols, how='inner')

        # Convert to arrays and keep node_0_id, seed_node_side
        node_id_cols = [F.col(f'node_{i}_id') for i in range(hop_k + 1)]
        node_type_cols = [F.col(f'node_{i}_type') for i in range(hop_k + 1)]
        edge_type_cols = [F.col(f'edge_{i}_type') for i in range(hop_k)]
        u_index_cols = [F.col(f'u_index_of_edge_{i}') for i in range(hop_k)]
        v_index_cols = [F.col(f'v_index_of_edge_{i}') for i in range(hop_k)]

        path_df = path_df.select(
            F.col('node_0_id'),
            F.col('seed_node_side'),
            F.lit(hop_k).alias('hop_k'),
            F.array(*node_id_cols).alias('node_id_array'),
            F.array(*node_type_cols).alias('node_type_array'),
            F.array(*edge_type_cols).alias('edge_type_array'),
            F.array(*u_index_cols).alias('u_index_array'),
            F.array(*v_index_cols).alias('v_index_array'),
            *[F.col(c) for c in graph.partition_columns]
        )

        path_df_list.append(path_df)
        logger.info(f"Processed hop_{hop_k}")

    # Union all paths
    all_paths_df = path_df_list[0]
    for df in path_df_list[1:]:
        all_paths_df = all_paths_df.unionByName(df)

    # Add path_id to distinguish different paths within same group
    window_spec = Window.partitionBy('node_0_id', 'seed_node_side', *graph.partition_columns).orderBy(F.rand())
    all_paths_df = all_paths_df.withColumn('path_id', F.row_number().over(window_spec) - 1)

    # Persist for reuse
    all_paths_df = spark_ctx.persist_manager.persist(
        all_paths_df,
        release_point='seed_node_paths_done',
        name='all_paths_df'
    )
    path_count = all_paths_df.count()
    logger.info(f"Combined {explore_hop} hops: {path_count} paths")

    # Common key columns for path data (including partition columns)
    path_key_cols = ['node_0_id', 'seed_node_side', 'path_id', 'hop_k'] + graph.partition_columns

    # Step 2: Extract nodes (skip position 0 using slice) and map to feat_ids
    logger.info("\nStep 2: Extracting nodes and mapping to feat_ids")
    logger.info("-"*70)

    # Slice arrays to exclude position 0
    sliced_nodes_df = all_paths_df.select(
        *path_key_cols,
        F.slice('node_id_array', 2, F.col('hop_k')).alias('node_id_array'),
        F.slice('node_type_array', 2, F.col('hop_k')).alias('node_type_array')
    )

    # Explode and extract node elements
    path_nodes = sliced_nodes_df.select(
        *path_key_cols,
        F.posexplode(F.arrays_zip('node_id_array', 'node_type_array')).alias('pos', 'node_struct')
    ).select(
        *path_key_cols, 'pos',
        F.col('node_struct.node_id_array').alias('node_id'),
        F.col('node_struct.node_type_array').alias('node_type')
    )

    # Map nodes to feat_ids and drop node_id
    node_join_cols = ['node_id', 'node_type'] + graph.partition_columns
    path_nodes = path_nodes.join(
        nodes_feat_id_df,
        on=node_join_cols,
        how='left'
    ).drop('node_id')

    path_nodes = spark_ctx.persist_manager.persist(
        path_nodes,
        release_point='seed_node_paths_done',
        name='path_nodes'
    )
    logger.info(f"✓ Extracted and mapped path nodes")

    # Step 3: Extract edges and map to feat_ids
    logger.info("\nStep 3: Extracting edges and mapping to feat_ids")
    logger.info("-"*70)

    path_edges = all_paths_df.select(
        *path_key_cols,
        'node_id_array',
        F.posexplode(F.arrays_zip('edge_type_array', 'u_index_array', 'v_index_array')).alias('pos', 'edge_struct')
    ).select(
        *path_key_cols, 'pos',
        F.element_at('node_id_array', F.col('edge_struct.u_index_array') + 1).alias('u_node_id'),
        F.element_at('node_id_array', F.col('edge_struct.v_index_array') + 1).alias('v_node_id'),
        F.col('edge_struct.edge_type_array').alias('edge_type'),
        F.col('edge_struct.u_index_array').alias('u_index'),
        F.col('edge_struct.v_index_array').alias('v_index')
    )

    # Map edges to feat_ids and drop u_node_id, v_node_id
    edge_join_cols = ['u_node_id', 'v_node_id', 'edge_type'] + graph.partition_columns
    path_edges = path_edges.join(
        edges_feat_id_df,
        on=edge_join_cols,
        how='inner'
    ).drop('u_node_id', 'v_node_id')

    path_edges = spark_ctx.persist_manager.persist(
        path_edges,
        release_point='seed_node_paths_done',
        name='path_edges'
    )
    logger.info(f"✓ Extracted and mapped path edges")

    logger.info(f"\n{'='*70}")
    logger.info(f"Path element extraction completed")
    logger.info(f"{'='*70}")

    return path_nodes, path_edges


def create_feat_vector_mapping(
    spark_ctx,
    graph,
    output_path: str,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> DataFrame:
    """
    Create element_type -> feat_vector mapping table.

    This mapping is static and should be created once before batch processing
    to avoid repeated disk reads.

    Args:
        spark_ctx: Spark context
        graph: Graph object with partition_columns and feature info
        output_path: Base output path containing features directory
        path_node_type_indices: All node types in paths
        path_edge_type_indices: All edge types in paths

    Returns:
        DataFrame with schema: (element_type, feat_id, feat_vector, partition_columns)
        - element_type: '0_{node_type}' for nodes, '1_{edge_type}' for edges
        - Returns None if no element types have features
    """
    logger.info("\n" + "="*70)
    logger.info("Creating element_type -> feat_vector mapping")
    logger.info("="*70)

    # Filter to types with features
    feat_node_type_indices = {
        idx for idx in path_node_type_indices
        if graph.nodes[graph.node_index_to_type[idx]]['feature_count'] > 0
    }
    feat_edge_type_indices = {
        idx for idx in path_edge_type_indices
        if graph.edges[graph.edge_index_to_type[idx]]['feature_count'] > 0
    }

    logger.info(f"  Node types with features: {len(feat_node_type_indices)}/{len(path_node_type_indices)}")
    logger.info(f"  Edge types with features: {len(feat_edge_type_indices)}/{len(path_edge_type_indices)}")

    if not feat_node_type_indices and not feat_edge_type_indices:
        raise ValueError(
            "No element types with features. "
            f"Checked {len(path_node_type_indices)} node types and {len(path_edge_type_indices)} edge types."
        )

    features_base_path = f"{output_path}/features"
    element_feat_dfs = []

    # Read node features
    for node_type_index in sorted(feat_node_type_indices):
        node_type_name = graph.node_index_to_type[node_type_index]
        node_feat_path = f"{features_base_path}/nodes/{node_type_name}"

        logger.info(f"  Reading node type: {node_type_name} (index: {node_type_index})")

        node_df = read_table(spark_ctx, node_feat_path)
        node_df = node_df.select(
            F.lit(f'0_{node_type_index}').alias('element_type'),
            F.col('feat_id'),
            F.col('feat_vector'),
            *[F.col(c) for c in graph.partition_columns]
        )
        element_feat_dfs.append(node_df)

    # Read edge features
    for edge_type_index in sorted(feat_edge_type_indices):
        edge_type_name = graph.edge_index_to_type[edge_type_index]
        edge_feat_path = f"{features_base_path}/edges/{edge_type_name}"

        logger.info(f"  Reading edge type: {edge_type_name} (index: {edge_type_index})")

        edge_df = read_table(spark_ctx, edge_feat_path)
        edge_df = edge_df.select(
            F.lit(f'1_{edge_type_index}').alias('element_type'),
            F.col('feat_id'),
            F.col('feat_vector'),
            *[F.col(c) for c in graph.partition_columns]
        )
        element_feat_dfs.append(edge_df)

    # Union all DataFrames
    element_feat_df = element_feat_dfs[0]
    for df in element_feat_dfs[1:]:
        element_feat_df = element_feat_df.unionByName(df)

    logger.info(f"  Created mapping with {len(element_feat_dfs)} element type(s)")
    logger.info("="*70)

    return element_feat_df


__all__ = [
    'create_feat_id_mappings',
    'create_feat_vector_mapping',
    'extract_path_elements'
]
