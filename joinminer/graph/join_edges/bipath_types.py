"""
Bipath Type Manipulation Module

Utilities for extracting and transforming path type specifications from bipath data.
Path types define the schema/structure of paths without actual node IDs - useful for
generating path exploration specifications and pattern matching.

Two categories of operations:
1. Exploration Path Extraction - Generate multiple paths of varying lengths for discovery
2. Bipath Decomposition - Split bipaths into their forward/backward components

Functions:
    extract_u_explore_paths: Extract u_node side exploration paths of hop_k length (seed_node_side=0)
    extract_v_explore_paths: Extract v_node side exploration paths of target_hop_k length (seed_node_side=1)
    get_forward_paths: Decompose bipaths into their forward path components
    get_backward_paths: Decompose bipaths into their backward path components
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def extract_u_explore_paths(paths_df: DataFrame, hop_k: int) -> DataFrame:
    """
    Extract u_node side exploration paths of hop_k length using DataFrame operations.

    EXPLORATION FUNCTION: Generates path specifications of exactly hop_k length for path discovery.
    Extracts the first hop_k edges (and hop_k+1 nodes) from bipaths with length >= 2*hop_k - 1.
    Always returns a DataFrame (empty if no paths qualify), with seed_node_side=0 column added.

    Args:
        paths_df: DataFrame with path type columns (must have hop_k column)
        hop_k: Number of hops to extract

    Returns:
        DataFrame with first hop_k edges, node types, and seed_node_side=0 column
    """
    # Use hop_k column directly for filtering
    filtered_df = paths_df.filter(F.col('hop_k') >= 2 * hop_k - 1)

    # Select columns including node types and edges
    select_cols = []

    # Add node types (0 to hop_k, total hop_k+1 nodes)
    for i in range(hop_k + 1):
        select_cols.append(f'node_{i}_type')

    # Add edge types and indices
    for i in range(hop_k):
        select_cols.extend([
            f'edge_{i}_type',
            f'u_index_of_edge_{i}',
            f'v_index_of_edge_{i}'
        ])

    # Select columns, add seed_node_side, remove duplicates, and coalesce to single partition
    result_df = (filtered_df
                 .select(*select_cols)
                 .distinct()
                 .withColumn('seed_node_side', F.lit(0))
                 .coalesce(1))

    return result_df  # Always return DataFrame, even if empty


def extract_v_explore_paths(paths_df: DataFrame, target_hop_k: int, max_hop: int) -> DataFrame:
    """
    Extract v_node side exploration paths of target_hop_k length with index transformation.

    EXPLORATION FUNCTION: Generates path specifications of exactly target_hop_k length for path discovery.
    Extracts the last target_hop_k edges (and target_hop_k+1 nodes) in reverse order.
    For a path with length path_hop_k, node index i becomes node_{path_hop_k - i}.
    Always returns a DataFrame (empty if no paths qualify), with seed_node_side=1 column added.

    Args:
        paths_df: DataFrame with path type columns (must have hop_k column)
        target_hop_k: Number of hops to extract from the end
        max_hop: Maximum hop number in schema (for knowing which columns exist)

    Returns:
        DataFrame with transformed backward paths, node types, and seed_node_side=1 column
    """
    # Use hop_k column directly for filtering
    filtered_df = paths_df.filter(F.col('hop_k') >= 2 * target_hop_k)

    # Build select expressions using WHEN for dynamic column selection
    select_exprs = []

    # Add node type expressions (need target_hop_k + 1 nodes)
    for out_idx in range(target_hop_k + 1):
        # Initialize with the first condition (minimum path length)
        min_path_len = 2 * target_hop_k
        src_idx = min_path_len - out_idx

        # Direct initialization with first WHEN condition
        node_expr = F.when(
            F.col('hop_k') == min_path_len,
            F.col(f'node_{src_idx}_type')
        )

        # Chain additional conditions for longer paths
        for path_len in range(min_path_len + 1, max_hop + 1):
            src_idx = path_len - out_idx

            node_expr = node_expr.when(
                F.col('hop_k') == path_len,
                F.col(f'node_{src_idx}_type')
            )

        # Add to select list with alias
        select_exprs.append(node_expr.alias(f'node_{out_idx}_type'))

    # Add edge expressions (target_hop_k edges)
    for out_idx in range(target_hop_k):
        # Initialize with the first condition (minimum path length)
        min_path_len = 2 * target_hop_k
        src_idx = min_path_len - 1 - out_idx

        # Direct initialization with first WHEN condition
        edge_expr = F.when(
            F.col('hop_k') == min_path_len,
            F.col(f'edge_{src_idx}_type')
        )
        u_expr = F.when(
            F.col('hop_k') == min_path_len,
            F.col('hop_k') - F.col(f'u_index_of_edge_{src_idx}')
        )
        v_expr = F.when(
            F.col('hop_k') == min_path_len,
            F.col('hop_k') - F.col(f'v_index_of_edge_{src_idx}')
        )

        # Chain additional conditions for longer paths
        for path_len in range(min_path_len + 1, max_hop + 1):
            src_idx = path_len - 1 - out_idx

            edge_expr = edge_expr.when(
                F.col('hop_k') == path_len,
                F.col(f'edge_{src_idx}_type')
            )
            u_expr = u_expr.when(
                F.col('hop_k') == path_len,
                F.col('hop_k') - F.col(f'u_index_of_edge_{src_idx}')
            )
            v_expr = v_expr.when(
                F.col('hop_k') == path_len,
                F.col('hop_k') - F.col(f'v_index_of_edge_{src_idx}')
            )

        # Add to select list with aliases
        select_exprs.extend([
            edge_expr.alias(f'edge_{out_idx}_type'),
            u_expr.alias(f'u_index_of_edge_{out_idx}'),
            v_expr.alias(f'v_index_of_edge_{out_idx}')
        ])

    # Single select operation with seed_node_side column
    result_df = (filtered_df
                 .select(*select_exprs)
                 .distinct()
                 .withColumn('seed_node_side', F.lit(1))
                 .coalesce(1))

    return result_df  # Always return DataFrame, even if empty


def get_forward_paths(spark, target_paths_df: DataFrame) -> DataFrame:
    """
    Decompose bipaths into their forward path components.

    DECOMPOSITION FUNCTION: Splits each bipath into its forward component.
    For a bipath with hop_k edges, the forward path has length (hop_k + 1) // 2.
    This function extracts the first portion of each bipath as its forward path.

    Args:
        spark: SparkSession for creating DataFrames
        target_paths_df: DataFrame with bipath type columns (hop_k, node_i_type, edge_i_type, ...)

    Returns:
        DataFrame with forward path types:
        - hop_k: forward path length = (original_hop_k + 1) // 2
        - node_i_type: node types from 0 to forward_hop
        - edge_i_type: edge types from 0 to forward_hop - 1
        - u_index_of_edge_i, v_index_of_edge_i: edge endpoint indices

    Example:
        5-hop bipath -> 3-hop forward path (nodes 0-3, edges 0-2)
        4-hop bipath -> 2-hop forward path (nodes 0-2, edges 0-1)
    """

    rows = target_paths_df.collect()

    # Early validation: fail fast if no input data
    if not rows:
        raise ValueError("No bipath data provided to get_forward_paths")

    forward_paths_by_hop = {}  # Group by forward_hop to create consistent schemas

    for row in rows:
        bipath_hop_k = row['hop_k']
        forward_hop = (bipath_hop_k + 1) // 2

        # Build forward path dict
        forward_path = {'hop_k': forward_hop}

        # Copy node types: node_0 to node_{forward_hop}
        for i in range(forward_hop + 1):
            forward_path[f'node_{i}_type'] = row[f'node_{i}_type']

        # Copy edge types and indices: edge_0 to edge_{forward_hop-1}
        for i in range(forward_hop):
            forward_path[f'edge_{i}_type'] = row[f'edge_{i}_type']
            forward_path[f'u_index_of_edge_{i}'] = row[f'u_index_of_edge_{i}']
            forward_path[f'v_index_of_edge_{i}'] = row[f'v_index_of_edge_{i}']

        # Group by forward_hop
        if forward_hop not in forward_paths_by_hop:
            forward_paths_by_hop[forward_hop] = []
        forward_paths_by_hop[forward_hop].append(forward_path)

    # Create DataFrames for each hop length and union with allowMissingColumns
    result_dfs = []
    for forward_hop, paths in forward_paths_by_hop.items():
        df = spark.createDataFrame(paths)
        result_dfs.append(df)

    # Union with allowMissingColumns=True since different hops have different columns
    result = result_dfs[0]
    for df in result_dfs[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result.distinct().coalesce(1)


def get_backward_paths(spark, target_paths_df: DataFrame) -> DataFrame:
    """
    Decompose bipaths into their backward path components.

    DECOMPOSITION FUNCTION: Splits each bipath into its backward component.
    For a bipath with hop_k edges, the backward path has length hop_k // 2.
    This function extracts the latter portion of each bipath and remaps indices to 0-based.

    Args:
        spark: SparkSession for creating DataFrames
        target_paths_df: DataFrame with bipath type columns (hop_k, node_i_type, edge_i_type, ...)

    Returns:
        DataFrame with backward path types:
        - hop_k: backward path length = original_hop_k // 2
        - node_i_type: node types remapped from node_{forward_hop+i} to node_i
        - edge_i_type: edge types remapped from edge_{forward_hop+i} to edge_i
        - u_index_of_edge_i, v_index_of_edge_i: edge endpoint indices (remapped)

    Example:
        5-hop bipath -> 2-hop backward path (nodes 3-5 -> 0-2, edges 3-4 -> 0-1)
        4-hop bipath -> 2-hop backward path (nodes 2-4 -> 0-2, edges 2-3 -> 0-1)
        1-hop bipath -> 0-hop backward path (node 1 -> 0, no edges)
    """

    rows = target_paths_df.collect()

    # Early validation: fail fast if no input data
    if not rows:
        raise ValueError("No bipath data provided to get_backward_paths")

    backward_paths_by_hop = {}  # Group by backward_hop to create consistent schemas

    for row in rows:
        bipath_hop_k = row['hop_k']
        forward_hop = (bipath_hop_k + 1) // 2
        backward_hop = bipath_hop_k // 2

        # Build backward path dict with remapped indices
        backward_path = {'hop_k': backward_hop}

        if backward_hop == 0:
            # 0-hop backward path: only has a single node (from 1-hop bipath)
            # The single node is the join node at position forward_hop
            backward_path['node_0_type'] = row[f'node_{forward_hop}_type']
        else:
            # Multi-hop backward path: remap nodes and edges
            # Remap node types: node_{forward_hop+i} -> node_i
            for i in range(backward_hop + 1):
                orig_idx = forward_hop + i
                backward_path[f'node_{i}_type'] = row[f'node_{orig_idx}_type']

            # Remap edge types and indices: edge_{forward_hop+i} -> edge_i
            for i in range(backward_hop):
                orig_idx = forward_hop + i
                backward_path[f'edge_{i}_type'] = row[f'edge_{orig_idx}_type']
                # Remap u_index and v_index: subtract forward_hop to get 0-based indices
                backward_path[f'u_index_of_edge_{i}'] = row[f'u_index_of_edge_{orig_idx}'] - forward_hop
                backward_path[f'v_index_of_edge_{i}'] = row[f'v_index_of_edge_{orig_idx}'] - forward_hop

        # Group by backward_hop
        if backward_hop not in backward_paths_by_hop:
            backward_paths_by_hop[backward_hop] = []
        backward_paths_by_hop[backward_hop].append(backward_path)

    # Create DataFrames for each hop length and union with allowMissingColumns
    result_dfs = []
    for backward_hop, paths in backward_paths_by_hop.items():
        df = spark.createDataFrame(paths)
        result_dfs.append(df)

    # Union with allowMissingColumns=True since different hops have different columns
    result = result_dfs[0]
    for df in result_dfs[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result.distinct().coalesce(1)
