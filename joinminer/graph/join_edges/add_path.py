"""
Add Path to Path with Deduplication

This module provides functionality to join two paths while ensuring no duplicate
edges exist in the resulting path. This is essential for valid path exploration
in graph-based link prediction.

This module orchestrates salting for data skew handling and deduplication logic.
Supports joining paths of arbitrary lengths (left path must have at least one edge).

IMPORTANT: This function is only for joining paths to existing paths (left_hop_k > 0).
For the first hop (adding edges to seed nodes), use a direct join instead.
"""

import logging
from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import broadcast

from joinminer.spark.operations.salt import salt_skewed_keys, replicate_for_salted_join
from .unique_edge_join import _unique_edge_join

logger = logging.getLogger(__name__)


def _get_path_type_columns(hop_k: int) -> List[str]:
    """
    Get path type column names for k-hop paths.

    Path type is defined by node types, edge types, and u/v positions for all edges.
    Node types are included to distinguish paths with different node type sequences.

    Args:
        hop_k: Total path length

    Returns:
        List of path type column names
    """
    columns = []
    # Add all node types first (from node_0 to node_{hop_k})
    for node_index in range(hop_k + 1):
        columns.append(f'node_{node_index}_type')
    # Then add edge types and positions
    for edge_index in range(hop_k):
        columns.append(f'edge_{edge_index}_type')
        columns.append(f'u_index_of_edge_{edge_index}')
        columns.append(f'v_index_of_edge_{edge_index}')
    return columns


def add_path_types_to_path(
    path_df: DataFrame,
    path_type_df: DataFrame,
    path_length: int,
    additional_join_columns: List[str] = None
) -> DataFrame:
    """
    Add path type specifications to paths by inner join.

    This function performs an inner join between path data and path type specifications to:
    1. Filter paths to only those matching target bipath patterns
    2. Enrich paths with specifications for the next edge to add

    The path type columns are systematically generated based on path length.
    Additional columns (e.g., seed_node_side) can be specified for task-specific joins.

    Args:
        path_df: DataFrame with paths of given length (0 for seed nodes)
        path_type_df: DataFrame with target path type specifications
        path_length: Number of edges in path_df (0 for seed nodes, 1 for hop 2, etc.)
        additional_join_columns: Extra columns for join (e.g., ['seed_node_side'])

    Returns:
        Filtered and enriched DataFrame ready for joining with outer edges

    Example:
        >>> # For seed nodes (path_length=0)
        >>> enriched_seeds = add_path_types_to_path(
        ...     path_df=seed_nodes,  # Has: node_0_id, node_0_type, seed_node_side
        ...     path_type_df=hop_1_types,  # Has: seed_node_side, node_0/1_type, edge_0_type, indices
        ...     path_length=0,
        ...     additional_join_columns=['seed_node_side']
        ... )
        >>> # enriched_seeds now has edge_0_type, node_1_type, indices added

        >>> # For paths with 1 edge (path_length=1)
        >>> enriched_paths = add_path_types_to_path(
        ...     path_df=hop_1_paths,  # Has: full hop 1 path structure
        ...     path_type_df=hop_2_types,  # Has: complete hop 2 path specifications
        ...     path_length=1,
        ...     additional_join_columns=['seed_node_side']
        ... )
        >>> # enriched_paths now has edge_1_type, node_2_type, indices added
    """
    logger.info(f"Adding path types to paths with length {path_length}")

    # Get path type columns systematically (works for 0 too)
    path_type_columns = _get_path_type_columns(path_length)

    # Combine with additional columns
    join_columns = (additional_join_columns or []) + path_type_columns

    logger.debug(f"  Join columns: {join_columns}")

    # Simple inner join with broadcast for small path_type_df
    result_df = path_df.join(
        broadcast(path_type_df),
        on=join_columns,
        how='inner'
    )

    logger.info(f"  Added path types to paths with length {path_length}")

    return result_df


def add_path_to_path(
    spark_ctx,
    left_path_df: DataFrame,
    right_path_df: DataFrame,
    join_columns: List[str],
    left_hop_k: int,
    right_hop_k: int,
    release_point: str,
    skew_threshold: int = 1000,
    salt_buckets: int = 100
) -> DataFrame:
    """
    Join two paths while ensuring no duplicate edges in the result.

    This function joins left and right paths at their connection point,
    automatically preventing any duplicate edges from appearing in the
    resulting combined path. It handles data skew through salting when
    needed and ensures semantic correctness by checking all edge pairs.

    IMPORTANT: This function requires left_hop_k > 0 (left path must have existing edges).
    For the first hop (adding edges to seed nodes with left_hop_k=0), use a direct join
    instead to avoid unnecessary overhead.

    An edge is considered duplicate if it has:
    - Same node IDs at both endpoints
    - Same node types at both endpoints
    - Same direction (same u_index and v_index)

    Args:
        spark_ctx: Spark context with fileio and persist_manager
        left_path_df: DataFrame containing left paths with left_hop_k edges
        right_path_df: DataFrame containing right paths with right_hop_k edges
        join_columns: Columns to join on (intersection node + optional salt)
        left_hop_k: Number of edges in left path (must be > 0)
        right_hop_k: Number of edges in right path (typically 1 for single edge)
        release_point: Persist lifecycle release point for memory management
        skew_threshold: Threshold for detecting skewed keys (default: 1000)
        salt_buckets: Number of salt buckets for handling skew (default: 100)

    Returns:
        DataFrame with joined paths containing left_hop_k + right_hop_k edges,
        with no duplicate edges

    Examples:
        >>> # Add single edge to paths (traditional use case)
        >>> extended_paths = add_path_to_path(
        ...     spark_ctx=spark_ctx,
        ...     left_path_df=hop_1_paths,      # Has 1 edge
        ...     right_path_df=outer_edges,     # Single edge
        ...     join_columns=['node_1_id', 'node_1_type', 'date'],
        ...     left_hop_k=1,
        ...     right_hop_k=1,
        ...     release_point='hop_2_done'
        ... )

        >>> # Join multi-hop paths (bipath construction)
        >>> bipaths = add_path_to_path(
        ...     spark_ctx=spark_ctx,
        ...     left_path_df=forward_paths,    # Has 2 edges
        ...     right_path_df=backward_paths,  # Has 1 edge
        ...     join_columns=['node_2_id', 'node_2_type'],
        ...     left_hop_k=2,
        ...     right_hop_k=1,
        ...     release_point='bipath_done'
        ... )
    """
    # Validate inputs
    if left_hop_k <= 0 or right_hop_k < 0:
        raise ValueError(
            f"left_hop_k must be > 0 (for paths with existing edges), "
            f"right_hop_k must be >= 0, "
            f"got left_hop_k={left_hop_k}, right_hop_k={right_hop_k}. "
            f"For first hop (left_hop_k=0), use direct join instead."
        )

    # Join paths with data skew handling and duplicate edge prevention
    logger.info(f"Joining {left_hop_k}-edge path with {right_hop_k}-edge path "
                f"(with salting and deduplication)")

    # Track salt columns for cleanup
    salt_columns = []

    # Step 1: Add salt to left DataFrame (paths) to handle data skew
    logger.debug("Adding salt to left paths for skew handling")
    left_salted, skewed_keys = salt_skewed_keys(
        spark_ctx,
        left_path_df,
        join_columns,
        skew_threshold,
        salt_buckets,
        "salt_value",
        release_point
    )

    # Step 2: If there are skewed keys, replicate right DataFrame (paths/edges)
    if skewed_keys is not None:
        logger.debug("Skewed keys detected, replicating right paths")
        right_replicated = replicate_for_salted_join(
            spark_ctx,
            right_path_df,
            join_columns,
            skewed_keys,
            "salt_value",
            release_point
        )
        salt_columns.append("salt_value")
    else:
        logger.debug("No skewed keys detected, proceeding without replication")
        right_replicated = right_path_df

    # Step 3: Perform unique edge join (ensures no duplicate edges in combined path)
    logger.debug("Performing unique edge join to prevent duplicate edges")
    result_df = _unique_edge_join(
        left_df=left_salted,
        right_df=right_replicated,
        join_columns=join_columns + salt_columns,  # Include salt columns in join
        left_hop_k=left_hop_k,
        right_hop_k=right_hop_k
    )

    # Step 4: Clean up salt columns
    if salt_columns:
        logger.debug(f"Removing salt columns: {salt_columns}")
        result_df = result_df.drop(*salt_columns)

    logger.info(f"Successfully joined paths with deduplication "
                f"(result has {left_hop_k + right_hop_k} edges)")

    return result_df


__all__ = [
    '_get_path_type_columns',
    'add_path_types_to_path',
    'add_path_to_path',
]
