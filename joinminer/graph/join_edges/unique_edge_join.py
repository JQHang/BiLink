"""
Unique Edge Join with Duplicate Detection

Provides pure duplicate detection logic for joining path DataFrames without
including any data skew handling. This allows the deduplication logic to be
reused in various contexts with or without salting.
"""

import logging
import operator
from functools import reduce
from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, array, element_at

logger = logging.getLogger(__name__)


def _unique_edge_join(
    left_df: DataFrame,
    right_df: DataFrame,
    join_columns: List[str],
    left_hop_k: int,
    right_hop_k: int
) -> DataFrame:
    """
    Join two path DataFrames while preventing duplicate edges.

    This function performs a pure duplicate-free join without any data skew
    handling. It checks all edge pairs between left and right paths to ensure
    no edge appears twice in the resulting paths.

    An edge is considered duplicate if it has:
    - Same node IDs at both endpoints
    - Same node types at both endpoints
    - Same direction (same u_index and v_index resolving to same nodes)

    Args:
        left_df: DataFrame containing left paths with left_hop_k edges
        right_df: DataFrame containing right paths with right_hop_k edges
        join_columns: Columns to join on (may include salt columns)
        left_hop_k: Number of edges in left paths
        right_hop_k: Number of edges in right paths

    Returns:
        DataFrame with joined paths, duplicate edges filtered out

    Example:
        >>> # Join two single-edge paths
        >>> result = _unique_edge_join(
        ...     left_df=paths_with_1_edge,
        ...     right_df=edges_to_add,
        ...     join_columns=['node_1_id', 'node_1_type', 'date'],
        ...     left_hop_k=1,
        ...     right_hop_k=1
        ... )
    """
    # Validate inputs
    if left_hop_k < 0 or right_hop_k < 0:
        raise ValueError(f"hop_k must be >= 0, got left_hop_k={left_hop_k}, right_hop_k={right_hop_k}")

    logger.debug(f"Performing unique edge join: left_hop_k={left_hop_k}, right_hop_k={right_hop_k}")

    # Check for duplicate columns (excluding join columns)
    left_cols = set(left_df.columns) - set(join_columns)
    right_cols = set(right_df.columns) - set(join_columns)
    duplicate_cols = left_cols & right_cols

    if duplicate_cols:
        raise ValueError(
            f"Found duplicate column(s) between DataFrames: {duplicate_cols}. "
            f"Please rename these columns before joining."
        )

    # Step 1: Build node ID and type arrays for left path
    # Left path has nodes from 0 to left_hop_k (total: left_hop_k + 1 nodes)
    left_max_nodes = left_hop_k + 1

    left_id_array_name = "_temp_left_node_ids"
    left_node_id_cols = [col(f'node_{i}_id') for i in range(left_max_nodes)]
    left_df = left_df.withColumn(left_id_array_name, array(*left_node_id_cols))

    left_type_array_name = "_temp_left_node_types"
    left_type_cols = [col(f'node_{i}_type') for i in range(left_max_nodes)]
    left_df = left_df.withColumn(left_type_array_name, array(*left_type_cols))

    # Step 2: Build node arrays for right path
    # Right path has nodes from left_hop_k to left_hop_k + right_hop_k
    # (total: right_hop_k + 1 nodes)
    right_max_nodes = right_hop_k + 1

    right_id_array_name = "_temp_right_node_ids"
    right_node_id_cols = [col(f'node_{i}_id') for i in range(left_hop_k, left_hop_k + right_max_nodes)]
    right_df = right_df.withColumn(right_id_array_name, array(*right_node_id_cols))

    right_type_array_name = "_temp_right_node_types"
    right_type_cols = [col(f'node_{i}_type') for i in range(left_hop_k, left_hop_k + right_max_nodes)]
    right_df = right_df.withColumn(right_type_array_name, array(*right_type_cols))

    # Step 3: Build basic join conditions
    basic_join_conditions = []
    for join_col in join_columns:
        basic_join_conditions.append(
            left_df[join_col] == right_df[join_col]
        )

    # Step 4: Build duplicate edge detection conditions
    # Check all edge pairs: each edge in left path vs each edge in right path
    duplicate_check_conditions = []

    for left_edge_index in range(left_hop_k):
        for right_edge_index in range(left_hop_k, left_hop_k + right_hop_k):
            # Get u/v index column names for both edges
            left_u_index_col = f'u_index_of_edge_{left_edge_index}'
            left_v_index_col = f'v_index_of_edge_{left_edge_index}'
            right_u_index_col = f'u_index_of_edge_{right_edge_index}'
            right_v_index_col = f'v_index_of_edge_{right_edge_index}'

            # Extract node IDs for left edge using element_at (1-based indexing)
            # Left edge u and v nodes are within range [0, left_hop_k]
            left_u_id = element_at(
                col(f"left.{left_id_array_name}"),
                (col(f"left.{left_u_index_col}") + 1).cast("int")
            )
            left_v_id = element_at(
                col(f"left.{left_id_array_name}"),
                (col(f"left.{left_v_index_col}") + 1).cast("int")
            )

            # Extract node IDs for right edge
            # Right edge u and v nodes are within range [left_hop_k, left_hop_k + right_hop_k]
            # Need to adjust index: array index = global index - left_hop_k
            right_u_id = element_at(
                col(f"right.{right_id_array_name}"),
                (col(f"right.{right_u_index_col}") - left_hop_k + 1).cast("int")
            )
            right_v_id = element_at(
                col(f"right.{right_id_array_name}"),
                (col(f"right.{right_v_index_col}") - left_hop_k + 1).cast("int")
            )

            # Extract node types for left edge
            left_u_type = element_at(
                col(f"left.{left_type_array_name}"),
                (col(f"left.{left_u_index_col}") + 1).cast("int")
            )
            left_v_type = element_at(
                col(f"left.{left_type_array_name}"),
                (col(f"left.{left_v_index_col}") + 1).cast("int")
            )

            # Extract node types for right edge
            right_u_type = element_at(
                col(f"right.{right_type_array_name}"),
                (col(f"right.{right_u_index_col}") - left_hop_k + 1).cast("int")
            )
            right_v_type = element_at(
                col(f"right.{right_type_array_name}"),
                (col(f"right.{right_v_index_col}") - left_hop_k + 1).cast("int")
            )

            # Edge is NOT duplicate if:
            # - Node IDs differ OR node types differ
            edge_not_duplicate = (
                (left_u_id != right_u_id) |
                (left_v_id != right_v_id) |
                (left_u_type != right_u_type) |
                (left_v_type != right_v_type)
            )

            duplicate_check_conditions.append(edge_not_duplicate)

    # Step 5: Combine all conditions and perform join
    all_conditions = basic_join_conditions + duplicate_check_conditions
    final_condition = reduce(operator.and_, all_conditions)

    logger.debug(f"Joining with {len(basic_join_conditions)} basic conditions and "
                f"{len(duplicate_check_conditions)} duplicate check conditions")

    joined_df = left_df.alias("left").join(
        right_df.alias("right"),
        final_condition,
        "inner"
    )

    # Step 6: Select columns and remove temporary arrays
    select_cols = []

    # Collect all temporary array column names
    temp_array_cols = {
        left_id_array_name,
        left_type_array_name,
        right_id_array_name,
        right_type_array_name
    }

    # Keep all left columns except temporary arrays
    for col_name in left_df.columns:
        if col_name not in temp_array_cols:
            select_cols.append(col(f"left.{col_name}").alias(col_name))

    # Keep all right columns except join columns and temporary arrays
    for col_name in right_df.columns:
        if col_name not in join_columns and col_name not in temp_array_cols:
            select_cols.append(col(f"right.{col_name}").alias(col_name))

    result_df = joined_df.select(*select_cols)

    logger.debug("Unique edge join completed successfully")

    return result_df
