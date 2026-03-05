"""
Bipath Computation Module

This module provides functionality to compute bi-paths by joining forward and backward paths
from different seed nodes. Bi-paths are essential for link prediction tasks where we need to
find meta-paths connecting two types of nodes.

A k-hop bipath is formed by:
- Forward path: from u_node (seed_node_side=0) with forward_hop_k edges
- Backward path: from v_node (seed_node_side=1) with backward_hop_k edges
- Joined at intersection node: node_{forward_hop_k}

Features:
- Automatic schema transformation for backward paths (reverses node/edge indices)
- Data skew handling through salting at intersection node
- Duplicate edge prevention using _unique_edge_join
- Path explosion control via sampling
- Completion tracking for fault recovery
"""

import logging
from typing import List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from joinminer.spark.io import read_table, write_table
from joinminer.spark.operations.sample import random_sample, skewed_random_sample
from .add_path import add_path_to_path, _get_path_type_columns

logger = logging.getLogger(__name__)


def compute_k_hop_bipaths(
    spark_ctx,
    hop_k: int,
    task_data_dir: str,
    partition_columns: List[str],
    max_instance: int,
    skew_threshold: int = 1000,
    salt_buckets: int = 100
) -> None:
    """
    Compute k-hop bipaths by joining forward and backward paths.

    Reads forward paths from hop_{forward_hop_k}/seed_node_side=0
    Reads backward paths from hop_{backward_hop_k}/seed_node_side=1
    Joins them at the intersection node to form bipaths
    Saves to {task_data_dir}/bipaths/hop_{hop_k}

    The bipath schema maintains uniform node numbering from 0 to hop_k:
    - node_0: u_node (starting point)
    - node_{hop_k}: v_node (ending point)
    - node_{forward_hop_k}: intersection node (where paths meet)

    Args:
        spark_ctx: Spark context with fileio, persist_manager, and table_state
        hop_k: Total length of bipath (forward_hop_k + backward_hop_k)
        task_data_dir: Base directory for task data (with URI scheme)
        partition_columns: Graph partition columns (e.g., ['date']) to preserve
        max_instance: Maximum bipath instances per group
        skew_threshold: (Deprecated) Kept for backward compatibility, not used
        salt_buckets: (Deprecated) Kept for backward compatibility, not used

    Note:
        Skew handling is now done automatically by skewed_random_sample with
        fixed parameters (threshold=10000, buckets=100)

    Example:
        >>> # Compute 3-hop bipaths (forward 2 hops, backward 1 hop)
        >>> compute_k_hop_bipaths(
        ...     spark_ctx=spark_ctx,
        ...     hop_k=3,
        ...     task_data_dir='file:///data/task',
        ...     partition_columns=['date'],
        ...     max_instance=1000
        ... )
    """
    output_path = f"{task_data_dir}/bipaths/hop_{hop_k}"

    # Check if already completed
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Bipaths for hop {hop_k} already computed at {output_path}")
        return

    logger.info(f"Computing {hop_k}-hop bipaths...")

    # Calculate forward and backward hop lengths
    # forward_hop_k is ceiling of hop_k/2, backward_hop_k is floor of hop_k/2
    forward_hop_k = (hop_k + 1) // 2
    backward_hop_k = hop_k // 2

    logger.info(f"  Forward hops: {forward_hop_k}, Backward hops: {backward_hop_k}")

    # Define release point for persist lifecycle management
    release_point = f'bipath_hop_{hop_k}_done'

    # Read forward paths from u_node side (seed_node_side=0)
    forward_path_path = f"{task_data_dir}/path_exploration/hop_{forward_hop_k}/seed_node_side=0"
    logger.info(f"  Reading forward paths from: {forward_path_path}")

    forward_path_df = read_table(spark_ctx, forward_path_path)

    # Read backward paths from v_node side (seed_node_side=1)
    backward_path_path = f"{task_data_dir}/path_exploration/hop_{backward_hop_k}/seed_node_side=1"
    logger.info(f"  Reading backward paths from: {backward_path_path}")

    backward_path_df = read_table(spark_ctx, backward_path_path)

    # Transform backward path to align with bipath schema
    logger.info("  Transforming backward paths...")
    backward_path_df = _transform_backward_path(
        spark_ctx=spark_ctx,
        backward_path_df=backward_path_df,
        forward_hop_k=forward_hop_k,
        backward_hop_k=backward_hop_k,
        partition_columns=partition_columns,
        max_instance=max_instance,
        release_point=release_point
    )

    # Compute join columns: intersection node + partition columns
    join_columns = [f'node_{forward_hop_k}_id', f'node_{forward_hop_k}_type'] + partition_columns

    # Join forward and backward paths with duplicate edge prevention
    logger.info("  Joining forward and backward paths with salting and deduplication...")
    bipath_df = add_path_to_path(
        spark_ctx=spark_ctx,
        left_path_df=forward_path_df,
        right_path_df=backward_path_df,
        join_columns=join_columns,
        left_hop_k=forward_hop_k,
        right_hop_k=backward_hop_k,
        release_point=release_point,
        skew_threshold=skew_threshold,
        salt_buckets=salt_buckets
    )

    # Distinct to (u, v, path_type, partition) level
    logger.info("  Deduplicating bipaths to (u, v, path_type) level...")
    group_columns = ['node_0_id', f'node_{hop_k}_id'] + _get_path_type_columns(hop_k) + partition_columns
    bipath_df = bipath_df.select(*group_columns).distinct()

    # Save results (no partitioning - bipaths are typically small)
    logger.info(f"  Saving bipaths to: {output_path}")
    write_table(
        spark_ctx,
        bipath_df,
        output_path,
        mode='overwrite'
    )

    # Mark complete
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"{hop_k}-hop bipath computation completed successfully")

    # Release persisted resources
    spark_ctx.persist_manager.mark_released(release_point)


def _transform_backward_path(
    spark_ctx,
    backward_path_df: DataFrame,
    forward_hop_k: int,
    backward_hop_k: int,
    partition_columns: List[str],
    max_instance: int,
    release_point: str
) -> DataFrame:
    """
    Transform backward path to align with bipath schema.

    Reverses node indices (e.g., node_0 becomes node_{bipath_hop_k})
    Reverses edge indices and flips u/v positions
    Samples with skew control before join

    Backward path schema (from v_node seed):
    - Input:  node_0 (v_node) -> ... -> node_{backward_hop_k} (intersection)
    - Output: node_{bipath_hop_k} (v_node) -> ... -> node_{forward_hop_k} (intersection)

    Uses skewed_random_sample with fixed parameters:
    - skew_threshold: 10000
    - salt_buckets: 100 (caps each group at ~1M records)

    Args:
        spark_ctx: Spark context with persist_manager
        backward_path_df: Backward paths from v_node seed
        forward_hop_k: Number of hops in forward path
        backward_hop_k: Number of hops in backward path
        partition_columns: Graph partition columns to preserve (e.g., ['date'])
        max_instance: Maximum instances per group for sampling
        release_point: Persist lifecycle release point

    Returns:
        Transformed DataFrame with reversed schema
    """
    bipath_hop_k = forward_hop_k + backward_hop_k

    logger.debug(f"Transforming backward path: bipath_hop_k={bipath_hop_k}")

    # Build select expressions for schema transformation
    select_expressions = []

    # Transform node ID and type columns (reverse indices)
    for hop_k in range(backward_hop_k + 1):
        # Map node_i to node_{bipath_hop_k - i}
        original_index = hop_k
        new_index = bipath_hop_k - hop_k

        select_expressions.append(
            F.col(f'node_{original_index}_id').alias(f'node_{new_index}_id')
        )
        select_expressions.append(
            F.col(f'node_{original_index}_type').alias(f'node_{new_index}_type')
        )

    # Transform edge type and position columns (reverse indices and flip u/v)
    for hop_k in range(backward_hop_k):
        # Map edge_i to edge_{bipath_hop_k - i - 1}
        original_index = hop_k
        new_index = bipath_hop_k - hop_k - 1

        # Edge type
        select_expressions.append(
            F.col(f'edge_{original_index}_type').alias(f'edge_{new_index}_type')
        )

        # Flip u/v indices: subtract from bipath_hop_k
        select_expressions.append(
            (F.lit(bipath_hop_k) - F.col(f'u_index_of_edge_{original_index}'))
            .alias(f'u_index_of_edge_{new_index}')
        )
        select_expressions.append(
            (F.lit(bipath_hop_k) - F.col(f'v_index_of_edge_{original_index}'))
            .alias(f'v_index_of_edge_{new_index}')
        )

    # Add partition columns (passthrough from graph configuration)
    for col_name in partition_columns:
        if col_name in backward_path_df.columns:
            select_expressions.append(F.col(col_name))

    # Apply transformation
    transformed_df = backward_path_df.select(*select_expressions)

    # Get sampling columns for grouping (intersection node + path type + partition columns)
    sampling_columns = _get_backward_sampling_columns(forward_hop_k, bipath_hop_k, partition_columns)

    # Sample with skew control at intersection node to prevent excessive fan-out
    # Uses skewed_random_sample which handles pre-filtering automatically
    logger.debug("Sampling backward paths with skew control...")
    transformed_df = skewed_random_sample(
        spark_ctx=spark_ctx,
        df=transformed_df,
        group_columns=sampling_columns,  # Group by intersection node + path type + partition
        n=max_instance,
        release_point=release_point
    )

    return transformed_df


def _get_backward_sampling_columns(
    forward_hop_k: int,
    bipath_hop_k: int,
    partition_columns: List[str]
) -> List[str]:
    """
    Get grouping columns for sampling backward paths.

    Sampling columns include:
    - Intersection node ID and type (node_{forward_hop_k}_id, node_{forward_hop_k}_type)
    - Path type columns for backward edges (edge types and u/v indices)
    - Partition columns (e.g., 'date') to group within same partition

    This ensures we sample paths evenly across different intersection nodes,
    different path types, and respect partition boundaries.

    Args:
        forward_hop_k: Number of hops in forward path
        bipath_hop_k: Total bipath length
        partition_columns: Graph partition columns (e.g., ['date'])

    Returns:
        List of column names for grouping in skewed_random_sample
    """
    sampling_columns = []

    # Intersection node columns
    sampling_columns.append(f'node_{forward_hop_k}_id')
    sampling_columns.append(f'node_{forward_hop_k}_type')

    # Path type columns for backward portion (to group same path types)
    for edge_index in range(forward_hop_k, bipath_hop_k):
        sampling_columns.append(f'edge_{edge_index}_type')
        sampling_columns.append(f'u_index_of_edge_{edge_index}')
        sampling_columns.append(f'v_index_of_edge_{edge_index}')

    # Partition columns to ensure sampling respects partition boundaries
    sampling_columns.extend(partition_columns)

    return sampling_columns


def _get_matched_bipaths_schema(hop_k: int):
    """
    Get schema for matched bipaths table.

    Args:
        hop_k: Number of hops in the bipath

    Returns:
        StructType schema for matched bipaths table
    """
    from pyspark.sql.types import (
        StructType, StructField, LongType, DoubleType, IntegerType
    )

    path_type_columns = _get_path_type_columns(hop_k)
    return StructType([
        *[StructField(col, IntegerType(), True) for col in path_type_columns],
        StructField('tgt_pair_count', LongType(), False),
        StructField('all_pair_count', LongType(), False),
        StructField('tgt_pair_percent', DoubleType(), False),
        StructField('hop_k', IntegerType(), False)
    ])


def match_k_hop_bipaths(
    spark_ctx,
    hop_k: int,
    task_data_dir: str,
    graph,
    u_node_type: str,
    v_node_type: str,
    partition_columns: List[str],
    partition_instances: List[List[str]],
    max_instance: int
) -> None:
    """
    Match k-hop bipaths with target pairs and calculate statistics.

    This function:
    1. Reads computed bipaths from {task_data_dir}/bipaths/hop_{hop_k}
    2. Filters out pairs that exist in existing_edges
    3. Samples paths per (u_node, path_type) to control explosion
    4. Joins with target_pairs to identify which paths connect target nodes
    5. Aggregates statistics by path_type
    6. Saves results as Parquet table and returns DataFrame

    Args:
        spark_ctx: Spark context with fileio, persist_manager, and table_state
        hop_k: Total bipath length
        task_data_dir: Base directory for task data (with URI scheme)
        graph: Graph object for accessing node type indices
        u_node_type: Node type name for u_node (node_0) from target edge config
        v_node_type: Node type name for v_node (node_{hop_k}) from target edge config
        partition_columns: Graph partition columns (e.g., ['date'])
        partition_instances: Partition instances to process
        max_instance: Maximum instances per (u_node, path_type) group

    Saves results to {task_data_dir}/matched_bipaths/hop_{hop_k} with columns:
        - edge_0_type, u_index_of_edge_0, v_index_of_edge_0, ... (path type columns)
        - tgt_pair_count: Number of target pairs connected by this path type
        - all_pair_count: Total number of pairs connected by this path type
        - tgt_pair_percent: Ratio of target pairs (NULL if insufficient sample size)
        - hop_k: Total path length

    Returns:
        None (results saved to disk)

    Example:
        >>> match_k_hop_bipaths(
        ...     spark_ctx=spark_ctx,
        ...     hop_k=3,
        ...     task_data_dir='file:///data/task',
        ...     partition_columns=['date'],
        ...     partition_instances=[['2021-01-01']],
        ...     max_instance=1000
        ... )
        >>> # Read results from disk
        >>> schema = _get_matched_bipaths_schema(3)
        >>> stats_df = read_table(spark_ctx, 'file:///data/task/matched_bipaths/hop_3', schema=schema)
        >>> stats_df.orderBy(col('tgt_pair_percent').desc()).show()
    """
    logger.info(f"Matching {hop_k}-hop bipaths with target pairs...")

    # Check if already computed
    output_path = f"{task_data_dir}/matched_bipaths/hop_{hop_k}"
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Match results already exist at {output_path}, reading table...")
        schema = _get_matched_bipaths_schema(hop_k)
        return read_table(spark_ctx, output_path, schema=schema)

    # Define release point for persist lifecycle
    release_point = f'match_hop_{hop_k}_done'

    # Get node type indices from graph
    u_node_type_index = graph.nodes[u_node_type]['type_index']
    v_node_type_index = graph.nodes[v_node_type]['type_index']

    # Read computed bipaths
    bipath_path = f"{task_data_dir}/bipaths/hop_{hop_k}"
    logger.info(f"  Reading bipaths from: {bipath_path}")

    bipath_df = read_table(
        spark_ctx,
        bipath_path
    )

    # Get path type columns: all edge types and u/v positions
    path_type_columns = _get_path_type_columns(hop_k)

    # Read existing edges to filter out (from prepared exist_target_edges)
    existing_edges_path = f"{task_data_dir}/exist_target_edges"
    logger.info(f"  Reading existing edges from: {existing_edges_path}")

    existing_edges_df = read_table(
        spark_ctx,
        existing_edges_path,
        partition_columns=partition_columns,
        partition_instances=partition_instances
    )

    # Rename existing edges columns to match pair columns and add node types
    # Node types must be included to prevent ID collisions between different entity types
    existing_edges_df = existing_edges_df.select(
        F.col('u_node_id').alias('node_0_id'),
        F.col('v_node_id').alias(f'node_{hop_k}_id'),
        F.lit(u_node_type_index).alias('node_0_type'),
        F.lit(v_node_type_index).alias(f'node_{hop_k}_type'),
        *[F.col(c) for c in partition_columns]
    )

    # Filter out pairs that exist in existing_edges
    # Must include node types to prevent ID collisions
    logger.info("  Filtering out existing edges...")
    filter_columns = [
        'node_0_id', 'node_0_type',
        f'node_{hop_k}_id', f'node_{hop_k}_type'
    ] + partition_columns
    bipath_df = bipath_df.join(
        existing_edges_df,
        on=filter_columns,
        how='left_anti'
    )

    # Check if any bipaths remain after filtering
    spark_ctx.persist_manager.persist(
        bipath_df,
        release_point=release_point,
        name='filtered_bipaths'
    )

    if bipath_df.count() == 0:
        logger.info(f"  No valid bipaths remain after filtering for hop {hop_k}")

        # Just mark as complete without saving data
        output_path = f"{task_data_dir}/matched_bipaths/hop_{hop_k}"
        spark_ctx.table_state.mark_complete(output_path)

        # Clean up and return empty DataFrame with correct schema
        spark_ctx.persist_manager.mark_released(release_point)
        schema = _get_matched_bipaths_schema(hop_k)
        return spark_ctx.spark.createDataFrame([], schema)

    # Sample paths per (u_node, path_type, partition) to control explosion
    logger.info("  Sampling paths per (u_node, path_type, partition)...")
    sample_group_columns = ['node_0_id'] + path_type_columns + partition_columns
    bipath_df = random_sample(bipath_df, sample_group_columns, max_instance)

    # Read target pairs
    target_pairs_path = f"{task_data_dir}/target_pairs"
    logger.info(f"  Reading target pairs from: {target_pairs_path}")

    target_pairs_df = read_table(spark_ctx, target_pairs_path)

    # Rename target pairs columns to match pair columns and add node types
    # Node types must be included to prevent ID collisions between different entity types
    target_pairs_df = target_pairs_df.select(
        F.col('u_node_id').alias('node_0_id'),
        F.col('v_node_id').alias(f'node_{hop_k}_id'),
        F.lit(u_node_type_index).alias('node_0_type'),
        F.lit(v_node_type_index).alias(f'node_{hop_k}_type'),
        *[F.col(c) for c in partition_columns]
    )

    # Add match marker to target pairs
    target_pairs_df = target_pairs_df.withColumn('match_mark', F.lit(1))

    spark_ctx.persist_manager.persist(
        target_pairs_df,
        release_point=release_point,
        name='target_pairs_marked'
    )

    # Left join with target pairs to identify matches
    # Must include node types to prevent ID collisions between different entity types
    logger.info("  Joining with target pairs...")
    join_columns = [
        'node_0_id', 'node_0_type',
        f'node_{hop_k}_id', f'node_{hop_k}_type'
    ] + partition_columns
    pairs_match_df = bipath_df.join(
        target_pairs_df,
        on=join_columns,
        how='left'
    )

    # Aggregate statistics by path_type
    logger.info("  Aggregating statistics by path_type...")
    stats_df = pairs_match_df.groupBy(*path_type_columns).agg(
        F.sum(F.col('match_mark')).alias('tgt_pair_count'),
        F.count('*').alias('all_pair_count')
    ).fillna(0, subset=['tgt_pair_count'])

    # Calculate percentage when sample size is sufficient
    # NULL indicates insufficient sample size (statistical reliability depends on all_pair_count only)
    stats_df = stats_df.withColumn(
        'tgt_pair_percent',
        F.when(
            F.col('all_pair_count') > 0,
            F.col('tgt_pair_count') / F.col('all_pair_count')
        ).otherwise(0.0)
    )

    # Add hop_k column for partitioning
    stats_df = stats_df.withColumn('hop_k', F.lit(hop_k))

    # Save as Parquet table (write_table will log row count)
    output_path = f"{task_data_dir}/matched_bipaths/hop_{hop_k}"
    write_table(
        spark_ctx,
        stats_df,
        output_path,
        mode='overwrite'
    )

    # Mark as complete
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"  Saved match results table to: {output_path}")

    # Release persisted resources
    spark_ctx.persist_manager.mark_released(release_point)


__all__ = [
    'compute_k_hop_bipaths',
    '_get_matched_bipaths_schema',
    'match_k_hop_bipaths',
]
