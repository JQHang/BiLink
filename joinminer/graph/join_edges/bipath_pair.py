"""
Bipath Pair Generation Module

This module provides functionality to generate candidate pairs (u_node, v_node) from
matched bipaths by reading explored forward/backward paths and joining them.

The process involves:
1. Reading forward paths from candidate_path_exploration/hop_{forward_hop}/seed_node_side=0
2. Reading backward paths from candidate_path_exploration/hop_{backward_hop}/seed_node_side=1
3. Filtering paths by target path types (from hop_paths_df)
4. Transforming backward paths to align schema with forward paths
5. Joining paths at intersection node to form complete bipaths
6. Extracting (u_node, v_node) pairs from bipaths
7. Filtering out existing edges
8. Sampling to control pair explosion
9. Saving results to bipath_pairs/hop_{hop_k}

Features:
- Schema transformation for backward paths (reverses node/edge indices)
- Path type filtering before join (reduces computation)
- Duplicate edge prevention
- Sampling per (u_node, path_type) group
- Completion tracking for fault recovery
"""

import logging
from typing import List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from joinminer.spark.io import read_table, write_table
from joinminer.spark.operations.sample import random_sample
from .add_path import add_path_to_path, add_path_types_to_path, _get_path_type_columns
from .bipath import _transform_backward_path
from .bipath_types import get_forward_paths, get_backward_paths

logger = logging.getLogger(__name__)


def generate_bipath_pairs_for_hop(
    spark_ctx,
    hop_k: int,
    hop_paths_df: DataFrame,
    task_data_dir: str,
    graph,
    existing_edges_df: DataFrame,
    partition_instances: List[List[str]],
    bilink_config: dict
) -> None:
    """
    Generate bipath pairs for a specific hop length.

    Reads forward and backward paths from candidate_path_exploration, joins them
    to form complete bipaths, and extracts (u_node, v_node) pairs. Saves results
    to {task_data_dir}/bipath_pairs/hop_{hop_k}.

    Args:
        spark_ctx: Spark context with fileio, persist_manager, and table_state
        hop_k: Total bipath length (number of edges)
        hop_paths_df: DataFrame with target path types for this hop (already filtered by hop_k)
        task_data_dir: Base directory for task data (with URI scheme)
        graph: Graph instance with node type information and partition columns
        existing_edges_df: DataFrame of existing edges to filter out (u_node_id, v_node_id)
        partition_instances: Partition instances to process
        bilink_config: BiLink configuration dictionary containing max_instance and seed_nodes

    Example:
        >>> generate_bipath_pairs_for_hop(
        ...     spark_ctx=spark_ctx,
        ...     hop_k=2,
        ...     hop_paths_df=hop_paths_df,
        ...     task_data_dir='file:///data/task',
        ...     graph=graph,
        ...     existing_edges_df=existing_edges_df,
        ...     partition_instances=[['2021-01-01'], ['2022-01-01']],
        ...     bilink_config=bilink_config
        ... )
    """
    output_path = f"{task_data_dir}/bipath_pairs/hop_{hop_k}"

    # Check if already completed
    is_complete, missing_partitions = spark_ctx.table_state.check_complete(
        output_path,
        partition_columns=graph.partition_columns,
        partition_instances=partition_instances
    )
    if is_complete:
        logger.info(f"  Bipath pairs for hop {hop_k} already computed for all partitions")
        return

    logger.info(
        f"  Generating bipath pairs for hop {hop_k} for {len(missing_partitions)} "
        f"partition(s): {missing_partitions}"
    )

    # Calculate forward and backward hop lengths
    forward_hop_k = (hop_k + 1) // 2
    backward_hop_k = hop_k // 2

    logger.info(f"    Forward hops: {forward_hop_k}, Backward hops: {backward_hop_k}")

    # Get max_instance from config
    max_instance = bilink_config['bipath_discovery']['max_instance']

    # Define release point for persist lifecycle management
    release_point = f'bipath_hop_{hop_k}_done'

    # Step 1: Read forward paths from candidate_path_exploration
    forward_path_path = f"{task_data_dir}/candidate_path_exploration/hop_{forward_hop_k}"
    logger.info(f"    Reading forward paths from: {forward_path_path}")

    forward_path_df = read_table(
        spark_ctx,
        forward_path_path,
        partition_columns=graph.partition_columns + ['seed_node_side'],
        partition_instances=[p + [0] for p in missing_partitions]  # seed_node_side=0
    )

    # Drop seed_node_side immediately after reading
    forward_path_df = forward_path_df.drop('seed_node_side')

    # Extract forward path types for filtering
    forward_path_types_df = get_forward_paths(spark_ctx.spark, hop_paths_df)

    # Step 2: Filter forward paths by target path types using broadcast join
    logger.info("    Filtering forward paths by target path types...")
    forward_path_df = forward_path_df.join(
        F.broadcast(forward_path_types_df),
        on=forward_path_types_df.columns,
        how='inner'
    )

    # Step 3: Read backward paths
    if backward_hop_k == 0:
        # Read seed nodes from configuration (v_node side)
        logger.info("    Reading seed nodes for v_node (backward_hop_k=0)...")

        seed_config = bilink_config['bipath_discovery']['exploration']['seed_nodes']
        v_node_config = seed_config['v_node']
        v_node_type = bilink_config['target_edge']['v_node_type']
        v_node_type_index = graph.nodes[v_node_type]['type_index']

        backward_path_df = read_table(
            spark_ctx,
            v_node_config['path'],
            partition_columns=graph.partition_columns,
            partition_instances=missing_partitions
        )

        # Create 0-hop path schema
        backward_path_df = backward_path_df.select(
            F.col(v_node_config['id_column']).alias('node_0_id'),
            F.lit(v_node_type_index).alias('node_0_type'),
            *[F.col(c) for c in graph.partition_columns]
        )

    else:  # backward_hop_k > 0
        backward_path_path = f"{task_data_dir}/candidate_path_exploration/hop_{backward_hop_k}"
        logger.info(f"    Reading backward paths from: {backward_path_path}")

        backward_path_df = read_table(
            spark_ctx,
            backward_path_path,
            partition_columns=graph.partition_columns + ['seed_node_side'],
            partition_instances=[p + [1] for p in missing_partitions]  # seed_node_side=1
        )

        # Drop seed_node_side immediately after reading
        backward_path_df = backward_path_df.drop('seed_node_side')

        # Extract backward path types for filtering
        backward_path_types_df = get_backward_paths(spark_ctx.spark, hop_paths_df)

        # Step 4: Filter backward paths by target path types using broadcast join
        logger.info("    Filtering backward paths by target path types...")
        backward_path_df = backward_path_df.join(
            F.broadcast(backward_path_types_df),
            on=backward_path_types_df.columns,
            how='inner'
        )

        # Step 5: Transform backward paths to align with bipath schema
        logger.info("    Transforming backward paths...")
        backward_path_df = _transform_backward_path(
            spark_ctx=spark_ctx,
            backward_path_df=backward_path_df,
            forward_hop_k=forward_hop_k,
            backward_hop_k=backward_hop_k,
            partition_columns=graph.partition_columns,
            max_instance=max_instance,
            release_point=release_point
        )

    # Step 6: Add backward path type specifications to forward paths
    logger.info("    Adding backward path type specifications to forward paths...")
    # Filter columns before passing to add_path_types_to_path
    path_type_columns = _get_path_type_columns(hop_k)
    hop_paths_df_filtered = hop_paths_df.select(*path_type_columns)

    forward_path_df = add_path_types_to_path(
        path_df=forward_path_df,
        path_type_df=hop_paths_df_filtered,
        path_length=forward_hop_k,
        additional_join_columns=[]
    )

    # Step 7: Join forward and backward paths at intersection node
    logger.info("    Joining forward and backward paths...")

    # Generate join columns based on backward_hop_k
    if backward_hop_k == 0:
        # Only intersection node and partition columns
        join_columns = [
            f'node_{forward_hop_k}_id',
            f'node_{forward_hop_k}_type'
        ] + graph.partition_columns
    else:
        # Include intersection node, backward path type columns, and partition columns
        join_columns = [
            f'node_{forward_hop_k}_id',
            f'node_{forward_hop_k}_type'
        ]
        # Add edge type columns for backward portion
        for i in range(forward_hop_k, hop_k):
            join_columns.extend([
                f'edge_{i}_type',
                f'u_index_of_edge_{i}',
                f'v_index_of_edge_{i}'
            ])
        # Add node type columns for backward portion
        for i in range(forward_hop_k + 1, hop_k + 1):
            join_columns.append(f'node_{i}_type')
        # Add partition columns
        join_columns.extend(graph.partition_columns)

    bipath_df = add_path_to_path(
        spark_ctx=spark_ctx,
        left_path_df=forward_path_df,
        right_path_df=backward_path_df,
        join_columns=join_columns,
        left_hop_k=forward_hop_k,
        right_hop_k=backward_hop_k,
        release_point=release_point,
        skew_threshold=1000,  # Deprecated but kept for compatibility
        salt_buckets=100  # Deprecated but kept for compatibility
    )

    # Step 8: Distinct to (u, v, path_type) level - 直接去重
    logger.info("    Extracting (u, v, path_type) tuples from bipaths...")
    pair_path_df = bipath_df.select(
        F.col('node_0_id').alias('u_node_id'),
        F.col(f'node_{hop_k}_id').alias('v_node_id'),
        *[f'edge_{i}_type' for i in range(hop_k)],
        *[f'{side}_index_of_edge_{i}' for i in range(hop_k) for side in ['u',
    'v']],
        *[F.col(c) for c in graph.partition_columns]
    ).distinct()

    # Step 9: Filter out existing edges
    logger.info("    Filtering out existing edges...")
    pair_path_df = pair_path_df.join(
        existing_edges_df,
        on=['u_node_id', 'v_node_id'] + graph.partition_columns,
        how='left_anti'
    )

    # Step 9: Sample per (u_node, path_type, partition)
    logger.info("    Sampling pairs per (u_node, path_type)...")
    path_type_columns = (
        [f'edge_{i}_type' for i in range(hop_k)] +
        [f'{side}_index_of_edge_{i}' for i in range(hop_k) for side in ['u', 'v']]
    )
    sample_group_columns = ['u_node_id'] + path_type_columns + graph.partition_columns
    pair_path_df = random_sample(pair_path_df, sample_group_columns, max_instance)

    # Step 10: Extract pairs (u_node, v_node) from bipaths
    logger.info("    Extracting candidate pairs from bipaths...")
    pairs_df = pair_path_df.select(
        'u_node_id', 'v_node_id', *graph.partition_columns
    ).distinct()

    # Step 11: Save results
    logger.info(f"    Saving pairs to: {output_path}")
    write_table(
        spark_ctx,
        pairs_df,
        output_path,
        partition_columns=graph.partition_columns,
        partition_instances=missing_partitions,
        mode='overwrite'
    )

    # Mark complete
    spark_ctx.table_state.mark_complete(
        output_path,
        partition_columns=graph.partition_columns,
        partition_instances=missing_partitions
    )
    logger.info(f"    Bipath pair generation for hop {hop_k} completed successfully")