#!/usr/bin/env python3
"""
Find Relevant Bi-paths for Link Discovery

This script identifies relevant bi-paths connecting source nodes to target nodes
through the knowledge graph for link prediction tasks. It is task-agnostic and
reads node types and column names from the BiLink configuration.

Features:
- YAML-driven graph configuration
- Support for multiple partition dates
- Configurable bi-path patterns
- Completion tracking for already processed partitions

Usage:
    python 0_relevant_bipaths_finder.py \
        --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \
        [--force] \
        [--verbose]
"""

import os
import argparse
import sys
import logging
from datetime import datetime
import pandas as pd

from joinminer import PROJECT_ROOT
from joinminer.graph import Graph
from joinminer.graph.join_edges import (
    add_path_to_path,
    compute_k_hop_bipaths,
    match_k_hop_bipaths,
    _get_matched_bipaths_schema
)
from joinminer.spark import SparkRunner
from joinminer.spark.io import read_table, write_table
from joinminer.spark.io.partition import parse_partition_spec
from joinminer.spark.operations.sample import random_sample
from joinminer.fileio import FileIO
from joinminer.utils import setup_logger
from pyspark.sql import functions as F

# Module-level logger
logger = logging.getLogger("joinminer")


# ==================== Helper Functions ====================

def prepare_target_pairs(spark_ctx, task_data_dir, max_target_pairs, bilink_config):
    """
    Prepare target pairs from known_target_edges (train partition only).
    Save to {task_data_dir}/target_pairs with _COMPLETE marker.

    Args:
        spark_ctx: Spark context with fileio
        task_data_dir: Path to task data directory (with URI scheme)
        max_target_pairs: Maximum number of target pairs to use
        bilink_config: BiLink configuration dictionary (for column name mapping)
    """
    output_path = f"{task_data_dir}/target_pairs"

    # Check if already completed
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Target pairs already prepared at {output_path}")
        return

    # Read known_target_edges (train partition only)
    known_edges_path = f"{task_data_dir}/known_target_edges/sample_type=train"
    logger.info(f"Reading target edges from: {known_edges_path}")
    target_pairs = read_table(spark_ctx, known_edges_path, format='parquet')

    # Select and rename columns using config-driven column names
    u_col = bilink_config['target_edge']['u_node_id_column']
    v_col = bilink_config['target_edge']['v_node_id_column']
    target_pairs = target_pairs.select(
        F.col(u_col).alias('u_node_id'),
        F.col(v_col).alias('v_node_id'),
        F.col('date')
    )

    # Sample if exceeds max_target_pairs
    total_count = target_pairs.count()
    if total_count > max_target_pairs:
        fraction = max_target_pairs / total_count
        target_pairs = target_pairs.sample(False, fraction, seed=42)
        logger.info(f"Sampled {max_target_pairs} from {total_count} target pairs")
    else:
        logger.info(f"Using all {total_count} target pairs")

    # Save to {task_data_dir}/target_pairs (no partitioning)
    write_table(spark_ctx, target_pairs, output_path)

    # Mark as complete
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Target pairs saved to {output_path}")


def generate_seed_nodes(spark_ctx, task_data_dir, graph, bilink_config, neg_ratio, partition_instances):
    """
    Generate seed nodes from target pairs with generic handling for both node types.
    Save to {task_data_dir}/path_exploration/hop_0 with _COMPLETE marker.

    Args:
        spark_ctx: Spark context with fileio
        task_data_dir: Path to task data directory
        graph: Graph object for accessing node type indices and element tables
        bilink_config: BiLink configuration for target edge types and seed node paths
        neg_ratio: Number of random nodes per target node
        partition_instances: Partition instances for filtering candidate nodes
    """
    output_path = f"{task_data_dir}/path_exploration/hop_0"

    # Check if already completed
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Seed nodes already generated at {output_path}")
        return

    # Get seed nodes configuration
    seed_config = bilink_config['bipath_discovery']['exploration']['seed_nodes']

    # Read and persist target pairs (used for both u and v nodes)
    target_pairs_path = f"{task_data_dir}/target_pairs"
    target_pairs = read_table(spark_ctx, target_pairs_path)
    spark_ctx.persist_manager.persist(
        target_pairs,
        release_point='seed_nodes_done',
        name='target_pairs'
    )

    all_seed_nodes = []

    # Process both node types uniformly
    for seed_node_side in ['u_node', 'v_node']:
        logger.info(f"Generating seed nodes for {seed_node_side}...")

        # Derive column name and node type index from current side
        target_col = f"{seed_node_side}_id"
        node_type = bilink_config['target_edge'][f'{seed_node_side}_type']
        node_type_index = graph.nodes[node_type]['type_index']

        # Get node configuration
        node_config = seed_config[seed_node_side]

        # Extract target nodes from target_pairs (with partition columns)
        select_cols = [F.col(target_col).alias('node_0_id')] + [F.col(c) for c in graph.partition_columns]
        target_nodes = target_pairs.select(*select_cols).distinct()
        spark_ctx.persist_manager.persist(
            target_nodes,
            release_point='seed_nodes_done',
            name=f'target_nodes_{seed_node_side}'
        )
        target_count = target_nodes.count()
        logger.info(f"Target {seed_node_side}s: {target_count}")

        # Read candidate nodes with partition filtering
        candidates_df = read_table(
            spark_ctx,
            node_config['path'],
            partition_columns=graph.partition_columns,
            partition_instances=partition_instances
        )

        # Prepare candidate nodes (rename id column, keep partition columns)
        candidate_nodes = candidates_df.select(
            F.col(node_config['id_column']).alias('node_0_id'),
            *[F.col(c) for c in graph.partition_columns]
        )

        # Filter out target nodes (based on node_0_id and partition columns)
        join_cols = ['node_0_id'] + graph.partition_columns
        candidates_filtered = candidate_nodes.join(
            target_nodes,
            on=join_cols,
            how='left_anti'
        )
        spark_ctx.persist_manager.persist(
            candidates_filtered,
            release_point='seed_nodes_done',
            name=f'candidates_filtered_{seed_node_side}'
        )

        # Sample random nodes
        random_count = int(target_count * neg_ratio)
        total_candidates = candidates_filtered.count()

        if total_candidates > random_count:
            fraction = random_count / total_candidates
            random_nodes = candidates_filtered.sample(False, fraction, seed=42)
        else:
            random_nodes = candidates_filtered

        spark_ctx.persist_manager.persist(
            random_nodes,
            release_point='seed_nodes_done',
            name=f'random_nodes_{seed_node_side}'
        )
        logger.info(f"Sampled {random_nodes.count()} random {seed_node_side}s")

        # Determine seed_node_side value (0 for u_node, 1 for v_node)
        side_value = 0 if seed_node_side == 'u_node' else 1

        # Combine target and random nodes, add node_0_type and seed_node_side
        seed_nodes = (
            target_nodes
            .union(random_nodes)
            .withColumn('node_0_type', F.lit(node_type_index))
            .withColumn('seed_node_side', F.lit(side_value))
            .select('node_0_id', 'node_0_type', 'seed_node_side', *graph.partition_columns)
        )

        all_seed_nodes.append(seed_nodes)

    # Combine all seed nodes (no distinct needed - seed_node_side makes rows unique)
    final_seed_nodes = all_seed_nodes[0].union(all_seed_nodes[1])

    # Save seed nodes (no partitioning for small table)
    write_table(
        spark_ctx,
        final_seed_nodes,
        output_path,
        partition_columns = ['seed_node_side']
    )
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Seed nodes saved to: {output_path}")

    # Release all remaining cached data
    spark_ctx.persist_manager.mark_released('seed_nodes_done')


def add_readable_type_columns(spark_ctx, graph, matched_bipath_df, max_hop):
    """
    Add human-readable type names to matched bipaths DataFrame and reorder columns.

    Adds *_type_name columns alongside numeric *_type columns for easier inspection.
    Uses broadcast join for efficient conversion of edge and node type indices to names.
    Reorders columns in logical sequence: hop_k, node_0, edge_0, node_1, edge_1, ..., statistics.

    Args:
        spark_ctx: Spark context with spark session
        graph: Graph object with node_types and edge_types
        matched_bipath_df: DataFrame with numeric type indices
        max_hop: Maximum hop number to determine column structure

    Returns:
        DataFrame with additional readable type columns in ordered sequence
    """
    logger.info("Adding human-readable type names...")

    # Create edge type mapping DataFrame (index -> name)
    edge_mapping_df = spark_ctx.spark.createDataFrame(
        list(graph.edge_index_to_type.items()),
        ["edge_id", "edge_name"]
    ).coalesce(1)  # Single partition for tiny lookup table

    # Create node type mapping DataFrame (index -> name)
    node_mapping_df = spark_ctx.spark.createDataFrame(
        list(graph.node_index_to_type.items()),
        ["node_id", "node_name"]
    ).coalesce(1)  # Single partition for tiny lookup table

    result_df = matched_bipath_df

    # Convert edge type columns (edge_0_type, edge_1_type, ..., edge_{max_hop-1}_type)
    for i in range(max_hop):
        edge_col = f'edge_{i}_type'
        if edge_col in result_df.columns:
            result_df = result_df.join(
                F.broadcast(edge_mapping_df),
                result_df[edge_col] == edge_mapping_df['edge_id'],
                'left'
            ).drop('edge_id').withColumnRenamed('edge_name', f'{edge_col}_name')

    # Convert node type columns if they exist (node_0_type, node_1_type, ..., node_{max_hop}_type)
    # Note: matched bipaths may not have node type columns, only edge types
    for i in range(max_hop + 1):
        node_col = f'node_{i}_type'
        if node_col in result_df.columns:
            result_df = result_df.join(
                F.broadcast(node_mapping_df),
                result_df[node_col] == node_mapping_df['node_id'],
                'left'
            ).drop('node_id').withColumnRenamed('node_name', f'{node_col}_name')

    # Reorder columns in logical sequence
    logger.info("Reordering columns for readability...")
    ordered_cols = ['hop_k']  # Start with hop_k

    # Add node and edge columns in sequence (node_0, edge_0, node_1, edge_1, ...)
    for i in range(max_hop + 1):
        # Add node_{i} columns if they exist (type first, then type_name)
        for suffix in ['_type', '_type_name']:
            col = f'node_{i}{suffix}'
            if col in result_df.columns:
                ordered_cols.append(col)

        # Add edge_{i} columns if i < max_hop
        if i < max_hop:
            # Order: edge_type, edge_type_name, u_index, v_index
            for suffix in ['_type', '_type_name']:
                col = f'edge_{i}{suffix}'
                if col in result_df.columns:
                    ordered_cols.append(col)

            for prefix in ['u_index_of_edge_', 'v_index_of_edge_']:
                col = f'{prefix}{i}'
                if col in result_df.columns:
                    ordered_cols.append(col)

    # Add statistics columns at the end (always present in matched bipaths schema)
    ordered_cols.extend(['tgt_pair_count', 'all_pair_count', 'tgt_pair_percent'])

    # Reorder DataFrame
    result_df = result_df.select(*ordered_cols)

    logger.info("Readable type columns added and reordered successfully")
    return result_df


def _process_single_hop(
    spark_ctx,
    hop_k: int,
    task_data_dir: str,
    join_edge_path: str,
    graph,
    partition_instances: list,
    max_hop: int,
    max_instance: int
):
    """
    Process a single hop of path exploration.

    Reads parent paths from hop_{hop_k-1}, joins with outer join edges,
    samples paths, and saves results to hop_{hop_k}.

    Args:
        spark_ctx: Spark context
        hop_k: Current hop number (1, 2, 3, ...)
        task_data_dir: Base directory for task data
        join_edge_path: Path to join edges (inner/outer)
        graph: Graph object for schema information
        partition_instances: Target partition instances for filtering edges
        max_hop: Maximum hop number for bipath exploration
        max_instance: Maximum number of path instances per group
    """
    output_path = f"{task_data_dir}/path_exploration/hop_{hop_k}"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Hop {hop_k} already complete, skipping")
        return

    logger.info(f"Processing hop {hop_k}...")

    # Determine which seed_node_side to read from parent hop
    parent_path = f"{task_data_dir}/path_exploration/hop_{hop_k - 1}"
    explore_hop = (max_hop + 1) // 2

    if hop_k == explore_hop and max_hop % 2 == 1:
        # Last hop with odd max_hop: only need forward paths from u_node (seed_node_side=0)
        # This is because odd max_hop means forward path is 1 hop longer than backward
        logger.info("  Reading parent paths from seed_node_side=0 only (last hop, odd max_hop)")
        parent_df = read_table(
            spark_ctx,
            parent_path,
            partition_columns=['seed_node_side'],
            partition_instances=[[0]]
        )
    else:
        # All other cases: read both directions (seed_node_side=0 and 1)
        # This enables bidirectional path exploration for bipath matching
        logger.info("  Reading parent paths from all seed_node_sides")
        parent_df = read_table(spark_ctx, parent_path)

    # Read outer join edges
    logger.info("  Reading outer join edges...")
    outer_edges_df = read_table(
        spark_ctx,
        f"{join_edge_path}/outer",
        partition_columns=graph.partition_columns,
        partition_instances=partition_instances
    )

    # Transform outer edges for joining with parent paths
    # Outer edge schema: join_node_id, add_node_id, join_node_type, add_node_type, edge_type, join_node_side
    # Need to map to: node_{hop_k-1}_id, node_{hop_k}_id, node_{hop_k-1}_type, node_{hop_k}_type,
    #                 edge_{hop_k-1}_type, u_index_of_edge_{hop_k-1}, v_index_of_edge_{hop_k-1}

    logger.info("  Transforming join edges...")
    join_edge_transformed = outer_edges_df.select(
        F.col('join_node_id').alias(f'node_{hop_k-1}_id'),
        F.col('add_node_id').alias(f'node_{hop_k}_id'),
        F.col('join_node_type').alias(f'node_{hop_k-1}_type'),
        F.col('add_node_type').alias(f'node_{hop_k}_type'),
        F.col('edge_type').alias(f'edge_{hop_k-1}_type'),
        # Calculate edge position based on join_node_side
        # If join_node_side=0 (u_node), then u_index=hop_k-1, v_index=hop_k
        # If join_node_side=1 (v_node), then u_index=hop_k, v_index=hop_k-1
        F.when(F.col('join_node_side') == 0, F.lit(hop_k-1))
         .otherwise(F.lit(hop_k))
         .alias(f'u_index_of_edge_{hop_k-1}'),
        F.when(F.col('join_node_side') == 1, F.lit(hop_k-1))
         .otherwise(F.lit(hop_k))
         .alias(f'v_index_of_edge_{hop_k-1}'),
        *[F.col(c) for c in graph.partition_columns]
    )

    # Join parent paths with transformed edges
    join_columns = [f'node_{hop_k-1}_id', f'node_{hop_k-1}_type'] + graph.partition_columns

    if hop_k == 1:
        # First hop: direct join (no deduplication or salting needed)
        # Seed nodes have no existing edges, so no duplicate edge checking required
        # Join edges are already sampled by max_neighbor parameter
        logger.info("  Joining seed nodes with first edges...")
        path_df = parent_df.join(
            join_edge_transformed,
            on=join_columns,
            how='inner'
        )

        # No sampling needed - join_edge already sampled by max_neighbor
        sampled_df = path_df

    else:
        # hop_k > 1: use salted join with duplicate edge prevention
        logger.info("  Using salted join with duplicate edge prevention...")

        # Define release point for persist lifecycle management
        release_point = f'hop_{hop_k}_done'

        # Add edge to path with deduplication
        path_df = add_path_to_path(
            spark_ctx=spark_ctx,
            left_path_df=parent_df,
            right_path_df=join_edge_transformed,
            join_columns=join_columns,
            left_hop_k=hop_k - 1,  # Number of existing edges in parent path
            right_hop_k=1,  # Adding single edge
            release_point=release_point,
            skew_threshold=1000,
            salt_buckets=100
        )

        # Sample paths to control path explosion
        logger.info("  Sampling paths to control path explosion...")
        group_columns = [
            'node_0_id', 'node_0_type', 'seed_node_side',  # Seed node identity
            *[f'edge_{i}_type' for i in range(hop_k)],  # Edge types
            *[f'{side}_index_of_edge_{i}' for i in range(hop_k) for side in ['u', 'v']]  # Edge positions
        ]
        sampled_df = random_sample(path_df, group_columns, max_instance)

    # Save results partitioned by seed_node_side
    logger.info("  Saving results...")
    write_table(
        spark_ctx,
        sampled_df,
        output_path,
        partition_columns=['seed_node_side'],
        mode='overwrite'
    )

    # Mark complete (direct marker under hop_k folder)
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Hop {hop_k} completed successfully")

    # Release persisted resources after all operations complete (only for hop_k > 1)
    if hop_k > 1:
        spark_ctx.persist_manager.mark_released(release_point)
        logger.info(f"  Released persisted resources for {release_point}")


# ==================== Main Function ====================

def find_relevant_bipaths(spark_ctx, bilink_config, graph_config):
    """
    Find relevant bi-paths for link discovery.

    This function identifies bi-path patterns connecting source nodes to potential
    target nodes based on the task configuration.

    Args:
        spark_ctx: SparkContext instance (passed by SparkRunner.run())
        bilink_config: BiLink configuration dictionary
        graph_config: Graph configuration dictionary
    """
    # Extract configuration
    task_data_dir = bilink_config['task_data_dir']
    sample_type_to_date_range = bilink_config['target_edge']['sample_type_to_date_range']

    # Check if already complete
    matched_bipath_path = f"{task_data_dir}/matched_bipaths"
    is_complete, _ = spark_ctx.table_state.check_complete(matched_bipath_path)
    if is_complete:
        logger.info(f"Matched bipaths table already exists at {matched_bipath_path}, skipping")
        return

    # Initialize Graph with graph_config
    logger.info("Initializing graph...")
    graph = Graph(graph_config, spark_ctx.fileio)
    logger.info("Graph initialized successfully")

    # Prepare target pairs from known_target_edges
    logger.info("\n" + "="*50)
    logger.info("Preparing target pairs from known_target_edges...")

    # Get max_target_pairs from config
    max_target_pairs = bilink_config['bipath_discovery']['exploration'].get('max_target_pairs', 200000)

    # Prepare target pairs
    prepare_target_pairs(
        spark_ctx=spark_ctx,
        task_data_dir=task_data_dir,
        max_target_pairs=max_target_pairs,
        bilink_config=bilink_config
    )

    # Derive target dates from train date range
    train_date_range = sample_type_to_date_range['train']
    year_dates = pd.date_range(
        start=train_date_range[0],
        end=train_date_range[1],
        freq='YS',  # Year Start = January 1st
        inclusive='left'  # Left-closed, right-open [start, end)
    )
    target_dates = [d.strftime('%Y-%m-%d') for d in year_dates]

    logger.info(f"Processing {len(target_dates)} target dates: {target_dates}")

    # Convert target_dates to partition_instances format
    partition_columns = graph.partition_columns  # Get from graph
    partition_instances = parse_partition_spec(
        partition_columns=partition_columns,
        partition_spec=target_dates  # Will auto-convert from 1D to 2D format
    )

    logger.info(f"Partition columns: {partition_columns}")
    logger.info(f"Partition instances: {partition_instances}")

    # Generate seed nodes
    logger.info("\n" + "="*50)
    logger.info("Generating seed nodes...")

    # Get seed node configuration
    neg_ratio = bilink_config['bipath_discovery']['exploration'].get('seed_nodes', {}).get('neg_ratio', 5)

    # Generate seed nodes
    generate_seed_nodes(
        spark_ctx=spark_ctx,
        task_data_dir=task_data_dir,
        graph=graph,
        bilink_config=bilink_config,
        neg_ratio=neg_ratio,
        partition_instances=partition_instances
    )

    # Extract join edge configuration
    join_edge_path = bilink_config['bipath_discovery']['join_edge_path']
    max_neighbor = bilink_config['bipath_discovery']['max_instance']

    logger.info(f"Join edge path: {join_edge_path}")
    logger.info(f"Max neighbor sampling: {max_neighbor}")

    # Use pre-generated join edges
    logger.info("Using pre-generated join edges (generated in data_prepare/build_graph_data.py)")

    # K-hop path exploration
    logger.info("\n" + "="*50)
    logger.info("Starting k-hop path exploration...")

    # Calculate exploration depth
    max_hop = bilink_config['bipath_discovery']['exploration']['max_hop']
    explore_hop = (max_hop + 1) // 2
    max_instance = bilink_config['bipath_discovery']['max_instance']

    logger.info(f"Max hop: {max_hop}, Explore hop: {explore_hop}, Max instance: {max_instance}")

    # Process each hop iteratively
    for hop_k in range(1, explore_hop + 1):
        logger.info(f"\n--- Processing hop {hop_k}/{explore_hop} ---")
        _process_single_hop(
            spark_ctx=spark_ctx,
            hop_k=hop_k,
            task_data_dir=task_data_dir,
            join_edge_path=join_edge_path,
            graph=graph,
            partition_instances=partition_instances,
            max_hop=max_hop,
            max_instance=max_instance
        )

    logger.info("\nK-hop path exploration completed successfully")

    # Compute and match bipaths for each hop
    logger.info("\n" + "="*50)
    logger.info(f"Computing and matching bipaths (1 to {max_hop})...")

    all_match_dfs = []
    for hop_k in range(1, max_hop + 1):
        logger.info(f"\n--- Processing {hop_k}-hop bipaths ---")

        # Compute k-hop bipaths
        compute_k_hop_bipaths(
            spark_ctx=spark_ctx,
            hop_k=hop_k,
            task_data_dir=task_data_dir,
            partition_columns=graph.partition_columns,
            max_instance=max_instance
        )

        # Match with target pairs and save to disk
        match_k_hop_bipaths(
            spark_ctx=spark_ctx,
            hop_k=hop_k,
            task_data_dir=task_data_dir,
            graph=graph,
            u_node_type=bilink_config['target_edge']['u_node_type'],
            v_node_type=bilink_config['target_edge']['v_node_type'],
            partition_columns=graph.partition_columns,
            partition_instances=partition_instances,
            max_instance=max_instance
        )

        # Read results from disk with schema
        schema = _get_matched_bipaths_schema(hop_k)
        match_path = f"{task_data_dir}/matched_bipaths/hop_{hop_k}"
        match_df = read_table(spark_ctx, match_path, schema=schema)

        all_match_dfs.append(match_df)

    # Union all DataFrames and save as table
    logger.info("\n" + "="*50)
    # Union with allowMissingColumns to handle different hop_k schemas
    combined_df = all_match_dfs[0]
    for df in all_match_dfs[1:]:
        combined_df = combined_df.unionByName(df, allowMissingColumns=True)

    # Add human-readable type names and reorder columns
    combined_df = add_readable_type_columns(spark_ctx, graph, combined_df, max_hop)

    # Sort path types
    combined_df = combined_df.orderBy(F.col('tgt_pair_percent').desc_nulls_last())

    # Persist before conversion (will be reused for both CSV and Parquet)
    spark_ctx.persist_manager.persist(
        combined_df,
        release_point='bipath_summary_done',
        name='combined_df_for_export'
    )

    # Export to CSV if configured
    if 'readable_matched_bipath' in bilink_config.get('bipath_discovery', {}):
        csv_path = bilink_config['bipath_discovery']['readable_matched_bipath']
        csv_path = f"{PROJECT_ROOT}/{csv_path}"

        logger.info(f"Exporting readable matched bipaths to CSV: {csv_path}")

        # Convert to pandas and save
        pandas_df = combined_df.toPandas()

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Save to CSV
        pandas_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(pandas_df)} bipath patterns to {csv_path}")

    # Save combined table to Parquet
    matched_bipath_path = f"{task_data_dir}/matched_bipaths/summary"
    write_table(
        spark_ctx,
        combined_df,
        matched_bipath_path,
        mode='overwrite'
    )

    # Release persist
    spark_ctx.persist_manager.mark_released('bipath_summary_done')

    spark_ctx.table_state.mark_complete(matched_bipath_path)
    logger.info(f"\nSaved matched bipaths table to {matched_bipath_path}")
    logger.info("Bipath discovery completed successfully!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Find relevant bi-paths for link discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find bi-paths using BiLink configuration
  python 0_relevant_bipaths_finder.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml

  # Force reprocess even if matched_bipath.json exists
  python 0_relevant_bipaths_finder.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --force
        """
    )

    # Required arguments
    parser.add_argument('--bilink-config', required=True,
                        help='Path to BiLink configuration YAML file')

    # Optional arguments
    parser.add_argument('--spark-mode', default='local', choices=['local', 'cluster'],
                        help='Spark execution mode (default: local)')
    parser.add_argument('--spark-platform', default='localhost', choices=['localhost', 'example'],
                        help='Spark platform (default: localhost)')
    parser.add_argument('--backends', nargs='+',
                        default=['local'],
                        choices=['local', 'hdfs', 's3'],
                        help='File backends to enable (default: local)')

    # Flags
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if already complete')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    task_name = "relevant_bipaths_finder"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{PROJECT_ROOT}/data/logs/{task_name}/{timestamp}.log"
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(log_file, level=log_level)

    # Initialize FileIO (must be first for reading configs)
    backend_configs = {backend: {} for backend in args.backends}
    fileio = FileIO(backend_configs)
    logger.info(f"FileIO initialized with schemes: {list(fileio.backends.keys())}")

    # Load bilink configuration
    bilink_config_path = f"file://{PROJECT_ROOT}/{args.bilink_config}"
    logger.info(f"Loading BiLink configuration from: {bilink_config_path}")
    bilink_config = fileio.read_yaml(bilink_config_path)
    logger.info("BiLink configuration loaded successfully")

    # Load graph configuration from bilink config
    graph_config_rel_path = bilink_config['graph_config']
    graph_config_path = f"file://{PROJECT_ROOT}/{graph_config_rel_path}"
    logger.info(f"Loading graph configuration from: {graph_config_path}")
    graph_config = fileio.read_yaml(graph_config_path)
    logger.info("Graph configuration loaded successfully")

    # Initialize SparkRunner (after FileIO)
    spark_runner = SparkRunner(
        mode=args.spark_mode,
        platform=args.spark_platform,
        fileio=fileio,
        config_dict={},
        ignore_complete=args.force
    )

    # Run main function
    spark_runner.run(
        find_relevant_bipaths,
        bilink_config,
        graph_config
    )

    logger.info("Script completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
