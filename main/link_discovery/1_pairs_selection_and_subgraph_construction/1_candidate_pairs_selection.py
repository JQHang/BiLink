#!/usr/bin/env python3
"""
Select Reliable Path Types for Candidate Pair Generation

This script selects the most reliable bi-path types from the matched bipaths
summary and generates path type specifications for each exploration hop.

Features:
- Select top K most reliable path types based on target pair percentage
- Generate forward (u_node) and backward (v_node) path specifications
- Support for bidirectional path exploration at each hop level

Usage:
    python 1_candidate_pairs_selection.py \
        --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \
        [--force] \
        [--verbose]
"""

import argparse
import sys
import logging
import pandas as pd
from datetime import datetime
from functools import reduce

from joinminer import PROJECT_ROOT
from joinminer.spark import SparkRunner
from joinminer.spark.io import read_table, write_table
from joinminer.spark.io.partition import parse_partition_spec
from joinminer.spark.operations.sample import random_sample
from joinminer.fileio import FileIO
from joinminer.utils import setup_logger
from joinminer.graph import Graph
from joinminer.graph.join_edges.bipath_types import extract_u_explore_paths, extract_v_explore_paths
from joinminer.graph.join_edges.add_path import add_path_to_path, add_path_types_to_path, _get_path_type_columns
from joinminer.graph.join_edges.bipath_pair import generate_bipath_pairs_for_hop
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

# Module-level logger
logger = logging.getLogger("joinminer")


# ==================== Helper Functions ====================

def derive_target_dates(sample_type_to_date_range):
    """
    Derive all unique January 1st dates from sample_type_to_date_range configuration.

    Uses pandas.date_range to generate year start dates within each date range.
    Date ranges are treated as left-closed, right-open intervals [start, end).

    Example:
        sample_type_to_date_range = {
            "train": ["2021-01-01", "2023-01-01"],
            "valid": ["2023-01-01", "2023-07-01"],
            "test": ["2023-07-01", "2024-01-01"],
            "infer": ["2024-01-01", "2025-01-01"]
        }

        Returns: ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
        (Note: 2025-01-01 is not included due to left-closed, right-open interval)

    Args:
        sample_type_to_date_range: Dictionary mapping sample types to [start_date, end_date]

    Returns:
        List of date strings in format "YYYY-01-01", sorted chronologically
    """
    all_dates = set()

    for date_range in sample_type_to_date_range.values():
        # Generate all year starts in range [start, end) using pandas
        year_dates = pd.date_range(
            start=date_range[0],
            end=date_range[1],
            freq='YS',        # Year Start = January 1st of each year
            inclusive='left'  # Left-closed, right-open interval [start, end)
        )

        # Convert to string format and add to set
        all_dates.update(d.strftime('%Y-%m-%d') for d in year_dates)

    return sorted(list(all_dates))


def select_target_paths(spark_ctx, config) -> DataFrame:
    """
    Select top K target path types from matched bipaths summary.

    Args:
        spark_ctx: Spark context with fileio
        config: BiLink configuration dictionary

    Returns:
        DataFrame with selected target path types (coalesced to 1 partition)

    Raises:
        ValueError: If fewer than K paths are available
    """
    task_data_dir = config['task_data_dir']
    top_k = config['bipath_discovery']['selection']['top_k']

    # Read matched bipaths summary
    summary_path = f"{task_data_dir}/matched_bipaths/summary"
    logger.info(f"Reading matched bipaths from: {summary_path}")

    summary_df = read_table(spark_ctx, summary_path)

    # Validate that we have at least K paths
    total_paths = summary_df.count()
    if total_paths < top_k:
        raise ValueError(
            f"Insufficient path types: only {total_paths} available, but top_k={top_k} requested. "
            f"Please reduce top_k or run bipath discovery with more data."
        )

    logger.info(f"Found {total_paths} total path types")

    # Sort by tgt_pair_percent and select top K
    # Use coalesce(1) since path types are small
    target_paths_df = summary_df.orderBy(
        F.col('tgt_pair_percent').desc_nulls_last()
    ).limit(top_k).coalesce(1)

    logger.info(f"Selected top {top_k} target path types")

    # Show selected paths for verification
    logger.info("Selected target paths:")
    target_paths_df.show(truncate=False)

    # Get max_hop to determine which columns to keep
    max_hop = target_paths_df.agg(F.max('hop_k')).collect()[0][0]

    # Get numeric path type columns using helper function
    path_type_columns = _get_path_type_columns(max_hop)

    # Keep only hop_k and numeric path type columns (drop *_type_name and statistics columns)
    target_paths_df = target_paths_df.select('hop_k', *path_type_columns)

    return target_paths_df


def _explore_paths_for_hop(
    spark_ctx,
    hop_k: int,
    explore_path_df: DataFrame,
    outer_edges_df: DataFrame,
    bilink_config: dict,
    graph,
    hop_missing_partitions: list,
    task_data_dir: str,
    max_hop: int,
    explore_hop: int
) -> None:
    """
    Explore paths for a single hop by joining with outer edges.

    This function handles path exploration differently based on hop number:
    - Hop 1: Reads seed nodes from config and performs direct join
    - Hop 2+: Reads parent paths, filters by supported path types, and uses add_path_to_path

    Args:
        spark_ctx: Spark context instance
        hop_k: Current hop number (1, 2, 3, ...)
        explore_path_df: DataFrame containing path types to explore
        outer_edges_df: Pre-persisted outer join edges
        bilink_config: BiLink configuration dictionary
        graph: Graph instance with schema information
        hop_missing_partitions: Partition instances to process for this hop
        task_data_dir: Base directory for task data
        max_hop: Maximum hop number from target paths
        explore_hop: Exploration hop depth (middle point of bipaths)
    """
    hop_output_path = f"{task_data_dir}/candidate_path_exploration/hop_{hop_k}"
    logger.info(f"\n--- Exploring paths for hop {hop_k} ---")

    # Step 1: Get parent paths or seed nodes based on hop
    if hop_k == 1:
        # Read seed nodes from configuration
        logger.info("  Reading seed nodes from configuration...")
        seed_config = bilink_config['bipath_discovery']['exploration']['seed_nodes']

        # Get node types from target_edge config
        u_node_type = bilink_config['target_edge']['u_node_type']
        v_node_type = bilink_config['target_edge']['v_node_type']

        # Convert node types to indices
        u_node_type_index = graph.nodes[u_node_type]['type_index']
        v_node_type_index = graph.nodes[v_node_type]['type_index']

        all_seed_nodes = []
        for seed_node_side in ['u_node', 'v_node']:
            node_config = seed_config[seed_node_side]

            # Read source data
            source_df = read_table(
                spark_ctx,
                node_config['path'],
                partition_columns=graph.partition_columns,
                partition_instances=hop_missing_partitions
            )

            # Rename ID column and add metadata
            seed_node_side_value = 0 if seed_node_side == 'u_node' else 1
            node_type_index = u_node_type_index if seed_node_side == 'u_node' else v_node_type_index

            seed_nodes = source_df.select(
                F.col(node_config['id_column']).alias('node_0_id'),
                F.lit(node_type_index).alias('node_0_type'),
                F.lit(seed_node_side_value).alias('seed_node_side'),
                *[F.col(c) for c in graph.partition_columns]
            )

            all_seed_nodes.append(seed_nodes)

        # Combine seed nodes from both sides
        parent_df = all_seed_nodes[0].union(all_seed_nodes[1])
        logger.info("  Seed nodes loaded from configuration")

    elif hop_k == explore_hop and max_hop % 2 == 1:
        # Special case: last hop with odd max_hop, only read forward paths (seed_node_side=0)
        parent_path = f"{task_data_dir}/candidate_path_exploration/hop_{hop_k-1}"
        logger.info(f"  Reading parent paths from: {parent_path}")
        logger.info("  Reading only forward paths (last hop with odd max_hop)")

        parent_df = read_table(
            spark_ctx,
            parent_path,
            partition_columns=graph.partition_columns + ['seed_node_side'],
            partition_instances=[p + [0] for p in hop_missing_partitions]
        )

    else:
        # Default case: hop 2+ with normal bidirectional exploration
        parent_path = f"{task_data_dir}/candidate_path_exploration/hop_{hop_k-1}"
        logger.info(f"  Reading parent paths from: {parent_path}")

        parent_df = read_table(
            spark_ctx,
            parent_path,
            partition_columns=graph.partition_columns,
            partition_instances=hop_missing_partitions
        )

    # Add path type specifications (applies to both seed nodes and parent paths)
    parent_df = add_path_types_to_path(
        path_df=parent_df,
        path_type_df=explore_path_df,
        path_length=hop_k - 1,  # Actual path length
        additional_join_columns=['seed_node_side']
    )
    logger.info("  Path types added and parent paths filtered")

    # Step 2: Transform outer edges for joining
    logger.info("  Transforming outer edges for joining...")

    # Outer edge schema: join_node_id, add_node_id, join_node_type, add_node_type, edge_type, join_node_side
    # Transform to match path schema for hop_k
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

    # Step 3: Join parent paths with edges based on hop
    # Join columns include node ID, type, and the enriched edge specifications
    join_columns = [
        f'node_{hop_k-1}_id',
        f'node_{hop_k-1}_type',
        f'edge_{hop_k-1}_type',  # Added by filter_and_enrich
        f'u_index_of_edge_{hop_k-1}',  # Added by filter_and_enrich
        f'v_index_of_edge_{hop_k-1}',   # Added by filter_and_enrich
        f'node_{hop_k}_type',
    ] + graph.partition_columns

    if hop_k == 1:
        # Direct join for first hop (no deduplication needed for seed nodes)
        logger.info("  Joining seed nodes with first edges (direct join)...")
        path_df = parent_df.join(
            join_edge_transformed,
            on=join_columns,
            how='inner'
        )

        # No sampling needed - join edges already sampled
        result_df = path_df

        logger.info(f"  Paths after join: {result_df.count()}")

    else:
        # Use add_path_to_path for deduplication (hop > 1)
        logger.info("  Using add_path_to_path with duplicate edge prevention...")

        release_point = f'hop_{hop_k}_done'

        path_df = add_path_to_path(
            spark_ctx=spark_ctx,
            left_path_df=parent_df,
            right_path_df=join_edge_transformed,
            join_columns=join_columns,
            left_hop_k=hop_k - 1,
            right_hop_k=1,
            release_point=release_point,
            skew_threshold=1000,
            salt_buckets=100
        )

        # Sample paths to control explosion
        logger.info("  Sampling paths to control explosion...")
        max_instance = bilink_config['bipath_discovery']['max_instance']

        group_columns = [
            'node_0_id', 'node_0_type', 'seed_node_side',  # Seed node identity
            *[f'edge_{i}_type' for i in range(hop_k)],  # Edge types
            *[f'u_index_of_edge_{i}' for i in range(hop_k)],  # U indices
            *[f'v_index_of_edge_{i}' for i in range(hop_k)]   # V indices
        ]

        result_df = random_sample(path_df, group_columns, max_instance)

    # Step 4: Write results
    logger.info(f"  Writing results to: {hop_output_path}")
    write_table(
        spark_ctx,
        result_df,
        hop_output_path,
        partition_columns=graph.partition_columns + ['seed_node_side'],
        mode='overwrite'
    )

    # Mark this hop as complete for the processed partitions
    spark_ctx.table_state.mark_complete(
        hop_output_path,
        partition_columns=graph.partition_columns,
        partition_instances=hop_missing_partitions
    )

    logger.info(f"  Hop {hop_k} exploration completed successfully")


# ==================== Main Function ====================

def select_candidate_pairs(spark_ctx, bilink_config, graph, target_dates):
    """
    Select target path types and generate path specifications for each hop.

    Args:
        spark_ctx: SparkContext instance (passed by SparkRunner.run())
        bilink_config: BiLink configuration dictionary
        graph: Graph instance with partition column metadata
        target_dates: List of target dates for partitioning
    """
    # Convert target_dates to partition format for completion check
    date_partitions = parse_partition_spec(
        partition_columns=graph.partition_columns,
        partition_spec=target_dates  # Will auto-convert from 1D to 2D format
    )

    # Extract task_data_dir for path construction
    task_data_dir = bilink_config['task_data_dir']

    # Check if overall candidate pairs selection is already complete
    candidate_pairs_path = f"{task_data_dir}/candidate_pairs"
    is_complete, missing_partitions = spark_ctx.table_state.check_complete(
        candidate_pairs_path,
        partition_columns=graph.partition_columns,
        partition_instances=date_partitions
    )

    if is_complete:
        logger.info(f"Candidate pairs already complete at {candidate_pairs_path}, skipping")
        return

    # Use missing_partitions for all downstream operations
    logger.info(f"Processing {len(missing_partitions)} missing partition(s): {missing_partitions}")

    # Read and persist outer join edges (used across all hop explorations)
    join_edge_path = bilink_config['bipath_discovery']['join_edge_path']
    logger.info(f"Reading outer join edges from: {join_edge_path}/outer")

    outer_edges_df = read_table(
        spark_ctx,
        f"{join_edge_path}/outer",
        partition_columns=graph.partition_columns,
        partition_instances=missing_partitions
    )

    # Persist for reuse across all hops
    spark_ctx.persist_manager.persist(
        outer_edges_df,
        release_point='explore_paths_done',
        name='outer_edges_df'
    )

    logger.info(f"Outer join edges loaded and persisted")

    logger.info("="*70)
    logger.info("Step 1: Select target path types")
    logger.info("="*70)

    # Select top K target paths (returns DataFrame, raises error if insufficient)
    target_paths_df = select_target_paths(spark_ctx, bilink_config)

    # Persist target_paths_df - will be reused across all hops
    spark_ctx.persist_manager.persist(
        target_paths_df,
        release_point='bipath_pairs_done',
        name='target_paths_df'
    )

    # Calculate exploration parameters from actual data
    max_hop = target_paths_df.agg(F.max('hop_k')).collect()[0][0]
    explore_hop = (max_hop + 1) // 2

    total_paths = target_paths_df.count()
    logger.info(f"\nPath statistics:")
    logger.info(f"  Total selected paths: {total_paths}")
    logger.info(f"  Max hop (from data): {max_hop}")
    logger.info(f"  Explore hop: {explore_hop}")

    # Generate path type DataFrames for each hop
    logger.info("\n" + "="*70)
    logger.info("Step 2: Generate path type specifications for each hop")
    logger.info("="*70)

    for hop_k in range(1, explore_hop + 1):
        logger.info(f"\n--- Processing hop {hop_k}/{explore_hop} ---")

        # Check if this hop's candidate path exploration is already complete
        hop_output_path = f"{task_data_dir}/candidate_path_exploration/hop_{hop_k}"
        hop_complete, hop_missing_partitions = spark_ctx.table_state.check_complete(
            hop_output_path,
            partition_columns=graph.partition_columns,
            partition_instances=missing_partitions  # Use cascaded missing_partitions
        )

        if hop_complete:
            logger.info(f"Hop {hop_k} already complete for all missing partitions, skipping")
            continue

        logger.info(f"Processing hop {hop_k} for {len(hop_missing_partitions)} partition(s): {hop_missing_partitions}")

        # Extract forward paths (u_node) - always returns DataFrame
        u_paths_df = extract_u_explore_paths(target_paths_df, hop_k)

        # Persist u_paths_df - used multiple times in this hop
        spark_ctx.persist_manager.persist(
            u_paths_df,
            release_point=f'hop_{hop_k}_done',
            name=f'u_paths_df_hop_{hop_k}'
        )

        u_paths_count = u_paths_df.count()
        logger.info(f"  Forward (u_node) paths: {u_paths_count}")
        if u_paths_count > 0:
            logger.info("  Sample u_node paths:")
            u_paths_df.show(3, truncate=False)

        # Extract backward paths (v_node) if condition met - always returns DataFrame
        if 2 * hop_k <= max_hop:
            v_paths_df = extract_v_explore_paths(target_paths_df, hop_k, max_hop)

            # Persist v_paths_df - used multiple times in this hop
            spark_ctx.persist_manager.persist(
                v_paths_df,
                release_point=f'hop_{hop_k}_done',
                name=f'v_paths_df_hop_{hop_k}'
            )

            v_paths_count = v_paths_df.count()
            logger.info(f"  Backward (v_node) paths: {v_paths_count}")
            if v_paths_count > 0:
                logger.info("  Sample v_node paths:")
                v_paths_df.show(3, truncate=False)

            # Simple union - both DataFrames have node_side column
            explore_path_df = u_paths_df.union(v_paths_df).coalesce(1)

            # Persist explore_path_df - used multiple times in this hop
            spark_ctx.persist_manager.persist(
                explore_path_df,
                release_point=f'hop_{hop_k}_done',
                name=f'explore_paths_df_hop_{hop_k}'
            )
        else:
            logger.info(f"  Backward (v_node) paths: 0")
            explore_path_df = u_paths_df

        # Log total for this hop
        total_count = explore_path_df.count()
        logger.info(f"  Total path types for hop {hop_k}: {total_count}")

        # Explore paths for this hop - join with outer edges and generate candidate paths
        _explore_paths_for_hop(
            spark_ctx=spark_ctx,
            hop_k=hop_k,
            explore_path_df=explore_path_df,
            outer_edges_df=outer_edges_df,
            bilink_config=bilink_config,
            graph=graph,
            hop_missing_partitions=hop_missing_partitions,
            task_data_dir=task_data_dir,
            max_hop=max_hop,
            explore_hop=explore_hop
        )

        # Release per-hop DataFrames (u_paths_df, v_paths_df)
        spark_ctx.persist_manager.mark_released(f'hop_{hop_k}_done')

    # Release outer join edges after all hops processed
    spark_ctx.persist_manager.mark_released('explore_paths_done')

    # Read existing edges to filter out (from prepared exist_target_edges)
    existing_edges_path = f"{task_data_dir}/exist_target_edges"
    logger.info(f"Reading existing edges from: {existing_edges_path}")

    existing_edges_df = read_table(
        spark_ctx,
        existing_edges_path,
        partition_columns=graph.partition_columns,
        partition_instances=missing_partitions
    )

    # Persist for filtering operations
    spark_ctx.persist_manager.persist(
        existing_edges_df,
        release_point='bipath_pairs_done',
        name='existing_edges_df'
    )

    # Step 3: Generate bipath pairs for each hop_k
    logger.info("\n" + "="*70)
    logger.info("Step 3: Generate bipath pairs from target paths")
    logger.info("="*70)

    pair_df_list = []
    for hop_k in range(1, max_hop + 1):
        logger.info(f"\n--- Processing bipath pairs for hop {hop_k}/{max_hop} ---")

        # Filter target paths for this specific hop
        hop_paths_df = target_paths_df.filter(F.col('hop_k') == hop_k)

        # Persist for this hop processing
        spark_ctx.persist_manager.persist(
            hop_paths_df,
            release_point=f'bipath_hop_{hop_k}_done',
            name=f'hop_paths_df_{hop_k}'
        )

        # Check if there are any paths for this hop
        if hop_paths_df.count() == 0:
            logger.info(f"  No paths found for hop {hop_k}, skipping...")
            spark_ctx.persist_manager.mark_released(f'bipath_hop_{hop_k}_done')
            continue

        # Generate bipath pairs for this hop_k
        generate_bipath_pairs_for_hop(
            spark_ctx=spark_ctx,
            hop_k=hop_k,
            hop_paths_df=hop_paths_df,
            task_data_dir=task_data_dir,
            graph=graph,
            existing_edges_df=existing_edges_df,
            partition_instances=missing_partitions,
            bilink_config=bilink_config
        )

        # Release after processing this hop
        spark_ctx.persist_manager.mark_released(f'bipath_hop_{hop_k}_done')
        logger.info(f"  Bipath pair generation for hop {hop_k} completed")

        # Read generated pairs and add to list
        pairs_path = f"{task_data_dir}/bipath_pairs/hop_{hop_k}"
        pairs_df = read_table(
            spark_ctx,
            pairs_path,
            partition_columns=graph.partition_columns,
            partition_instances=missing_partitions
        )
        pair_df_list.append(pairs_df)

    # Release target_paths_df after all hops are processed
    spark_ctx.persist_manager.mark_released('bipath_pairs_done')

    # Step 4: Union all distinct bipath pairs and save
    logger.info("\n" + "="*70)
    logger.info("Step 4: Combining and saving all candidate pairs")
    logger.info("="*70)

    all_pairs_df = reduce(lambda x, y: x.unionByName(y), pair_df_list).distinct()

    # Save combined pairs to candidate_pairs
    logger.info(f"Saving candidate pairs to: {candidate_pairs_path}")
    write_table(
        spark_ctx,
        all_pairs_df,
        candidate_pairs_path,
        partition_columns=graph.partition_columns,
        mode='overwrite'
    )

    # Mark completion
    spark_ctx.table_state.mark_complete(
        candidate_pairs_path,
        partition_columns=graph.partition_columns,
        partition_instances=missing_partitions
    )
    logger.info(f"Candidate pairs saved successfully")

    logger.info("\n" + "="*70)
    logger.info("Candidate pair selection completed successfully!")
    logger.info("="*70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Select reliable path types for candidate pair generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select reliable paths using BiLink configuration
  python 1_candidate_pairs_selection.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml

  # Force reprocess
  python 1_candidate_pairs_selection.py \\
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
    task_name = "candidate_pairs_selection"
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

    # Initialize Graph
    logger.info("Initializing graph...")
    graph = Graph(graph_config, fileio)
    logger.info("Graph initialized successfully")

    # Derive target_dates from sample_type_to_date_range
    sample_type_to_date_range = bilink_config['target_edge']['sample_type_to_date_range']
    target_dates = derive_target_dates(sample_type_to_date_range)

    logger.info(f"Derived target dates: {target_dates}")

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
        select_candidate_pairs,
        bilink_config,
        graph,
        target_dates
    )

    logger.info("Script completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
