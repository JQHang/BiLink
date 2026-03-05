#!/usr/bin/env python3
"""
Prepare Sample Pairs and Construct Subgraphs

This script combines sample pair generation and subgraph construction (bipath
instance collection) into a single pipeline. For each processing mode, it first
generates the sample pairs, then collects bipath instances to construct subgraphs
formed by path instances for each candidate pair.

Processing Modes:
- train: Generate labeled training pairs with multiple negative ratios, then collect
         bipaths for training subgraphs
- eval:  Generate labeled validation and test pairs, then collect bipaths for
         evaluation subgraphs
- infer: Generate unlabeled inference pairs with batch assignment, then collect
         bipaths for inference subgraphs

Usage:
    # Train mode: generate training pairs and construct subgraphs
    python 2_sample_pairs_and_subgraph_construction.py \
        --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \
        --mode train \
        --neg-ratios 3 5 7 10

    # Eval mode: generate valid/test pairs and construct subgraphs
    python 2_sample_pairs_and_subgraph_construction.py \
        --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \
        --mode eval \
        --neg-ratio 10

    # Infer mode: generate unlabeled pairs and construct subgraphs
    python 2_sample_pairs_and_subgraph_construction.py \
        --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \
        --mode infer \
        --batch-ids 0,1,2,3
"""

import argparse
import sys
import logging
import pandas as pd
from datetime import datetime
from math import ceil
from functools import reduce

from joinminer import PROJECT_ROOT
from joinminer.spark import SparkRunner
from joinminer.spark.io import read_table, write_table
from joinminer.spark.io.partition import parse_partition_spec
from joinminer.fileio import FileIO
from joinminer.utils import setup_logger
from joinminer.graph import Graph
from joinminer.graph.join_edges.pair_bipaths import generate_bipaths_for_pairs
from joinminer.graph.join_edges.add_path import _get_path_type_columns
from joinminer.graph.join_edges.path_feat import prepare_path_features
from joinminer.graph.join_edges.bipath_collect import bipaths_collection
from joinminer.graph.join_edges.bipath_collect_feat import create_feat_vector_mapping
from joinminer.graph.join_edges.prepare_collect import prepare_collect
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

# Module-level logger
logger = logging.getLogger("joinminer")


# ==============================================================================
# Part 1: Sample Pairs - Helper Functions
# ==============================================================================

def derive_target_partitions(date_range):
    """
    Derive target partition instances from date range.

    Uses pandas.date_range to generate year start dates within the range.
    Date ranges are treated as left-closed, right-open intervals [start, end).

    Args:
        date_range: Two-element list [start_date, end_date]

    Returns:
        List of partition instances in format [[date1], [date2], ...]
    """
    year_dates = pd.date_range(
        start=date_range[0],
        end=date_range[1],
        freq='YS',
        inclusive='left'
    )
    return [[d.strftime('%Y-%m-%d')] for d in year_dates]


# ==============================================================================
# Part 2: Subgraph Construction - Helper Functions
# ==============================================================================

def parse_batch_ids(value):
    """Parse comma-separated batch IDs string into list of integers.

    Args:
        value: Comma-separated string like "0,1,2,3"

    Returns:
        List of integers [0, 1, 2, 3]
    """
    return [int(x.strip()) for x in value.split(',')]


def derive_target_dates(sample_type_to_date_range, mode):
    """
    Derive all unique January 1st dates from sample_type_to_date_range configuration.

    Uses pandas.date_range to generate year start dates within each date range.
    Date ranges are treated as left-closed, right-open intervals [start, end).

    Args:
        sample_type_to_date_range: Dictionary mapping sample types to [start_date, end_date]
        mode: Processing mode to filter sample types:
              - 'train': only use train dates
              - 'eval': only use valid + test dates
              - 'infer': only use infer dates

    Returns:
        List of date strings in format "YYYY-01-01", sorted chronologically
    """
    # Filter sample types based on mode
    if mode == 'train':
        filtered_ranges = {k: v for k, v in sample_type_to_date_range.items() if k == 'train'}
    elif mode == 'eval':
        filtered_ranges = {k: v for k, v in sample_type_to_date_range.items() if k in ['valid', 'test']}
    elif mode == 'infer':
        filtered_ranges = {k: v for k, v in sample_type_to_date_range.items() if k == 'infer'}

    all_dates = set()

    for date_range in filtered_ranges.values():
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
    Select top K target path types from matched bipaths summary for collection.

    Uses collection.top_k instead of selection.top_k to select more path types
    for comprehensive bipath feature collection.

    Args:
        spark_ctx: Spark context with fileio
        config: BiLink configuration dictionary

    Returns:
        DataFrame with selected target path types (coalesced to 1 partition)

    Raises:
        ValueError: If fewer than K paths are available
    """
    task_data_dir = config['task_data_dir']
    top_k = config['bipath_discovery']['collection']['top_k']

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

    logger.info(f"Selected top {top_k} target path types for collection")

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


# ==============================================================================
# Part 1: Sample Pairs Generation
# ==============================================================================

def generate_labeled_pairs(spark_ctx, config, graph, sample_type, negative_ratios):
    """
    Generate labeled training/validation/test sample pairs.

    Reads ground truth target pairs and candidate pairs from bipath exploration,
    matches them to identify positives, and samples negatives according to
    specified ratios. Generates separate datasets for each negative ratio.

    Args:
        spark_ctx: Spark context instance (passed by SparkRunner.run())
        config: BiLink configuration dictionary from YAML
        graph: Graph instance with partition column metadata
        sample_type: 'train', 'valid', or 'test'
        negative_ratios: List of negative sampling ratios [3, 5, 7, 10, 20]

    Processing:
        1. Derive target partitions from sample_type_to_date_range[sample_type]
        2. Read known_target_pairs (ground truth citations)
        3. Read candidate_pairs (from candidate pair selection)
        4. For each negative_ratio:
           - Match positives and sample up to max_positive
           - Sample negatives (pos_count * neg_ratio)
           - Save to sample_pairs/sample_type={sample_type}/neg_ratio={neg_ratio}

    Output:
        Multiple datasets (one per negative_ratio):
        - Path: {task_data_dir}/sample_pairs/sample_type={sample_type}/neg_ratio={neg_ratio}
        - Schema: u_node_id, v_node_id, label (0 or 1)
        - Partitioned by: sample_type, neg_ratio (Hive partition format)
    """
    logger.info("="*70)
    logger.info(f"Generating labeled pairs for sample_type='{sample_type}'")
    logger.info("="*70)

    # Auto-derive configuration from BiLink config
    task_data_dir = config['task_data_dir']

    # Check if already complete
    sample_pairs_path = f"{task_data_dir}/sample_pairs/sample_type={sample_type}"
    is_complete, _ = spark_ctx.table_state.check_complete(sample_pairs_path)

    if is_complete:
        logger.info(f"All samples already complete at '{sample_pairs_path}', skipping")
        return

    # Get configuration from target_edge
    target_edge_config = config['target_edge']
    sample_type_to_date_range = target_edge_config['sample_type_to_date_range']

    # Derive target partitions automatically
    logger.info(f"Deriving target partitions for sample_type='{sample_type}'...")
    date_range = sample_type_to_date_range[sample_type]
    target_partitions = derive_target_partitions(date_range)
    logger.info(f"Target partitions: {target_partitions}")

    # Step 1: Read known target pairs (ground truth)
    logger.info("\nStep 1: Reading known target pairs (ground truth)...")

    known_target_pairs_path = f"{task_data_dir}/known_target_edges"
    u_node_id_column = target_edge_config['u_node_id_column']
    v_node_id_column = target_edge_config['v_node_id_column']
    partition_columns = graph.partition_columns

    known_pairs_df = read_table(
        spark_ctx,
        known_target_pairs_path,
        partition_columns=["sample_type"],
        partition_instances=[[sample_type]]
    )

    # Apply column name mapping (rename to standard u_node_id, v_node_id)
    known_pairs_df = known_pairs_df.withColumnRenamed(u_node_id_column, 'u_node_id')
    known_pairs_df = known_pairs_df.withColumnRenamed(v_node_id_column, 'v_node_id')

    # Select only needed columns
    target_pairs_df = known_pairs_df.select('u_node_id', 'v_node_id', *partition_columns)

    # Step 2: Read candidate pairs
    logger.info("\nStep 2: Reading candidate pairs...")

    candidate_pairs_path = f"{task_data_dir}/candidate_pairs"

    candidate_pairs_df = read_table(
        spark_ctx,
        candidate_pairs_path,
        partition_columns=partition_columns,
        partition_instances=target_partitions
    )

    # Persist for multiple uses across different negative ratios
    spark_ctx.persist_manager.persist(
        candidate_pairs_df,
        release_point='labeled_generation_done',
        name='candidate_pairs_df'
    )

    # Step 3: Calculate positive samples (shared across all neg_ratios)
    logger.info("\nStep 3: Calculating positive samples (intersection of candidate and target)...")

    positive_pairs = candidate_pairs_df.join(
        target_pairs_df,
        on=['u_node_id', 'v_node_id', *partition_columns],
        how='inner'
    ).select('u_node_id', 'v_node_id', *partition_columns).distinct()

    # Persist for multiple uses
    spark_ctx.persist_manager.persist(
        positive_pairs,
        release_point='labeled_generation_done',
        name='positive_pairs_df'
    )

    positive_count = positive_pairs.count()
    logger.info(f"Total positive pairs found: {positive_count}")

    # Get sampling configuration
    selection_config = config['bipath_discovery']['selection']
    max_positive = selection_config['labeled_samples']['max_positive'][sample_type]

    # Sample positives if needed (same samples used for all neg_ratios)
    if positive_count > max_positive:
        logger.info(f"Sampling {max_positive} positive pairs from {positive_count}")
        sample_fraction = max_positive / positive_count
        positive_samples = positive_pairs.sample(withReplacement=False, fraction=sample_fraction)
        sampled_positive_count = max_positive
    else:
        logger.info(f"Keeping all {positive_count} positive pairs")
        positive_samples = positive_pairs
        sampled_positive_count = positive_count

    # Add label and persist for reuse
    positive_labeled = positive_samples.withColumn('label', F.lit(1))
    spark_ctx.persist_manager.persist(
        positive_labeled,
        release_point='labeled_generation_done',
        name='positive_labeled'
    )

    # Step 4: Calculate negative sample pool (shared across all neg_ratios)
    logger.info("\nStep 4: Calculating negative sample pool (candidates NOT in target pairs)...")

    negative_pool = candidate_pairs_df.join(
        target_pairs_df,
        on=['u_node_id', 'v_node_id', *partition_columns],
        how='left_anti'
    ).select('u_node_id', 'v_node_id', *partition_columns).distinct()

    # Persist negative pool for reuse
    spark_ctx.persist_manager.persist(
        negative_pool,
        release_point='labeled_generation_done',
        name='negative_pool'
    )

    negative_pool_count = negative_pool.count()
    logger.info(f"Total negative pairs available: {negative_pool_count}")

    # Step 5A: Sample negatives for all ratios and add neg_ratio column
    logger.info(f"\nStep 5: Preparing samples for {len(negative_ratios)} negative ratio(s)...")
    logger.info(f"Using {sampled_positive_count} positive samples for all ratios")

    all_negative_samples = []
    for neg_ratio in negative_ratios:
        target_negative_count = sampled_positive_count * neg_ratio

        if negative_pool_count > target_negative_count:
            logger.info(f"Sampling {target_negative_count} negatives for ratio={neg_ratio}")
            sample_fraction = target_negative_count / negative_pool_count
            neg_sample = negative_pool.sample(withReplacement=False, fraction=sample_fraction)
        else:
            logger.info(f"Using all {negative_pool_count} negatives for ratio={neg_ratio}")
            neg_sample = negative_pool

        # Add neg_ratio and label columns
        neg_sample = neg_sample.withColumn('neg_ratio', F.lit(neg_ratio)) \
                               .withColumn('label', F.lit(0))
        all_negative_samples.append(neg_sample)

    # Union all negative samples
    logger.info("Combining negative samples from all ratios...")
    all_negatives = reduce(lambda x, y: x.unionByName(y), all_negative_samples)

    # Step 5B: Replicate positive samples for all ratios
    logger.info(f"Replicating {sampled_positive_count} positive samples for {len(negative_ratios)} ratios")

    all_positive_samples = []
    for neg_ratio in negative_ratios:
        pos_sample = positive_labeled.withColumn('neg_ratio', F.lit(neg_ratio))
        all_positive_samples.append(pos_sample)

    # Union all positive samples
    all_positives = reduce(lambda x, y: x.unionByName(y), all_positive_samples)

    # Step 5C: Combine all samples and persist
    logger.info("Combining all positive and negative samples...")
    all_samples = all_positives.unionByName(all_negatives)

    spark_ctx.persist_manager.persist(
        all_samples,
        release_point='labeled_generation_done',
        name='all_samples'
    )

    # Step 5D: Calculate statistics via GroupBy
    logger.info("Calculating statistics for all ratios...")
    stats_df = all_samples.groupBy('neg_ratio', 'label').count()
    stats_collected = stats_df.collect()

    # Organize statistics by neg_ratio
    stats_dict = {}
    for row in stats_collected:
        ratio = row['neg_ratio']
        label = row['label']
        count = row['count']
        if ratio not in stats_dict:
            stats_dict[ratio] = {'positive': 0, 'negative': 0}
        if label == 1:
            stats_dict[ratio]['positive'] = count
        else:
            stats_dict[ratio]['negative'] = count

    # Log statistics for each neg_ratio
    logger.info("\nSample statistics:")
    for neg_ratio in sorted(negative_ratios):
        pos = stats_dict[neg_ratio]['positive']
        neg = stats_dict[neg_ratio]['negative']
        total = pos + neg
        ratio = neg / pos if pos > 0 else 0

        logger.info(f"\nneg_ratio={neg_ratio}:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Positive: {pos}")
        logger.info(f"  Negative: {neg}")
        logger.info(f"  Actual ratio: {ratio:.2f}")

    # Step 5E: Single write with neg_ratio partitioning
    labeled_output_path = f"{task_data_dir}/sample_pairs/sample_type={sample_type}"

    logger.info(f"\nSaving all samples to: {labeled_output_path}")
    logger.info(f"Partitioning by: neg_ratio")

    write_table(
        spark_ctx,
        all_samples,
        labeled_output_path,
        partition_columns=['neg_ratio'],
        mode='overwrite'
    )

    # Mark as complete at sample_type level (covers all neg_ratios)
    spark_ctx.table_state.mark_complete(labeled_output_path)
    logger.info(f"Completed all neg_ratios for sample_type={sample_type}")

    # Release persisted DataFrames
    spark_ctx.persist_manager.mark_released('labeled_generation_done')

    logger.info("\n" + "="*70)
    logger.info(f"Labeled pair generation for '{sample_type}' completed successfully!")
    logger.info("="*70)


def generate_unlabeled_pairs(spark_ctx, config, graph):
    """
    Generate unlabeled inference sample pairs with batch assignment.

    Reads candidate pairs from bipath exploration, assigns batch IDs based on
    configured batch sizes for distributed inference processing.

    Args:
        spark_ctx: Spark context instance (passed by SparkRunner.run())
        config: BiLink configuration dictionary from YAML
        graph: Graph instance with partition column metadata

    Processing:
        1. Derive target partitions from sample_type_to_date_range['infer']
        2. Read candidate_pairs (from candidate pair selection)
        3. Extract unique u_nodes and v_nodes
        4. Calculate batch assignments based on batch_size config
        5. Generate batch_id: u_batch_id * v_batch_count + v_batch_id
        6. Save with batch_id partitioning

    Batch ID Calculation:
        If u_node_count = 1.5M, u_batch_size = 1M:
            u_batch_count = 2

        If v_node_count = 3B, v_batch_size = 500M:
            v_batch_count = 6

        Then:
            batch_id = u_batch_id * v_batch_count + v_batch_id
            Range: 0 to (2*6-1) = 11 total batches

    Output:
        - Path: {task_data_dir}/sample_pairs/sample_type=infer
        - Schema: u_node_id, v_node_id, batch_id
        - Partitioned by: sample_type (Hive partition), batch_id (sub-partition)
    """
    logger.info("="*70)
    logger.info("Generating unlabeled pairs for inference")
    logger.info("="*70)

    # Auto-derive configuration from BiLink config
    task_data_dir = config['task_data_dir']
    unlabeled_output_path = f"{task_data_dir}/sample_pairs/sample_type=infer"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(unlabeled_output_path)

    if is_complete:
        logger.info(f"Already complete at '{unlabeled_output_path}', skipping")
        return

    # Derive target partitions automatically
    logger.info("Deriving target partitions for inference...")
    sample_type_to_date_range = config['target_edge']['sample_type_to_date_range']
    date_range = sample_type_to_date_range['infer']
    target_partitions = derive_target_partitions(date_range)
    logger.info(f"Target partitions: {target_partitions}")

    # Step 1: Read candidate pairs
    logger.info("\nStep 1: Reading candidate pairs...")

    candidate_pairs_path = f"{task_data_dir}/candidate_pairs"
    partition_columns = graph.partition_columns

    candidate_pairs_df = read_table(
        spark_ctx,
        candidate_pairs_path,
        partition_columns=partition_columns,
        partition_instances=target_partitions
    )

    # Persist for multiple uses
    spark_ctx.persist_manager.persist(
        candidate_pairs_df,
        release_point='unlabeled_generation_done',
        name='candidate_pairs_df'
    )

    candidate_count = candidate_pairs_df.count()
    logger.info(f"Total candidate pairs: {candidate_count}")

    # Step 2: Extract unique u_nodes
    logger.info("\nStep 2: Extracting unique u_nodes...")

    # Get u_node batch size configuration
    selection_config = config['bipath_discovery']['selection']
    u_batch_size = selection_config['unlabeled_samples']['batch_size']['u_node']

    u_nodes_df = candidate_pairs_df.select('u_node_id', *partition_columns).distinct()

    # Persist for batch calculation
    spark_ctx.persist_manager.persist(
        u_nodes_df,
        release_point='unlabeled_generation_done',
        name='u_nodes_df'
    )

    u_node_count = u_nodes_df.count()
    u_batch_count = ceil(u_node_count / u_batch_size)

    logger.info(f"  Total u_nodes: {u_node_count}")
    logger.info(f"  u_node batch size: {u_batch_size}")
    logger.info(f"  u_node batch count: {u_batch_count}")

    # Step 3: Assign u_node batch IDs
    logger.info("\nStep 3: Assigning u_node batch IDs...")
    u_nodes_with_batch = u_nodes_df.withColumn(
        'u_batch_id',
        F.floor(F.rand() * u_batch_count).cast('int')
    )

    # Step 4: Extract unique v_nodes
    logger.info("\nStep 4: Extracting unique v_nodes...")

    # Get v_node batch size configuration
    v_batch_size = selection_config['unlabeled_samples']['batch_size']['v_node']

    v_nodes_df = candidate_pairs_df.select('v_node_id', *partition_columns).distinct()

    # Persist for batch calculation
    spark_ctx.persist_manager.persist(
        v_nodes_df,
        release_point='unlabeled_generation_done',
        name='v_nodes_df'
    )

    v_node_count = v_nodes_df.count()
    v_batch_count = ceil(v_node_count / v_batch_size)

    logger.info(f"  Total v_nodes: {v_node_count}")
    logger.info(f"  v_node batch size: {v_batch_size}")
    logger.info(f"  v_node batch count: {v_batch_count}")

    # Step 5: Assign v_node batch IDs
    logger.info("\nStep 5: Assigning v_node batch IDs...")
    v_nodes_with_batch = v_nodes_df.withColumn(
        'v_batch_id',
        F.floor(F.rand() * v_batch_count).cast('int')
    )

    # Persist for joining
    spark_ctx.persist_manager.persist(
        v_nodes_with_batch,
        release_point='unlabeled_generation_done',
        name='v_nodes_with_batch'
    )

    # Step 6: Generate pair batch IDs
    logger.info("\nStep 6: Generating pair batch IDs...")

    # Join candidate_pairs with u_node batch IDs
    pairs_with_u_batch = candidate_pairs_df.join(
        u_nodes_with_batch,
        on=['u_node_id', *partition_columns],
        how='inner'
    )

    # Join with v_node batch IDs
    pairs_with_batches = pairs_with_u_batch.join(
        v_nodes_with_batch,
        on=['v_node_id', *partition_columns],
        how='inner'
    )

    # Calculate final batch_id
    # batch_id = u_batch_id * v_batch_count + v_batch_id
    unlabeled_pairs_df = pairs_with_batches.withColumn(
        'batch_id',
        (F.col('u_batch_id') * v_batch_count + F.col('v_batch_id')).cast('int')
    ).select('u_node_id', 'v_node_id', 'batch_id', *partition_columns)

    total_batch_count = u_batch_count * v_batch_count
    logger.info(f"  Total batch count: {total_batch_count}")

    # Step 7: Save unlabeled pairs with batch_id partitioning
    logger.info(f"\nStep 7: Saving unlabeled pairs to: {unlabeled_output_path}")
    write_table(
        spark_ctx,
        unlabeled_pairs_df,
        unlabeled_output_path,
        partition_columns=['batch_id'],
        mode='overwrite'
    )

    # Mark as complete (no specific partition instances for batch_id)
    spark_ctx.table_state.mark_complete(unlabeled_output_path)

    # Release all persisted DataFrames
    spark_ctx.persist_manager.mark_released('unlabeled_generation_done')

    logger.info("\n" + "="*70)
    logger.info("Unlabeled pair generation completed successfully!")
    logger.info("="*70)


# ==============================================================================
# Part 2: Subgraph Construction
# ==============================================================================

def collect_pair_bipaths(spark_ctx, bilink_config, graph, sample_config):
    """
    Collect bipath instances for sample pairs using top K reliable path types.

    Args:
        spark_ctx: SparkContext instance (passed by SparkRunner.run())
        bilink_config: BiLink configuration dictionary
        graph: Graph instance with partition column metadata
        sample_config: Sample configuration dictionary with:
            - pair_path: Path to sample pairs (e.g., 'hdfs:///sample_pairs/sample_type=train')
            - partition_columns: List of partition column names (e.g., ['neg_ratio'])
            - partition_instances: List of partition instances (e.g., [[3], [5], [7]])
            - bipath_dates: List of target dates for bipath collection (e.g., ['2021', '2022'])
            - output_path: Base output path for collected bipaths
    """
    # Check if bipaths collection is already complete
    bipaths_collection_path = f"{sample_config['output_path']}/bipaths_collection"
    is_complete, _ = spark_ctx.table_state.check_complete(bipaths_collection_path)
    if is_complete:
        logger.info(f"Bipaths collection already complete at {bipaths_collection_path}, skipping")
        return

    logger.info("Starting bipath collection for sample pairs")
    logger.info(f"  Pair path: {sample_config['pair_path']}")
    logger.info(f"  Partition columns: {sample_config['partition_columns']}")
    logger.info(f"  Partition instances: {sample_config['partition_instances']}")
    logger.info(f"  Bipath dates: {sample_config['bipath_dates']}")
    logger.info(f"  Output path: {sample_config['output_path']}")

    # Convert bipath_dates to partition format
    date_partitions = parse_partition_spec(
        partition_columns=graph.partition_columns,
        partition_spec=sample_config['bipath_dates']
    )

    logger.info(f"Processing {len(date_partitions)} date partition(s)")

    # Phase 1: Select target path types
    logger.info("\n" + "="*70)
    logger.info("Phase 1: Select target path types")
    logger.info("="*70)

    target_paths_df = select_target_paths(spark_ctx, bilink_config)

    # Persist target_paths_df - will be reused across all sample types
    spark_ctx.persist_manager.persist(
        target_paths_df,
        release_point='collect_pair_bipaths_done',
        name='target_paths_df'
    )

    # Calculate exploration parameters
    max_hop = target_paths_df.agg(F.max('hop_k')).collect()[0][0]
    explore_hop = (max_hop + 1) // 2

    logger.info(f"Max hop: {max_hop}, Explore hop: {explore_hop}")

    # Extract path node and edge type indices from selected paths
    path_node_type_indices = set()
    path_edge_type_indices = set()

    path_rows = target_paths_df.collect()
    for row in path_rows:
        # Collect node types (0 to max_hop)
        for i in range(max_hop + 1):
            node_type_col = f'node_{i}_type'
            if node_type_col in row and row[node_type_col] is not None:
                path_node_type_indices.add(row[node_type_col])

        # Collect edge types (0 to max_hop-1)
        for i in range(max_hop):
            edge_type_col = f'edge_{i}_type'
            if edge_type_col in row and row[edge_type_col] is not None:
                path_edge_type_indices.add(row[edge_type_col])

    # Convert to type names for logging
    path_node_types = {graph.node_index_to_type[idx] for idx in path_node_type_indices}
    path_edge_types = {graph.edge_index_to_type[idx] for idx in path_edge_type_indices}

    logger.info(f"Paths contain {len(path_node_types)} node types: {path_node_types}")
    logger.info(f"Paths contain {len(path_edge_types)} edge types: {path_edge_types}")

    # Phase 2: Read and persist sample pairs
    logger.info("\n" + "="*70)
    logger.info("Phase 2: Load sample pairs")
    logger.info("="*70)

    logger.info(f"Reading sample pairs from: {sample_config['pair_path']}")

    sample_pairs_df = read_table(
        spark_ctx,
        sample_config['pair_path'],
        partition_columns=sample_config['partition_columns'],
        partition_instances=sample_config['partition_instances']
    )

    spark_ctx.persist_manager.persist(
        sample_pairs_df,
        release_point='collect_pair_bipaths_done',
        name='sample_pairs_df'
    )

    pair_count = sample_pairs_df.count()
    logger.info(f"Loaded {pair_count} sample pairs (including all partitions and labels)")

    # Compute unique pairs (remove neg_ratio/batch_id/label dimensions)
    select_columns = ['u_node_id', 'v_node_id'] + graph.partition_columns
    unique_pairs_df = sample_pairs_df.select(*select_columns).distinct()

    spark_ctx.persist_manager.persist(
        unique_pairs_df,
        release_point='collect_pair_bipaths_done',
        name='unique_pairs_df'
    )

    unique_pair_count = unique_pairs_df.count()
    logger.info(f"Computed {unique_pair_count} unique pairs")

    # Generate bipaths for all partitions at once
    generate_bipaths_for_pairs(
        spark_ctx=spark_ctx,
        bilink_config=bilink_config,
        graph=graph,
        target_paths_df=target_paths_df,
        unique_pairs_df=unique_pairs_df,
        max_hop=max_hop,
        explore_hop=explore_hop,
        output_path=sample_config['output_path'],
        bipath_dates=sample_config['bipath_dates']
    )

    # Phase 3: Prepare path features
    logger.info("\n" + "="*70)
    logger.info("Phase 3: Prepare path features")
    logger.info("="*70)

    prepare_path_features(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=sample_config['output_path'],
        max_hop=max_hop,
        target_paths_df=target_paths_df,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices,
        scaler_base_path=sample_config['scaler_base_path'],
        load_scaler=sample_config['load_scaler']
    )

    # Phase 4: Prepare collection data (full processing by seed node dimension)
    logger.info("\n" + "="*70)
    logger.info("Phase 4: Prepare collection data")
    logger.info("="*70)

    prepare_collect(
        spark_ctx=spark_ctx,
        graph=graph,
        bilink_config=bilink_config,
        output_path=sample_config['output_path'],
        max_hop=max_hop,
        target_paths_df=target_paths_df,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )

    # Phase 5: Collect pair bipaths (batch processing)
    logger.info("\n" + "="*70)
    logger.info("Phase 5: Collect bipaths with feature IDs")
    logger.info("="*70)

    pair_batch = bilink_config['bipath_discovery']['collection']['pair_batch']
    num_batches = (unique_pair_count + pair_batch - 1) // pair_batch

    logger.info(f"Total pairs: {unique_pair_count}, batch size: {pair_batch}, num batches: {num_batches}")

    # Step 5.1: Assign batch_id and save
    batch_pairs_path = f"{sample_config['output_path']}/batch_pairs"
    is_complete, _ = spark_ctx.table_state.check_complete(batch_pairs_path)
    if not is_complete:
        logger.info(f"Saving batch pairs to {batch_pairs_path}")
        unique_pairs_with_batch = unique_pairs_df.withColumn(
            "batch_id",
            (F.rand() * num_batches).cast("int")
        )
        write_table(spark_ctx, unique_pairs_with_batch, batch_pairs_path,
                    partition_columns=['batch_id'])
        spark_ctx.table_state.mark_complete(batch_pairs_path)
    else:
        logger.info(f"Batch pairs already exist at {batch_pairs_path}")

    # Step 5.2: Read and persist prepared data
    logger.info("\nStep 5.2: Reading and persisting prepared data")
    logger.info("-"*70)

    base_path = sample_config['output_path']

    # Read seed_node_feat
    seed_node_feat_df = read_table(spark_ctx, f"{base_path}/seed_node_feat")
    seed_node_feat_df = spark_ctx.persist_manager.persist(
        seed_node_feat_df,
        release_point='collect_pair_bipaths_done',
        name='seed_node_feat_df'
    )
    logger.info(f"  Loaded seed_node_feat: {seed_node_feat_df.count()} records")

    # Read seed_node_elements
    seed_node_elements_df = read_table(spark_ctx, f"{base_path}/seed_node_elements")
    seed_node_elements_df = spark_ctx.persist_manager.persist(
        seed_node_elements_df,
        release_point='collect_pair_bipaths_done',
        name='seed_node_elements_df'
    )
    logger.info(f"  Loaded seed_node_elements: {seed_node_elements_df.count()} records")

    # Read seed_node_paths
    seed_node_paths_df = read_table(spark_ctx, f"{base_path}/seed_node_paths")
    seed_node_paths_df = spark_ctx.persist_manager.persist(
        seed_node_paths_df,
        release_point='collect_pair_bipaths_done',
        name='seed_node_paths_df'
    )
    logger.info(f"  Loaded seed_node_paths: {seed_node_paths_df.count()} records")

    # Read pair_bipaths
    pair_bipaths_df = read_table(spark_ctx, f"{base_path}/pair_bipaths")
    pair_bipaths_df = spark_ctx.persist_manager.persist(
        pair_bipaths_df,
        release_point='collect_pair_bipaths_done',
        name='pair_bipaths_df'
    )
    logger.info(f"  Loaded pair_bipaths: {pair_bipaths_df.count()} records")

    # Create element feat_vector mapping (static, reused across batches)
    element_feat_df = create_feat_vector_mapping(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=sample_config['output_path'],
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )
    element_feat_df = spark_ctx.persist_manager.persist(
        element_feat_df,
        release_point='collect_pair_bipaths_done',
        name='element_feat_df'
    )
    logger.info(f"  Created element_feat_df: {element_feat_df.count()} records")

    logger.info("  All prepared data loaded and persisted")

    # Step 5.3: Process each batch
    for batch_idx in range(num_batches):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")
        logger.info(f"{'='*70}")

        batch_pairs_df = read_table(spark_ctx, f"{batch_pairs_path}/batch_id={batch_idx}")
        batch_output_path = f"{bipaths_collection_path}/batch_id={batch_idx}"

        bipaths_collection(
            spark_ctx=spark_ctx,
            graph=graph,
            bilink_config=bilink_config,
            sample_config=sample_config,
            batch_pairs_df=batch_pairs_df,
            sample_pairs_df=sample_pairs_df,
            seed_node_feat_df=seed_node_feat_df,
            seed_node_elements_df=seed_node_elements_df,
            seed_node_paths_df=seed_node_paths_df,
            pair_bipaths_df=pair_bipaths_df,
            path_node_type_indices=path_node_type_indices,
            path_edge_type_indices=path_edge_type_indices,
            output_path=batch_output_path,
            element_feat_df=element_feat_df
        )

    # Mark bipaths collection as complete
    spark_ctx.table_state.mark_complete(bipaths_collection_path)

    # Release shared DataFrames
    spark_ctx.persist_manager.mark_released('collect_pair_bipaths_done')

    logger.info("\n" + "="*70)
    logger.info("Bipath collection completed successfully!")
    logger.info("="*70)


# ==============================================================================
# CLI and Main
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare sample pairs and construct subgraphs (bipath instances)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train mode: generate training pairs and construct subgraphs
  python 2_sample_pairs_and_subgraph_construction.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --mode train \\
      --neg-ratios 3 5 7 10

  # Eval mode: generate valid/test pairs and construct subgraphs
  python 2_sample_pairs_and_subgraph_construction.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --mode eval \\
      --neg-ratio 10

  # Infer mode: generate unlabeled pairs and construct subgraphs
  python 2_sample_pairs_and_subgraph_construction.py \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --mode infer \\
      --batch-ids 0,1,2,3
        """
    )

    # Required arguments
    parser.add_argument('--bilink-config', required=True,
                        help='Path to BiLink configuration YAML file')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'eval', 'infer'],
                        help='Processing mode: train (multiple neg_ratios), eval (valid+test with neg_ratio), infer (batch_ids)')

    # Mode-specific arguments
    parser.add_argument('--neg-ratios', type=int, nargs='+',
                        help='[train mode] List of negative ratios to process (e.g., 3 5 7 10)')
    parser.add_argument('--neg-ratio', type=int,
                        help='[eval mode] Single negative ratio to process for valid and test')
    parser.add_argument('--batch-ids', type=parse_batch_ids,
                        help='[infer mode] Comma-separated batch IDs (e.g., 0,1,2,3)')

    # Optional Spark configuration
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


def validate_args(args):
    """
    Validate mode-specific argument requirements and mutual exclusivity.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If arguments are invalid or inconsistent with mode
    """
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'train':
        # Train mode requires --neg-ratios
        if not args.neg_ratios:
            raise ValueError("Train mode requires --neg-ratios argument")
        if args.neg_ratio or args.batch_ids:
            raise ValueError("Train mode only accepts --neg-ratios (not --neg-ratio or --batch-ids)")
        logger.info(f"  Processing neg_ratios: {args.neg_ratios}")

    elif args.mode == 'eval':
        # Eval mode requires --neg-ratio
        if not args.neg_ratio:
            raise ValueError("Eval mode requires --neg-ratio argument")
        if args.neg_ratios or args.batch_ids:
            raise ValueError("Eval mode only accepts --neg-ratio (not --neg-ratios or --batch-ids)")
        logger.info(f"  Processing valid and test with neg_ratio: {args.neg_ratio}")

    elif args.mode == 'infer':
        # Infer mode requires --batch-ids
        if not args.batch_ids:
            raise ValueError("Infer mode requires --batch-ids argument")
        if args.neg_ratios or args.neg_ratio:
            raise ValueError("Infer mode only accepts --batch-ids (not --neg-ratios or --neg-ratio)")
        logger.info(f"  Processing batch_ids: {args.batch_ids}")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def main():
    """Main entry point - generates sample pairs and constructs subgraphs."""
    args = parse_args()

    # Setup logging
    task_name = "sample_pairs_and_subgraph_construction"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{PROJECT_ROOT}/data/logs/{task_name}/{timestamp}.log"
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(log_file, level=log_level)

    logger.info("="*70)
    logger.info("Sample Pairs Preparation and Subgraph Construction")
    logger.info("="*70)

    # Validate arguments based on mode
    validate_args(args)

    # Initialize FileIO (must be first for reading configs)
    backend_configs = {backend: {} for backend in args.backends}
    fileio = FileIO(backend_configs)
    logger.info(f"FileIO initialized with schemes: {list(fileio.backends.keys())}")

    # Load BiLink configuration
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

    # Initialize SparkRunner (after FileIO)
    spark_runner = SparkRunner(
        mode=args.spark_mode,
        platform=args.spark_platform,
        fileio=fileio,
        config_dict={},
        ignore_complete=args.force
    )

    # Extract configuration paths
    sample_type_to_date_range = bilink_config['target_edge']['sample_type_to_date_range']
    task_data_dir = bilink_config['task_data_dir']
    sample_pairs_base_path = f"{task_data_dir}/sample_pairs"
    collected_bipaths_base_path = f"{task_data_dir}/pair_bipaths"

    # ========================================================================
    # Execute mode-specific pipeline
    # ========================================================================

    if args.mode == 'train':
        # Phase 1: Generate labeled training pairs
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Generating labeled training pairs")
        logger.info("="*70)

        spark_runner.run(
            generate_labeled_pairs,
            bilink_config,
            graph,
            'train',
            args.neg_ratios
        )

        # Phase 2: Collect bipaths for training pairs
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Collecting bipaths for training pairs")
        logger.info("="*70)

        train_dates = derive_target_dates(sample_type_to_date_range, 'train')
        sample_config = {
            'pair_path': f'{sample_pairs_base_path}/sample_type=train',
            'partition_columns': ['neg_ratio'],
            'partition_instances': [[nr] for nr in args.neg_ratios],
            'bipath_dates': train_dates,
            'output_path': f'{collected_bipaths_base_path}/mode=train',
            'load_scaler': False,
            'scaler_base_path': f'{collected_bipaths_base_path}/mode=train/scalers'
        }
        spark_runner.run(
            collect_pair_bipaths,
            bilink_config,
            graph,
            sample_config
        )

    elif args.mode == 'eval':
        # Phase 1: Generate labeled valid and test pairs
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Generating labeled validation and test pairs")
        logger.info("="*70)

        for sample_type in ['valid', 'test']:
            logger.info(f"\n--- Processing sample_type='{sample_type}' with neg_ratio={args.neg_ratio} ---")
            spark_runner.run(
                generate_labeled_pairs,
                bilink_config,
                graph,
                sample_type,
                [args.neg_ratio]
            )

        # Phase 2: Collect bipaths for eval pairs
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Collecting bipaths for evaluation pairs")
        logger.info("="*70)

        eval_dates = derive_target_dates(sample_type_to_date_range, 'eval')
        sample_config = {
            'pair_path': f'{sample_pairs_base_path}',
            'partition_columns': ['sample_type', 'neg_ratio'],
            'partition_instances': [['valid', args.neg_ratio], ['test', args.neg_ratio]],
            'bipath_dates': eval_dates,
            'output_path': f'{collected_bipaths_base_path}/mode=eval',
            'load_scaler': True,
            'scaler_base_path': f'{collected_bipaths_base_path}/mode=train/scalers'
        }
        spark_runner.run(
            collect_pair_bipaths,
            bilink_config,
            graph,
            sample_config
        )

    elif args.mode == 'infer':
        # Phase 1: Generate unlabeled inference pairs
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Generating unlabeled inference pairs")
        logger.info("="*70)

        spark_runner.run(
            generate_unlabeled_pairs,
            bilink_config,
            graph
        )

        # Phase 2: Collect bipaths for each batch
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Collecting bipaths for inference batches")
        logger.info("="*70)

        infer_dates = derive_target_dates(sample_type_to_date_range, 'infer')
        for batch_id in args.batch_ids:
            logger.info(f"\n--- Processing batch_id={batch_id} ---")
            sample_config = {
                'pair_path': f'{sample_pairs_base_path}/sample_type=infer/batch_id={batch_id}',
                'partition_columns': None,
                'partition_instances': None,
                'bipath_dates': infer_dates,
                'output_path': f'{collected_bipaths_base_path}/mode=infer/batch_id={batch_id}',
                'load_scaler': True,
                'scaler_base_path': f'{collected_bipaths_base_path}/mode=train/scalers'
            }
            spark_runner.run(
                collect_pair_bipaths,
                bilink_config,
                graph,
                sample_config
            )

    # ========================================================================
    # Done
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("Script completed successfully!")
    logger.info("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
