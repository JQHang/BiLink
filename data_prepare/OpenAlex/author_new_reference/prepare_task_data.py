#!/usr/bin/env python3
"""
Prepare Task Data for Author-New-Reference Link Discovery

This script generates the necessary tables for the author-new-reference prediction task.
These tables are used throughout the entire pipeline: training, validation, testing,
inference, and bipath exploration.

Tables Generated:
1. known_target_edges - Author-paper citation pairs with train/valid/test/infer splits
2. exist_target_edges - Authors' existing citations (for filtering already-known relationships)
3. candidate_authors - Active authors (published in past 5 years)

Usage:
    python 1_prepare_task_data.py \\
        --bilink-config main/config/bilink/author_citation.yaml \\
        --context-table-dir file:///path/to/context_table \\
        [--force] \\
        [--verbose]
"""

import argparse
import sys
import logging
from datetime import datetime

import pandas as pd
from pyspark.sql.functions import col, when, trunc, to_date, broadcast, year

from joinminer import PROJECT_ROOT
from joinminer.spark import SparkRunner
from joinminer.spark.io import read_table, write_table
from joinminer.spark.operations import ordered_sample
from joinminer.fileio import FileIO
from joinminer.utils import setup_logger

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


# ==================== Shared Computation Functions ====================

def compute_author_first_citation(spark_ctx, works_author_path, works_citation_path,
                                  candidate_authors_path):
    """
    Compute first citation date for each author-paper pair and load candidate authors.

    This is a shared computation used by both known_target_edges and exist_target_edges
    tables. The results are cached using persist_manager for reuse.

    Args:
        spark_ctx: SparkContext from SparkRunner
        works_author_path: Path to works_author context table
        works_citation_path: Path to works_citation context table
        candidate_authors_path: Path to pre-computed candidate authors table

    Returns:
        Tuple of (author_first_cite_df, author_work_df, candidate_authors_df)
        all cached for downstream use
    """
    logger.info("Computing shared author first citation data...")

    # Load author-work relationships and cache
    author_work_df = read_table(spark_ctx, works_author_path, format='parquet')
    author_work_df = author_work_df.select("author_id", "work_id", "date")
    spark_ctx.persist_manager.persist(
        author_work_df,
        release_point='target_edges_done',
        name='author_work'
    )

    # Load citation relationships
    cite_work_df = read_table(spark_ctx, works_citation_path, format='parquet')
    cite_work_df = cite_work_df.select("work_id", "cited_work_id", "date")

    # Join to get author-citation pairs
    author_cite_work_df = author_work_df.join(
        cite_work_df,
        on=["work_id", "date"],
        how="inner"
    )
    author_cite_work_df = author_cite_work_df.select(
        "author_id", "cited_work_id", "date"
    ).distinct()

    # Keep only first citation per author-paper pair
    author_first_cite_df = ordered_sample(
        author_cite_work_df,
        group_columns=["author_id", "cited_work_id"],
        order_config={"date": "asc"},
        n=1
    )

    # Cache for downstream use
    spark_ctx.persist_manager.persist(
        author_first_cite_df,
        release_point='target_edges_done',
        name='author_first_citation'
    )

    # Load and cache candidate authors (used by both target edge functions)
    candidate_authors_df = read_table(spark_ctx, candidate_authors_path, format='parquet')
    spark_ctx.persist_manager.persist(
        candidate_authors_df,
        release_point='target_edges_done',
        name='candidate_authors'
    )

    logger.info("✓ Shared author first citation data and candidate authors cached")

    return author_first_cite_df, author_work_df, candidate_authors_df


# ==================== Table Generation Functions ====================

def generate_known_target_edges(spark_ctx, author_first_cite_df, author_work_df,
                               candidate_authors_df, known_target_edges_path,
                               sample_type_to_date_range):
    """
    Generate known_target_edges table.

    This table contains author-paper citation pairs with train/valid/test/infer splits.
    Each row represents the first time an author cited a specific paper.

    Processing steps:
    1. Use pre-computed author_first_cite_df (passed as parameter)
    2. Remove self-citations (author's own papers)
    3. Add sample_type labels based on date ranges
    4. Filter to only active authors using pre-computed candidate_authors DataFrame
    5. Normalize dates to Jan 1st of each year

    Args:
        spark_ctx: SparkContext from SparkRunner
        author_first_cite_df: Pre-computed first citation DataFrame (cached)
        author_work_df: Pre-loaded author-work DataFrame (cached)
        candidate_authors_df: Pre-loaded candidate authors DataFrame (cached)
        known_target_edges_path: Output path for known target edges
        sample_type_to_date_range: Dictionary mapping sample types to date ranges
    """
    logger.info("Generating known target edges...")

    # Remove self-citations (author's own papers)
    self_work_df = author_work_df.select(
        col("author_id"),
        col("work_id").alias("cited_work_id")
    ).distinct()

    author_first_cite_df = author_first_cite_df.join(
        self_work_df,
        on=["author_id", "cited_work_id"],
        how='left_anti'
    )

    # Add sample_type labels
    sample_type_expr = None
    for sample_type, date_range in sample_type_to_date_range.items():
        cond = (col("date") >= date_range[0]) & (col("date") < date_range[1])
        if sample_type_expr is None:
            sample_type_expr = when(cond, sample_type)
        else:
            sample_type_expr = sample_type_expr.when(cond, sample_type)

    label_df = author_first_cite_df.withColumn("sample_type", sample_type_expr)
    label_df = label_df.filter(col("sample_type").isNotNull())

    # Normalize dates to Jan 1st of each year
    label_df = label_df.withColumn("date", trunc(col("date"), "year"))

    # Filter to only active authors using cached candidate_authors
    label_df = label_df.join(
        candidate_authors_df,
        on=["author_id", "date"],
        how="inner"
    )

    # Save results
    logger.info(f"Saving known target edges to {known_target_edges_path}")
    write_table(
        spark_ctx,
        label_df,
        known_target_edges_path,
        format='parquet',
        mode='overwrite',
        partition_columns=["sample_type"]
    )

    # Mark as complete
    spark_ctx.table_state.mark_complete(known_target_edges_path)

    logger.info("✓ Known target edges generated successfully")


def generate_exist_target_edges(spark_ctx, author_first_cite_df, author_work_df,
                               candidate_authors_df, exist_target_edges_path, target_dates):
    """
    Generate exist_target_edges table.

    This table contains papers that authors have already cited before each query date.
    It's used to filter out known relationships during bipath exploration.

    Processing steps:
    1. Use pre-computed author_first_cite_df (passed as parameter)
    2. For each query date, include citations made before that date
    3. Also include author's own papers (self-citations to exclude)
    4. Filter to only active authors using pre-computed candidate_authors DataFrame

    Args:
        spark_ctx: SparkContext from SparkRunner
        author_first_cite_df: Pre-computed first citation DataFrame (cached)
        author_work_df: Pre-loaded author-work DataFrame (cached)
        candidate_authors_df: Pre-loaded candidate authors DataFrame (cached)
        exist_target_edges_path: Output path for exist target edges
        target_dates: List of query dates to generate exist_target_edges for
    """
    logger.info("Generating exist target edges...")

    # Convert target_dates to partition format
    date_partitions = [[date] for date in target_dates]

    # Rename date to first_date
    author_first_cite_df = author_first_cite_df.withColumnRenamed('date', 'first_date')

    # Create DataFrame of target dates
    time_df = spark_ctx.spark.createDataFrame(
        [[date] for date in target_dates],
        ['date']
    )

    # Cross join with time points and filter
    exist_edge_df = author_first_cite_df.join(
        broadcast(time_df),
        col('first_date') < col('date'),
        how='inner'
    ).select(
        col('author_id').alias('u_node_id'),
        col('cited_work_id').alias('v_node_id'),
        col('date')
    ).distinct()

    # Add author's own papers (self-citations to exclude)
    self_work_df = author_work_df.select(
        col("author_id"),
        col("work_id"),
        col('date').alias("publish_date")
    ).join(
        broadcast(time_df),
        col('publish_date') < col('date'),
        how='inner'
    ).select(
        col('author_id').alias('u_node_id'),
        col('work_id').alias('v_node_id'),
        col('date')
    ).distinct()

    # Merge both sources
    exist_edge_df = exist_edge_df.unionByName(self_work_df).distinct()

    # Filter to only active authors using cached candidate_authors
    exist_edge_df = exist_edge_df.join(
        candidate_authors_df,
        (exist_edge_df.u_node_id == candidate_authors_df.author_id) &
        (exist_edge_df.date == candidate_authors_df.date),
        how="inner"
    ).select(
        exist_edge_df.u_node_id,
        exist_edge_df.v_node_id,
        exist_edge_df.date
    ).distinct()

    # Save results
    logger.info(f"Saving exist target edges to {exist_target_edges_path}")
    write_table(
        spark_ctx,
        exist_edge_df,
        exist_target_edges_path,
        format='parquet',
        mode='overwrite',
        partition_columns=["date"],
        partition_instances=date_partitions
    )

    # Mark as complete
    spark_ctx.table_state.mark_complete(
        exist_target_edges_path,
        partition_columns=['date'],
        partition_instances=date_partitions
    )

    logger.info("✓ Exist target edges generated successfully")


def generate_candidate_authors(spark_ctx, works_author_path, candidate_authors_path, target_dates):
    """
    Generate candidate_authors table.

    This table contains authors who published papers in the past 5 years relative
    to each query date. It's used to filter the search space to only active researchers.

    Processing steps:
    1. Get all author publication years
    2. For each query date, include authors who published 1-5 years before

    Args:
        spark_ctx: SparkContext from SparkRunner
        works_author_path: Path to works_author context table
        candidate_authors_path: Output path for candidate authors
        target_dates: List of query dates to generate candidate_authors for
    """

    # Convert target_dates to partition format
    date_partitions = [[date] for date in target_dates]

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(
        candidate_authors_path,
        partition_columns=['date'],
        partition_instances=date_partitions
    )
    if is_complete:
        logger.info(f"Candidate authors already exist at {candidate_authors_path}, skipping")
        return

    logger.info("Generating candidate authors...")

    # Read author publication years
    author_work_df = read_table(spark_ctx, works_author_path, format='parquet')
    author_work_df = author_work_df.select("author_id", "date")
    author_work_df = author_work_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    author_work_df = author_work_df.withColumn("publish_year", year("date"))
    author_work_df = author_work_df.drop("date").distinct()

    # Create DataFrame of target dates
    time_df = spark_ctx.spark.createDataFrame(
        [[date] for date in target_dates],
        ['date']
    )
    time_df = time_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
                     .withColumn("target_year", year("date")) \
                     .withColumn("min_year", col("target_year") - 5) \
                     .withColumn("max_year", col("target_year") - 1)

    # Cross join and filter to authors active in past 5 years
    valid_author_df = author_work_df.join(
        broadcast(time_df),
        (col('publish_year') >= col('min_year')) &
        (col('publish_year') <= col('max_year')),
        how='inner'
    ).select('author_id', 'date').distinct()

    # Save results
    logger.info(f"Saving candidate authors to {candidate_authors_path}")
    write_table(
        spark_ctx,
        valid_author_df,
        candidate_authors_path,
        format='parquet',
        mode='overwrite',
        partition_columns=["date"],
        partition_instances=date_partitions
    )

    # Mark as complete
    spark_ctx.table_state.mark_complete(
        candidate_authors_path,
        partition_columns=['date'],
        partition_instances=date_partitions
    )

    logger.info("✓ Candidate authors generated successfully")


# ==================== Main Orchestration ====================

def prepare_all_task_data(spark_ctx, context_table_dir, task_table_dir,
                         target_dates, sample_type_to_date_range):
    """
    Orchestrate generation of all task data tables with optimized computation order.

    Execution flow:
    1. Generate candidate_authors (handles completion internally, independent)
    2. Check if target edges need generation
    3. If needed, compute shared author_first_citation data once
    4. Generate known_target_edges and exist_target_edges using shared data
    5. Release cached data after completion

    Args:
        spark_ctx: SparkContext from SparkRunner
        context_table_dir: Base directory for context tables (with URI scheme)
        task_table_dir: Full path to task-specific directory (with URI scheme)
        target_dates: List of target dates for partitioning
        sample_type_to_date_range: Dictionary mapping sample types to date ranges
    """
    logger.info("=" * 60)
    logger.info("Starting task data preparation")
    logger.info(f"Context tables: {context_table_dir}")
    logger.info(f"Task output dir: {task_table_dir}")
    logger.info("=" * 60)

    # Construct input paths (context tables)
    works_author_path = f"{context_table_dir}/works_author"
    works_citation_path = f"{context_table_dir}/works_citation"

    # Construct output paths (task tables)
    known_target_edges_path = f"{task_table_dir}/known_target_edges"
    exist_target_edges_path = f"{task_table_dir}/exist_target_edges"
    candidate_authors_path = f"{task_table_dir}/candidate_authors"

    # Convert target_dates to partition format for completion check
    date_partitions = [[date] for date in target_dates]

    # Step 1: Generate candidate_authors (handles completion internally)
    logger.info("\n[1/3] Generating candidate authors...")
    generate_candidate_authors(
        spark_ctx,
        works_author_path,
        candidate_authors_path,
        target_dates
    )

    # Step 2: Check if we need to generate target edges
    known_complete, _ = spark_ctx.table_state.check_complete(known_target_edges_path)
    exist_complete, _ = spark_ctx.table_state.check_complete(
        exist_target_edges_path,
        partition_columns=['date'],
        partition_instances=date_partitions
    )

    # Generate target edges if at least one is incomplete
    if not known_complete or not exist_complete:
        # Compute shared data (author first citation + candidate authors)
        logger.info("\n[2/3] Computing shared data...")
        author_first_cite_df, author_work_df, candidate_authors_df = compute_author_first_citation(
            spark_ctx,
            works_author_path,
            works_citation_path,
            candidate_authors_path
        )

        # Step 3a: Generate known_target_edges if needed
        if not known_complete:
            logger.info("\n[3a/3] Generating known target edges...")
            generate_known_target_edges(
                spark_ctx,
                author_first_cite_df,
                author_work_df,
                candidate_authors_df,
                known_target_edges_path,
                sample_type_to_date_range
            )
        else:
            logger.info("\n[3a/3] Known target edges already complete, skipping")

        # Step 3b: Generate exist_target_edges if needed
        if not exist_complete:
            logger.info("\n[3b/3] Generating exist target edges...")
            generate_exist_target_edges(
                spark_ctx,
                author_first_cite_df,
                author_work_df,
                candidate_authors_df,
                exist_target_edges_path,
                target_dates
            )
        else:
            logger.info("\n[3b/3] Exist target edges already complete, skipping")

        # Release cached data
        logger.info("\nReleasing cached computation data...")
        spark_ctx.persist_manager.mark_released('target_edges_done')
    else:
        logger.info("\n[2-3/3] Both known and exist target edges already complete, skipping")

    logger.info("\n" + "=" * 60)
    logger.info("All task data prepared successfully!")
    logger.info("=" * 60)


# ==================== CLI Argument Parsing ====================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare task data for author-new-reference link discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all task data tables (local)
  python 1_prepare_task_data.py \\
      --bilink-config main/config/bilink/author_citation.yaml \\
      --context-table-dir file:///path/to/context_table

  # Generate on HDFS cluster
  python 1_prepare_task_data.py \\
      --bilink-config main/config/bilink/author_citation.yaml \\
      --context-table-dir hdfs:///user/example/.../context_table \\
      --spark-mode cluster --spark-platform example \\
      --backends local hdfs

  # Force regenerate all tables
  python 1_prepare_task_data.py \\
      --bilink-config main/config/bilink/author_citation.yaml \\
      --context-table-dir file:///path/to/context_table \\
      --force
        """
    )

    # Required arguments
    parser.add_argument('--bilink-config', required=True,
                        help='Path to BiLink configuration file (contains task_data_dir and sample_type_to_date_range)')
    parser.add_argument('--context-table-dir', required=True,
                        help='Base directory for context tables (with URI scheme, e.g., file:/// or hdfs:///)')

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
                        help='Force regenerate tables even if they already exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logger (configure the module-level logger)
    task_name = "prepare_task_data"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{PROJECT_ROOT}/data/logs/{task_name}/{timestamp}.log"
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logger(log_file, level=log_level)

    # Initialize FileIO
    backend_configs = {backend: {} for backend in args.backends}
    fileio = FileIO(backend_configs)
    logger.info(f"FileIO initialized with schemes: {list(fileio.backends.keys())}")

    # Read BiLink configuration
    logger.info(f"Reading BiLink configuration from {args.bilink_config}")
    bilink_config_path = f"{PROJECT_ROOT}/{args.bilink_config}"
    bilink_config = fileio.read_yaml(f"file://{bilink_config_path}")

    # Extract configuration parameters
    task_data_dir = bilink_config['task_data_dir']
    sample_type_to_date_range = bilink_config['target_edge']['sample_type_to_date_range']

    # Derive target_dates from sample_type_to_date_range
    target_dates = derive_target_dates(sample_type_to_date_range)

    logger.info(f"Context table directory: {args.context_table_dir}")
    logger.info(f"Task table directory: {task_data_dir}")
    logger.info(f"Sample type to date range: {sample_type_to_date_range}")
    logger.info(f"Derived target dates: {target_dates}")

    # Initialize SparkRunner
    spark_runner = SparkRunner(
        mode=args.spark_mode,
        platform=args.spark_platform,
        fileio=fileio,
        config_dict={},
        ignore_complete=args.force
    )

    # Run main function
    spark_runner.run(
        prepare_all_task_data,
        args.context_table_dir,
        task_data_dir,
        target_dates,
        sample_type_to_date_range
    )

    logger.info("Script completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
