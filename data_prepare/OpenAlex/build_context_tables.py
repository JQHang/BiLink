#!/usr/bin/env python3
"""
Build OpenAlex Context Tables

This script processes raw OpenAlex parquet data and creates normalized relationship tables
for graph construction. It builds 7 context tables from works data:
- works_basic: Basic work information with one-hot encoding
- works_citation: Citation relationships
- works_author: Work-author relationships
- works_institution: Work-institution relationships
- works_source: Work-source (journal/venue) relationships
- works_topic: Work-topic relationships
- author_institution: Author-institution affiliations

Features:
- Uses completion markers to skip already-built tables
- Persist manager for efficient DataFrame caching
- URI-aware file paths (file://, hdfs://, s3://)
- Supports multiple Spark data formats (parquet, orc, etc.)

Usage:
    python build_base_tables.py \
        --raw-data-path "file:///path/to/raw_data" \
        --context-table-path "file:///path/to/context_table" \
        [--works-format parquet] \
        [--works-dir works] \
        [--spark-mode local] \
        [--spark-platform localhost] \
        [--force] \
        [--verbose]
"""

import argparse
import sys
import logging
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import (
    add_months, trunc, col, lit, when, explode, count, regexp_replace, coalesce
)

from joinminer import PROJECT_ROOT
from joinminer.fileio import FileIO
from joinminer.spark import SparkRunner
from joinminer.spark.io import write_table
from joinminer.utils import setup_logger

# Module-level logger
logger = logging.getLogger("joinminer")


# ==================== Schema Definition ====================

WORKS_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("publication_date", StringType(), True),
    StructField("type", StringType(), True),
    StructField("language", StringType(), True),

    # Author information
    StructField("authorships", ArrayType(
        StructType([
            StructField("author_position", StringType(), True),
            StructField("author", StructType([
                StructField("id", StringType(), True),
                StructField("display_name", StringType(), True)
            ]), True),
            StructField("institutions", ArrayType(
                StructType([
                    StructField("id", StringType(), True),
                    StructField("display_name", StringType(), True)
                ])
            ), True)
        ])
    ), True),

    # Referenced works
    StructField("referenced_works", ArrayType(StringType()), True),

    # Source information
    StructField("primary_location", StructType([
        StructField("source", StructType([
            StructField("id", StringType(), True),
            StructField("display_name", StringType(), True),
            StructField("type", StringType(), True),
            StructField("is_oa", BooleanType(), True),
            StructField("is_in_doaj", BooleanType(), True)
        ]), True),
        StructField("is_oa", BooleanType(), True),
    ]), True),

    # Topic information
    StructField("topics", ArrayType(
        StructType([
            StructField("id", StringType(), True),
            StructField("display_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
    ), True)
])


# ==================== Helper Functions ====================

def clean_openalex_ids(df, id_columns):
    """
    Remove OpenAlex ID URL prefix.

    Converts https://openalex.org/W1234567890 to W1234567890

    Args:
        df: DataFrame
        id_columns: List of ID column names to clean

    Returns:
        DataFrame with cleaned IDs
    """
    for col_name in id_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, regexp_replace(col(col_name), "^https://openalex\\.org/", ""))
    return df


# ==================== Table Builder Functions ====================

def build_works_basic(spark_ctx, works_df, output_path):
    """Build and save works basic information table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    # Select required columns
    works_basic_df = works_df.select('id', 'type', 'language', 'date')

    # Convert type to one-hot format
    type_list = ["article", "book-chapter", "preprint", "dataset", "review"]
    works_basic_df = works_basic_df.groupBy('id', 'date', 'language') \
                                   .pivot("type", type_list) \
                                   .agg(lit(1)) \
                                   .fillna(0)

    for type_name in type_list:
        type_col = f"type_is_{type_name.replace('-', '_')}"
        works_basic_df = works_basic_df.withColumnRenamed(type_name, type_col)

    # Convert language to one-hot format
    works_basic_df = works_basic_df.withColumn(
                         'language_is_en',
                         when(col("language") == 'en', 1).otherwise(0)
                     ).drop('language')

    # Rename column
    works_basic_df = works_basic_df.withColumnRenamed('id', 'work_id')
    works_basic_df = works_basic_df.filter(col("work_id").isNotNull())

    # Clean OpenAlex ID prefix
    works_basic_df = clean_openalex_ids(works_basic_df, ['work_id'])

    # Save table
    write_table(spark_ctx, works_basic_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "type_is_article": "Work type is article (one-hot encoded)",
            "type_is_book_chapter": "Work type is book chapter (one-hot encoded)",
            "type_is_preprint": "Work type is preprint (one-hot encoded)",
            "type_is_dataset": "Work type is dataset (one-hot encoded)",
            "type_is_review": "Work type is review article (one-hot encoded)",
            "language_is_en": "Work language is English (one-hot encoded)"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    # Mark complete
    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_works_cite(spark_ctx, works_df, output_path):
    """Build and save works citation relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    works_cite_df = works_df.select('id', 'referenced_works', 'date')
    works_cite_df = works_cite_df.withColumn("cited_work_id",
                                             explode(col("referenced_works"))).drop("referenced_works")
    works_cite_df = works_cite_df.withColumnRenamed('id', 'work_id')
    works_cite_df = works_cite_df.filter((col("work_id").isNotNull()) &
                                         (col("cited_work_id").isNotNull()))

    works_cite_df = clean_openalex_ids(works_cite_df, ['work_id', 'cited_work_id'])
    write_table(spark_ctx, works_cite_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {}  # Pure relationship table, no features
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_works_author(spark_ctx, works_df, output_path):
    """Build and save works-author relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    works_author_df = works_df.select('id', 'authorships', 'date')
    works_author_df = works_author_df.withColumn("authorship",
                                                 explode(col("authorships"))).drop("authorships")
    works_author_df = works_author_df.select(
                            col("id").alias("work_id"),
                            "date",
                            col("authorship.author.id").alias("author_id"),
                            col("authorship.author_position").alias("author_position")
                        )
    works_author_df = works_author_df.filter((col("work_id").isNotNull()) &
                                             (col("author_id").isNotNull()))

    # Convert position to one-hot format
    works_author_df = works_author_df.withColumn(
                         'author_position_is_first',
                         when(col("author_position") == 'first', 1).otherwise(0)
                     )
    works_author_df = works_author_df.withColumn(
                         'author_position_is_middle',
                         when(col("author_position") == 'middle', 1).otherwise(0)
                     )
    works_author_df = works_author_df.withColumn(
                         'author_position_is_last',
                         when(col("author_position") == 'last', 1).otherwise(0)
                     )
    works_author_df = works_author_df.drop('author_position')

    works_author_df = clean_openalex_ids(works_author_df, ['work_id', 'author_id'])
    write_table(spark_ctx, works_author_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "author_position_is_first": "Author is first author (one-hot encoded)",
            "author_position_is_middle": "Author is middle author (one-hot encoded)",
            "author_position_is_last": "Author is last author (one-hot encoded)"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_works_institution(spark_ctx, works_df, output_path):
    """Build and save works-institution relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    works_institution_df = works_df.select('id', 'authorships', 'date')
    works_institution_df = works_institution_df.withColumn("authorship",
                                                           explode(col("authorships"))).drop("authorships")
    works_institution_df = works_institution_df.withColumn("institution",
                                                           explode(col("authorship.institutions")))
    works_institution_df = works_institution_df.select(
                                col("id").alias("work_id"),
                                "date",
                                col("institution.id").alias("institution_id"),
                                col("authorship.author_position").alias("author_position")
                            )
    works_institution_df = works_institution_df.filter((col("work_id").isNotNull()) &
                                                       (col("institution_id").isNotNull()))

    # Aggregate by work and institution
    works_institution_df = works_institution_df.groupby("work_id", "date", "institution_id") \
                                               .pivot("author_position", ['first', 'middle', 'last']) \
                                               .count() \
                                               .fillna(0, subset=['first', 'middle', 'last'])
    works_institution_df = works_institution_df.withColumnRenamed("first", "first_author_count") \
                                               .withColumnRenamed("middle", "middle_author_count") \
                                               .withColumnRenamed("last", "last_author_count")

    works_institution_df = clean_openalex_ids(works_institution_df, ['work_id', 'institution_id'])
    write_table(spark_ctx, works_institution_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "first_author_count": "Number of first authors from this institution",
            "middle_author_count": "Number of middle authors from this institution",
            "last_author_count": "Number of last authors from this institution"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_works_source(spark_ctx, works_df, output_path):
    """Build and save works-source relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    works_source_df = works_df.select(
                            col("id").alias("work_id"),
                            "date",
                            col("primary_location.source.id").alias("source_id"),
                            col("primary_location.source.type").alias("source_type"),
                            # OA flags with null handling
                            coalesce(col("primary_location.source.is_oa").cast("int"), lit(0)).alias("source_is_oa"),
                            col("primary_location.source.is_oa").isNull().cast("int").alias("source_is_oa_is_null"),
                            coalesce(col("primary_location.source.is_in_doaj").cast("int"), lit(0)).alias("source_is_in_doaj"),
                            col("primary_location.source.is_in_doaj").isNull().cast("int").alias("source_is_in_doaj_is_null"),
                            coalesce(col("primary_location.is_oa").cast("int"), lit(0)).alias("article_is_oa"),
                            col("primary_location.is_oa").isNull().cast("int").alias("article_is_oa_is_null")
                        )
    works_source_df = works_source_df.filter((col("work_id").isNotNull()) &
                                             (col("source_id").isNotNull()))

    # Convert source_type to one-hot format
    works_source_df = works_source_df.withColumn(
                         'source_type_is_journal',
                         when(col("source_type") == 'journal', 1).otherwise(0)
                     )
    works_source_df = works_source_df.withColumn(
                         'source_type_is_book_series',
                         when(col("source_type") == 'book series', 1).otherwise(0)
                     )
    works_source_df = works_source_df.withColumn(
                         'source_type_is_conference',
                         when(col("source_type") == 'conference', 1).otherwise(0)
                     )
    works_source_df = works_source_df.withColumn(
                         'source_type_is_ebook_platform',
                         when(col("source_type") == 'ebook platform', 1).otherwise(0)
                     )
    works_source_df = works_source_df.withColumn(
                         'source_type_is_repository',
                         when(col("source_type") == 'repository', 1).otherwise(0)
                     )
    works_source_df = works_source_df.drop('source_type')

    works_source_df = clean_openalex_ids(works_source_df, ['work_id', 'source_id'])
    write_table(spark_ctx, works_source_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "source_is_oa": "Source is open access (null filled with 0)",
            "source_is_oa_is_null": "Source OA status is null (1 if null, 0 otherwise)",
            "source_is_in_doaj": "Source is in Directory of Open Access Journals (null filled with 0)",
            "source_is_in_doaj_is_null": "Source DOAJ status is null (1 if null, 0 otherwise)",
            "article_is_oa": "Article is open access (null filled with 0)",
            "article_is_oa_is_null": "Article OA status is null (1 if null, 0 otherwise)",
            "source_type_is_journal": "Source type is journal (one-hot encoded)",
            "source_type_is_book_series": "Source type is book series (one-hot encoded)",
            "source_type_is_conference": "Source type is conference (one-hot encoded)",
            "source_type_is_ebook_platform": "Source type is ebook platform (one-hot encoded)",
            "source_type_is_repository": "Source type is repository (one-hot encoded)"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_works_topic(spark_ctx, works_df, output_path):
    """Build and save works-topic relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    works_topic_df = works_df.select(
                            col("id").alias("work_id"),
                            "date",
                            "topics"
                        )
    works_topic_df = works_topic_df.withColumn("topic",
                                               explode(col("topics"))).drop("topics")
    works_topic_df = works_topic_df.select(
                            "work_id",
                            "date",
                            col("topic.id").alias("topic_id"),
                            coalesce(col("topic.score"), lit(0.0)).alias("topic_score"),
                            col("topic.score").isNull().cast("int").alias("topic_score_is_null")
                        )
    works_topic_df = works_topic_df.filter((col("work_id").isNotNull()) &
                                           (col("topic_id").isNotNull()))

    works_topic_df = clean_openalex_ids(works_topic_df, ['work_id', 'topic_id'])
    write_table(spark_ctx, works_topic_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "topic_score": "Relevance score for this topic (0-1, null filled with 0)",
            "topic_score_is_null": "Topic score is null (1 if null, 0 otherwise)"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


def build_author_institution(spark_ctx, works_df, output_path):
    """Build and save author-institution relationships table."""
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Table {output_path} already complete, skipping")
        return

    logger.info(f"Building table {output_path}...")

    author_institution_df = works_df.select('id', 'authorships', 'date')
    author_institution_df = author_institution_df.withColumn("authorship",
                                                             explode(col("authorships"))).drop("authorships")
    author_institution_df = author_institution_df.withColumn("institution",
                                                             explode(col("authorship.institutions")))
    author_institution_df = author_institution_df.select(
                                col("authorship.author.id").alias("author_id"),
                                "date",
                                col("institution.id").alias("institution_id")
                            )
    author_institution_df = author_institution_df.filter((col("author_id").isNotNull()) &
                                                         (col("institution_id").isNotNull()))

    # Aggregate statistics
    author_institution_df = author_institution_df.groupby("author_id", "date", "institution_id") \
                                                 .agg(count('*').alias('author_use_institution_count'))

    author_institution_df = clean_openalex_ids(author_institution_df, ['author_id', 'institution_id'])
    write_table(spark_ctx, author_institution_df, output_path, format='parquet', mode='overwrite')

    # Generate and save context_info
    context_info = {
        "partition_columns": [],
        "feature_columns": {
            "author_use_institution_count": "Number of times author affiliated with institution"
        }
    }
    spark_ctx.fileio.write_json(f"{output_path}/_context_info.json", context_info)

    spark_ctx.table_state.mark_complete(output_path)
    logger.info(f"Table {output_path} built successfully")


# ==================== Main Function ====================

def build_context_tables(spark_ctx, args):
    """
    Build OpenAlex context tables from works data.

    Args:
        spark_ctx: SparkRunner instance
        args: Command-line arguments
    """
    # Read works data
    works_path = f"{args.raw_data_path}/{args.works_dir}"
    logger.info(f"Reading works data from: {works_path} (format: {args.works_format})")

    works_df = spark_ctx.spark.read.format(args.works_format).schema(WORKS_SCHEMA).load(works_path)

    # Set publication_month as partition column 'date'
    works_df = works_df.withColumn(
        "date",
        trunc(add_months(col("publication_date"), 1), "month")
    )

    # Persist for efficient reuse
    works_df = spark_ctx.persist_manager.persist(
        works_df,
        release_point='all_tables_done',
        name='works_df'
    )
    logger.info("Works data loaded and persisted")

    # Table building configuration
    tables = [
        ('works_basic', build_works_basic),
        ('works_citation', build_works_cite),
        ('works_author', build_works_author),
        ('works_institution', build_works_institution),
        ('works_source', build_works_source),
        ('works_topic', build_works_topic),
        ('author_institution', build_author_institution),
    ]

    # Build tables
    for table_name, builder_func in tables:
        logger.info(f"Building {table_name} table...")
        output_path = f"{args.context_table_path}/{table_name}"
        builder_func(spark_ctx, works_df, output_path)

    # Release persisted DataFrames
    logger.info("Releasing persisted DataFrames...")
    spark_ctx.persist_manager.mark_released('all_tables_done')

    logger.info("All context tables built successfully!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Build OpenAlex context tables from raw works data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build tables with default settings
  python build_base_tables.py \\
      --raw-data-path "file:///data/openalex/raw" \\
      --context-table-path "file:///data/openalex/context_tables"

  # Force rebuild all tables
  python build_base_tables.py \\
      --raw-data-path "file:///data/openalex/raw" \\
      --context-table-path "file:///data/openalex/context_tables" \\
      --force
        """
    )

    # Required arguments
    parser.add_argument('--raw-data-path', required=True,
                        help='Path to raw OpenAlex data (must include URI scheme: file://, hdfs://, s3://)')
    parser.add_argument('--context-table-path', required=True,
                        help='Output path for context tables (must include URI scheme)')

    # Optional arguments
    parser.add_argument('--works-format', default='parquet',
                        help='Input data format (default: parquet)')
    parser.add_argument('--works-dir', default='works',
                        help='Works directory name within raw-data-path (default: works)')
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
                        help='Force rebuild tables even if they already exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    task_name = "openalex_preprocess"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{PROJECT_ROOT}/data/logs/{task_name}/{timestamp}.log"
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(log_file, level=log_level)

    try:
        # Initialize FileIO
        backend_configs = {backend: {} for backend in args.backends}
        fileio = FileIO(backend_configs)
        logger.info(f"FileIO initialized with schemes: {list(fileio.backends.keys())}")

        # Initialize SparkRunner
        spark_runner = SparkRunner(
            mode=args.spark_mode,
            platform=args.spark_platform,
            fileio=fileio,
            config_dict={"spark.sql.parquet.datetimeRebaseModeInWrite": "CORRECTED"},
            ignore_complete=args.force  # Pass --force flag to ignore completion checks
        )

        # Run main function
        spark_runner.run(build_context_tables, args)

        logger.info("Script completed successfully")
        return 0

    except Exception as e:
        logging.error(f"Script failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
