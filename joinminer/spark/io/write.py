"""
Table writing operations for PySpark.

Provides functions for writing tables to various file systems with intelligent
file size optimization and partition handling.
"""

import logging
from typing import List, Optional, Dict
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, lit, rand, broadcast, expr
from pyspark.sql.functions import count as _count, sum as _sum

from joinminer.utils import time_costing
from joinminer.spark.io.cleanup import cleanup_table
from joinminer.spark.io.show import show_table
from joinminer.spark.io.size_estimator import fill_column_sizes

logger = logging.getLogger(__name__)

# Write operation constants
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB
MAX_NUM_FILES = 1500


def _calculate_output_stats(
    df: DataFrame,
    col_sizes: Dict[str, int],
    partition_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Calculate row count, estimated size and file count for each output path.

    Internally collects to Python to avoid Spark expression parsing issues
    with many columns, then returns a DataFrame created from Python data.

    Args:
        df: Input DataFrame
        col_sizes: Column size configuration
            - int: fixed size → count(col) * size
            - str: expression → sum(expr)
        partition_columns: Partition columns (optional)

    Returns:
        DataFrame with columns:
        - partition_columns (if any)
        - row_count: row count for this path
        - _size_{col_name}: size estimate for each column
        - estimated_size: total estimated size in bytes
        - num_files: optimal file count based on MAX_FILE_SIZE
    """
    from pyspark.sql import Row

    # Build aggregation expressions
    agg_exprs = [_count("*").alias("row_count")]
    size_col_names = []

    for col_name, size_spec in col_sizes.items():
        size_col = f"_size_{col_name}"
        size_col_names.append(size_col)

        if isinstance(size_spec, int):
            # Fixed size: count(col) * size
            agg_exprs.append((_count(col_name) * lit(size_spec)).alias(size_col))
        else:
            # Expression: sum(expr)
            agg_exprs.append(_sum(expr(size_spec)).alias(size_col))

    # Partition or global aggregation
    if partition_columns:
        agg_df = df.groupBy(*partition_columns).agg(*agg_exprs)
    else:
        agg_df = df.agg(*agg_exprs)

    # Collect to Python (tiny DataFrame, avoids Spark expression issues)
    stats_rows = agg_df.collect()

    # Compute estimated_size and num_files in Python
    stats_list = []
    for row in stats_rows:
        row_dict = row.asDict()
        estimated_size = sum(row_dict.get(c) or 0 for c in size_col_names)
        num_files = int(estimated_size / MAX_FILE_SIZE) + 1
        row_dict["estimated_size"] = estimated_size
        row_dict["num_files"] = num_files
        stats_list.append(row_dict)

    # Create DataFrame from Python data
    stats_df = df.sparkSession.createDataFrame([Row(**r) for r in stats_list])

    return stats_df


def _assign_file_num(
    df: DataFrame,
    stats_df: DataFrame,
    total_files: int,
    partition_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Assign file_num to each row based on statistics.

    Args:
        df: Input DataFrame
        stats_df: Pre-calculated statistics DataFrame
        total_files: Total number of files to write
        partition_columns: Partition columns (optional)

    Returns:
        DataFrame with file_num column
    """
    if partition_columns:
        # Calculate cumulative file count for global file numbering
        window_spec = Window.orderBy(*partition_columns)
        partition_stats = stats_df.withColumn(
            "cumulative_num_files",
            _sum("num_files").over(window_spec) - col("num_files")
        ).select(
            *partition_columns, "num_files", "cumulative_num_files"
        )

        # Join and assign file_num
        df_with_num_files = df.join(broadcast(partition_stats), on=partition_columns, how="left")
        df_with_file_num = df_with_num_files.withColumn(
            "file_num",
            (rand() * col("num_files")).cast("int") + col("cumulative_num_files")
        ).drop("num_files", "cumulative_num_files")
    else:
        # Non-partitioned: simple global file_num
        df_with_file_num = df.withColumn("file_num", (rand() * total_files).cast("int"))

    return df_with_file_num


def _execute_write(
    spark_ctx,
    df_with_file_num: DataFrame,
    path: str,
    format: str,
    mode: str,
    total_files: int,
    partition_columns: Optional[List[str]] = None,
    partition_instances: Optional[List[List[str]]] = None,
    options: Optional[Dict] = None
) -> None:
    """
    Repartition and write DataFrame, with batch support for large file counts.

    Args:
        spark_ctx: SparkRunner instance (for cleanup)
        df_with_file_num: DataFrame with file_num column
        path: Output path with URI scheme
        format: File format (parquet/orc/csv/json)
        mode: Write mode (overwrite/append)
        total_files: Total number of files to write
        partition_columns: Partition columns (optional)
        partition_instances: Partition instances for cleanup (optional)
        options: Additional Spark write options (includes maxRecordsPerFile)
    """
    # Repartition
    df_repartitioned = df_with_file_num.repartition(total_files, "file_num")

    if total_files < MAX_NUM_FILES:
        # Write in one batch
        df_batch = df_repartitioned.drop("file_num")
        writer = df_batch.write.format(format).mode(mode)
        if partition_columns:
            writer = writer.partitionBy(*partition_columns)
        for key, value in (options or {}).items():
            writer = writer.option(key, value)
        writer.save(path)
    else:
        # Multi-batch write
        if mode == 'overwrite':
            cleanup_table(spark_ctx, path, partition_columns, partition_instances)

        df_repartitioned.persist()

        for num_files_start in range(0, total_files, MAX_NUM_FILES):
            num_files_end = min(num_files_start + MAX_NUM_FILES, total_files)
            df_batch = df_repartitioned.filter(
                (col("file_num") >= num_files_start) & (col("file_num") < num_files_end)
            ).drop("file_num")

            writer = df_batch.write.format(format).mode("append")
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            for key, value in (options or {}).items():
                writer = writer.option(key, value)
            writer.save(path)

        df_repartitioned.unpersist()


@time_costing
def write_table(spark_ctx,
                df: DataFrame,
                path: str,
                format: str = 'parquet',
                mode: str = 'overwrite',
                partition_columns: Optional[List[str]] = None,
                partition_instances: Optional[List[List[str]]] = None,
                col_sizes: Optional[Dict[str, int]] = None,
                **options) -> None:
    """
    Write table data with intelligent file size optimization.

    Args:
        spark_ctx: SparkRunner instance
        df: DataFrame to save
        path: Output path with URI scheme (file://, hdfs://, s3://)
        format: File format (parquet/orc/csv/json)
        mode: Write mode (overwrite/append/errorIfExists/ignore)
        partition_columns: List of partition column names
        partition_instances: Optional partition instances for cleanup in multi-batch writes.
                           Format: [['2021-01-01'], ['2021-01-02']]
                           If not provided, will be auto-extracted when needed.
        col_sizes: Column size estimates dict (optional)
        **options: Additional Spark write options

    Example:
        >>> # Write non-partitioned table
        >>> write_table(spark_runner, df, 'file:///data/output')

        >>> # Write partitioned table
        >>> write_table(
        ...     spark_runner, df, 'hdfs:///data/events',
        ...     partition_columns=['date'],
        ...     partition_instances=[['2021-01-01'], ['2021-01-02']]
        ... )

    Note:
        - In overwrite mode, Spark automatically cleans up for single-batch writes.
        - For multi-batch writes, cleanup_table() is called explicitly before writing.
        - Completion marking should be done separately using spark_ctx.table_state.mark_complete().
    """
    if col_sizes is None:
        col_sizes = {}

    # Persist DataFrame
    df = df.persist()

    # Show sample data
    show_table(df)

    # Fill column sizes (user-specified or type-based defaults)
    col_sizes_filled = fill_column_sizes(df, col_sizes)

    # Log schema with estimated sizes
    logger.info(f"Writing table to {path} ({len(df.columns)} columns):")
    for field in df.schema.fields:
        col_name = field.name
        col_type = field.dataType.simpleString()
        size = col_sizes_filled[col_name]
        logger.info(f"  {col_name:<30} {col_type:<20} {size}")

    # Calculate output statistics (collects internally, returns DataFrame from Python data)
    stats_df = _calculate_output_stats(df, col_sizes_filled, partition_columns)
    # No persist() needed - DataFrame created from Python data, already in memory

    # Show detail info: _size_{col} columns
    size_cols = (partition_columns or []) + [f"_size_{col_name}" for col_name in col_sizes_filled.keys()]
    logger.info("Output statistics (column sizes):")
    show_table(stats_df.select(*size_cols))

    # Show core info: partition columns + row_count + estimated_size + num_files
    core_cols = (partition_columns or []) + ["row_count", "estimated_size", "num_files"]
    logger.info("Output statistics (core):")
    show_table(stats_df.select(*core_cols))

    # Calculate totals
    totals = stats_df.agg(
        _sum("row_count").alias("total_rows"),
        _sum("estimated_size").alias("total_size"),
        _sum("num_files").alias("total_files")
    ).collect()[0]

    total_rows = totals["total_rows"]
    total_size = totals["total_size"]
    total_files = totals["total_files"]

    if total_rows == 0:
        logger.info("No data to save, skipping")
        df.unpersist()
        return

    # Calculate max rows per file based on average row size
    avg_row_size = total_size / total_rows
    file_row_count = int((MAX_FILE_SIZE / avg_row_size) * 1.05)

    logger.info(
        f"Total: {total_rows} rows, {total_size} bytes, {total_files} files. "
        f"Avg row size: {avg_row_size:.1f} bytes, max rows per file: {file_row_count}"
    )

    # Add to options for write
    options["maxRecordsPerFile"] = file_row_count

    # Extract partition_instances (if partitioned and not provided)
    if partition_columns and partition_instances is None:
        distinct_rows = stats_df.select(*partition_columns).collect()
        partition_instances = [[row[c] for c in partition_columns] for row in distinct_rows]
        logger.info(
            f"Auto-extracted {len(partition_instances)} partition instances: "
            f"{partition_instances[:5]}{'...' if len(partition_instances) > 5 else ''}"
        )

    # Assign file_num to rows
    df_with_file_num = _assign_file_num(df, stats_df, total_files, partition_columns)

    # Write with batching support
    _execute_write(
        spark_ctx=spark_ctx,
        df_with_file_num=df_with_file_num,
        path=path,
        format=format,
        mode=mode,
        total_files=total_files,
        partition_columns=partition_columns,
        partition_instances=partition_instances,
        options=options
    )

    # Cleanup
    df.unpersist()


__all__ = ['write_table']
