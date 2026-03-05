"""
Table reading operations for PySpark.

Provides functions for reading tables from various file systems with partition support.
"""

import logging
from typing import List, Optional, Any
from pyspark.sql import SparkSession, DataFrame

from joinminer.spark.io.partition import build_partition_path

logger = logging.getLogger(__name__)


def read_table(spark_ctx,
               path: str,
               format: str = 'parquet',
               partition_columns: Optional[List[str]] = None,
               partition_instances: Optional[List[List[str]]] = None,
               schema=None,
               **options) -> DataFrame:
    """
    Read table data from file system.

    Args:
        spark_ctx: SparkRunner instance
        path: Table path with URI scheme (file://, hdfs://, s3://)
        format: File format (parquet/orc/csv/json)
        partition_columns: List of partition column names
        partition_instances: List of partition instances to read
                           格式: [['2021-01-01'], ['2021-01-02']]
        schema: Optional schema for the DataFrame
        **options: Additional Spark read options

    Returns:
        DataFrame

    Example:
        >>> # Read entire table
        >>> df = read_table(spark_runner, 'file:///data/users', format='parquet')

        >>> # Read specific partition instances
        >>> df = read_table(
        ...     spark_runner, 'hdfs:///data/events',
        ...     partition_columns=['date'],
        ...     partition_instances=[['2021-01-01'], ['2021-01-02']]
        ... )

    Note:
        For partition completeness checks, use spark_ctx.table_state.check_complete()
        before calling this function.
    """
    spark = spark_ctx.spark
    # Display partition info for logging
    partition_display = (
        partition_instances if partition_instances and len(partition_instances) <= 5
        else (partition_instances[:5] + ['...'] if partition_instances else [])
    )
    logger.info(
        f"Reading table from {path} (format={format}, "
        f"partition_columns={partition_columns}, "
        f"num_partitions={len(partition_instances) if partition_instances else 0}, "
        f"partition_instances={partition_display})"
    )

    # Create reader
    reader = spark.read.format(format)

    # Add schema if provided
    if schema is not None:
        reader = reader.schema(schema)

    # Add additional options
    for key, value in options.items():
        reader = reader.option(key, value)

    # Read data
    if not partition_columns or not partition_instances:
        # Non-partitioned table or reading entire table
        df = reader.load(path)
    else:
        # Read specific partitions
        partition_paths = []
        for partition_instance in partition_instances:
            if len(partition_columns) != len(partition_instance):
                raise ValueError(
                    f"Number of partition columns ({len(partition_columns)}) "
                    f"and values ({len(partition_instance)}) don't match"
                )

            # Build partition path using utility function
            partition_path = build_partition_path(path, partition_columns, partition_instance)
            partition_paths.append(partition_path)

        partition_path_display = (
            partition_paths if len(partition_paths) <= 5
            else partition_paths[:5] + ['...']
        )
        logger.info(f"Reading {len(partition_paths)} specific partition paths: {partition_path_display}")

        # Read specified partitions with basePath to preserve partition columns
        df = reader.option("basePath", path).load(partition_paths)

    return df


__all__ = ['read_table']
