"""
Table cleanup operations for PySpark.

Provides functions for cleaning up (deleting) table or partition data.
"""

import logging
from typing import List, Optional

from joinminer.spark.io.partition import build_partition_path

logger = logging.getLogger(__name__)


def cleanup_table(spark_ctx,
                  base_path: str,
                  partition_columns: List[str] = None,
                  partition_instances: List[List[str]] = None) -> None:
    """
    清理表或指定分区的数据 - URI-aware

    警告：这是破坏性操作，会永久删除数据文件。

    Args:
        spark_ctx: SparkRunner instance
        base_path: 表的基础路径，可以包含URI scheme
        partition_columns: 分区列名列表（None表示非分区表）
        partition_instances: 要清理的分区实例列表（None表示清理整个表）
                            格式: [['2021-01-01'], ['2022-01-01']]

    Examples:
        >>> # 清理非分区表
        >>> cleanup_table(spark_runner, 'hdfs:///data/table')

        >>> # 清理分区表的所有数据
        >>> cleanup_table(spark_runner, 'hdfs:///data/table', ['date'])

        >>> # 清理特定分区实例
        >>> cleanup_table(
        ...     spark_runner,
        ...     'hdfs:///data/table',
        ...     ['date'],
        ...     [['2021-01-01'], ['2022-01-01']]
        ... )

    Note:
        This is typically called by write_table() when mode='overwrite'.
        For manual cleanup, ensure you have backups before calling this function.
    """
    fileio = spark_ctx.fileio

    if not partition_columns:
        # 非分区表：删除整个目录
        if fileio.exists(base_path):
            fileio.delete(base_path, recursive=True)
            logger.info(f"Cleaned up table: {base_path}")
    elif partition_instances:
        # 分区表：删除指定分区
        for partition_instance in partition_instances:
            partition_path = build_partition_path(
                base_path, partition_columns, partition_instance
            )
            if fileio.exists(partition_path):
                fileio.delete(partition_path, recursive=True)
                logger.debug(f"Cleaned up partition: {partition_path}")
        logger.info(f"Cleaned up {len(partition_instances)} partition instances in {base_path}")
    else:
        # 分区表但未指定分区：删除整个表
        if fileio.exists(base_path):
            fileio.delete(base_path, recursive=True)
            logger.info(f"Cleaned up entire partitioned table: {base_path}")


__all__ = ['cleanup_table']
