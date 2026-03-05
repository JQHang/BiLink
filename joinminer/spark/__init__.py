"""
Spark module for JoinMiner.

This module provides a comprehensive set of tools for working with PySpark,
organized into functional categories:

- SparkRunner: Core runtime for Spark session management (root level)
- managers: Auxiliary managers (PersistManager, PartitionManager)
- operations: DataFrame operations (filter, join, aggregate, assemble, sample, etc.)
- io: Input/output operations (read_table, write_table, cleanup_table)
- platforms: Platform-specific implementations (example, localhost)

使用示例:
    # Core runtime (root level)
    from joinminer.spark import SparkRunner

    # Managers
    from joinminer.spark.managers import PersistManager, PartitionManager

    # DataFrame operations
    from joinminer.spark.operations import (
        filter, aggregate, join,
        select, fillna, assemble, sample_random
    )

    # I/O operations
    from joinminer.spark.io import read_table, write_table, cleanup_table

子模块说明:
    - spark_runner.py: 核心 Spark 会话管理（顶层）
    - managers/: 辅助管理器（持久化、分区管理）
    - operations/: DataFrame 转换操作（函数）
    - io/: 输入输出操作（函数）
    - platforms/: 平台特定配置
"""

from joinminer.spark.spark_runner import SparkRunner, SparkContext

__all__ = ['SparkRunner', 'SparkContext']
