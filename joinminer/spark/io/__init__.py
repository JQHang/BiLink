"""
Input/Output operations for Spark DataFrames.

This module provides functions for reading from and writing to various storage systems,
including local file systems, HDFS, and S3.

All I/O functions support URI-aware paths (file://, hdfs://, s3://).

Submodules:
    - read: Read DataFrames from storage
    - write: Write DataFrames to storage
    - cleanup: Clean up table files/partitions
    - partition: Partition utility functions
    - size_estimator: Estimate DataFrame storage size (utility for write operations)
"""

from .read import read_table
from .write import write_table
from .show import show_table
from .cleanup import cleanup_table
from .partition import build_partition_path, parse_partition_spec
from .size_estimator import (
    TYPE_SIZES,
    estimate_vector_size,
    fill_column_sizes,
)

__all__ = [
    'read_table',
    'write_table',
    'show_table',
    'cleanup_table',
    'build_partition_path',
    'parse_partition_spec',
    'TYPE_SIZES',
    'estimate_vector_size',
    'fill_column_sizes',
]
