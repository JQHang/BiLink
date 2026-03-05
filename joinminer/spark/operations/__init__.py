"""
DataFrame operations for data manipulation.

This module provides various operations on Spark DataFrames including
filtering, joining, aggregating, sampling, feature assembly, and other transformations.

All operations are stateless functions that take a DataFrame as input and return
a transformed DataFrame.
"""

from .aggregate import aggregate
from .assemble import assemble
from .filter import filter, filter_by_columns
from .fillna import fillna
from .join import join
from .row import add_row_number
from .salt import salt_skewed_keys, replicate_for_salted_join
from .sample import random_sample, ordered_sample
from .select import select

__all__ = [
    # Aggregation
    'aggregate',
    # Feature assembly
    'assemble',
    # Filtering
    'filter',
    'filter_by_columns',
    # Null handling
    'fillna',
    # Joining
    'join',
    # Row operations
    'add_row_number',
    # Skew handling
    'salt_skewed_keys',
    'replicate_for_salted_join',
    # Sampling
    'random_sample',
    'ordered_sample',
    # Selection
    'select',
]
