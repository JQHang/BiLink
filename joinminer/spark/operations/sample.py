"""
Sampling operations for PySpark DataFrames.

This module provides functions for sampling rows from DataFrames using various strategies:
- Random sampling per group
- Random sampling with skew handling (pre-filtering)
- Ordered (top-N) sampling per group
"""

import logging
from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, rand
from pyspark.sql.window import Window
import pyspark.sql.functions as F

from .salt import salt_skewed_keys

logger = logging.getLogger(__name__)


def random_sample(df, group_columns, n):
    """
    Samples up to 'n' random records for each group defined by 'group_columns'.

    Args:
        df: DataFrame to sample from
        group_columns: List of column names defining groups for sampling
        n: Maximum number of samples per group

    Returns:
        DataFrame with up to n random records per group

    Example:
        >>> df = random_sample(df, group_columns=['user_id'], n=10)
    """
    logger.info(f"Random sampling {n} rows per group (columns: {group_columns})")

    window_spec = Window.partitionBy(*group_columns).orderBy(rand())
    sampled_df = (df
        .withColumn("_row_number", F.row_number().over(window_spec))
        .filter(col("_row_number") <= n)
        .drop("_row_number")
    )

    return sampled_df


def skewed_random_sample(
    spark_ctx,
    df: DataFrame,
    group_columns: List[str],
    n: int,
    release_point: str
) -> DataFrame:
    """
    Random sample with pre-filtering for skewed groups.

    This function first identifies and caps skewed groups using salt_skewed_keys,
    then performs random sampling. Designed for cases with severe data skew where
    a few groups dominate the data distribution.

    The pre-filtering step prevents excessive memory usage and computation time
    during the groupBy+sample operation by capping each group at approximately
    1 million records before sampling.

    Fixed parameters for consistent skew handling:
    - skew_threshold: 10000 (groups with > 10k records are considered skewed)
    - salt_buckets: 100 (caps each skewed group at ~1M records = 10k * 100)

    Args:
        spark_ctx: Spark context with persist_manager (required for persistence)
        df: Input DataFrame to sample from
        group_columns: List of column names defining groups for sampling
        n: Number of rows to sample per group (must be <= 1,000,000)
        release_point: When to release persisted data (required for lifecycle management)

    Returns:
        DataFrame with n randomly sampled rows per group

    Raises:
        ValueError: If n > 1,000,000 (groupBy+sampling not suitable for larger values)

    Example:
        >>> # Sample 1000 paths per intersection node with automatic skew control
        >>> sampled_df = skewed_random_sample(
        ...     spark_ctx=spark_ctx,
        ...     df=backward_paths,
        ...     group_columns=['node_id', 'node_type'],
        ...     n=1000,
        ...     release_point='sampling_done'
        ... )

    Note:
        This function is more expensive than random_sample() due to the pre-filtering
        overhead. Only use it when you expect severe data skew (e.g., power-law
        distributions in graph data).
    """
    # Validate n to prevent misuse
    if n > 1000000:
        raise ValueError(
            f"n={n} exceeds maximum of 1,000,000. "
            f"GroupBy+sampling approach is not suitable for larger values. "
            f"Consider alternative sampling strategies for n > 1M."
        )

    # Fixed parameters for consistent behavior across the codebase
    SKEW_THRESHOLD = 10000
    SALT_BUCKETS = 100

    logger.info(f"Skewed random sampling {n} rows per group with pre-filtering "
                f"(threshold={SKEW_THRESHOLD}, buckets={SALT_BUCKETS})")

    # Pre-filter using salt_skewed_keys to identify and cap skewed groups
    # This caps each group at approximately SKEW_THRESHOLD * SALT_BUCKETS = 1M records
    # The salt column is used for filtering, not for actual salted join
    filtered_df, _ = salt_skewed_keys(
        spark_ctx=spark_ctx,
        df=df,
        join_columns=group_columns,
        skew_threshold=SKEW_THRESHOLD,
        salt_buckets=SALT_BUCKETS,
        salt_column_name="_temp_salt_for_sampling",
        release_point=release_point
    )

    # Drop the temporary salt column (we only used it for pre-filtering)
    filtered_df = filtered_df.drop("_temp_salt_for_sampling")

    logger.debug("Pre-filtering complete, performing random sampling")

    # Perform random sampling on the filtered data using standard random_sample
    return random_sample(filtered_df, group_columns, n)


def ordered_sample(df, group_columns, order_config, n):
    """
    Select top N records for each group based on ordering configuration.

    Args:
        df: DataFrame to sample from
        group_columns: List of column names defining groups
        order_config: Dict mapping column names to sort direction
                     e.g., {"score": "desc", "date": "asc"}
        n: Number of records to keep per group

    Returns:
        DataFrame with top n records per group according to ordering

    Example:
        >>> df = ordered_sample(
        ...     df,
        ...     group_columns=['user_id'],
        ...     order_config={'score': 'desc', 'timestamp': 'asc'},
        ...     n=5
        ... )
    """
    logger.info(f"Ordered sampling top {n} rows per group (columns: {group_columns}, order: {order_config})")

    # Build ordering list
    order_list = []
    for col_name, direction in order_config.items():
        if direction.lower() == "desc":
            order_list.append(F.col(col_name).desc())
        else:
            order_list.append(F.col(col_name).asc())

    # Add randomness to handle ties
    order_list.append(F.rand())

    # Apply window function
    window_spec = Window.partitionBy(*group_columns).orderBy(*order_list)

    result_df = (df
        .withColumn("_rank", F.row_number().over(window_spec))
        .filter(F.col("_rank") <= n)
        .drop("_rank")
    )

    return result_df


__all__ = [
    'random_sample',
    'skewed_random_sample',
    'ordered_sample',
]
