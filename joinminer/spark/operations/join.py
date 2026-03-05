from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, lit, rand, floor, ceil, least,
    explode, sequence, broadcast, when
)

import logging
from typing import Union, List, Tuple
from .salt import salt_skewed_keys, replicate_for_salted_join

logger = logging.getLogger(__name__)


def join(left_df, right_df, join_columns, join_type='inner', broadcast_hint = False):
    """
    Join two PySpark DataFrames.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        join_columns: Columns to join on, can be:
            - str: Single column name
            - list: List of column names
        join_type: Type of join ('inner', 'left', 'right', 'outer', 'semi', 'anti', 'cross')
        
    Returns:
        Joined DataFrame
    """
    # Validate join type
    valid_join_types = ['inner', 'left', 'right', 'outer', 'semi', 'anti', 'cross']
    if join_type not in valid_join_types:
        raise ValueError(f"Invalid join_type '{join_type}'. Must be one of {valid_join_types}")
    
    # Check for duplicate columns (excluding join columns)
    if join_type not in ['semi', 'anti', 'cross']:
        join_cols_set = set(join_columns if isinstance(join_columns, list) else [join_columns])
        left_cols = set(left_df.columns) - join_cols_set
        right_cols = set(right_df.columns) - join_cols_set
        duplicate_cols = left_cols & right_cols
        
        if duplicate_cols:
            raise ValueError(
                f"Found duplicate columns in DataFrames: {duplicate_cols}. "
                f"Please rename columns before joining."
            )
    
    # Log join operation
    logger.info(f"Perform {join_type} join on columns: {join_columns}")
    
    # Perform join
    if join_type == 'cross':
        result_df = left_df.crossJoin(right_df)
    else:
        if broadcast_hint:
            result_df = left_df.join(broadcast(right_df), join_columns, join_type)
        else:
            result_df = left_df.join(right_df, join_columns, join_type)
    
    return result_df


def left_salted_join(
    spark_ctx,
    left_df: DataFrame,
    right_df: DataFrame,
    join_columns: List[str],
    skew_threshold: int = 1000,
    salt_buckets: int = 100,
    salt_column_name: str = "salt_value",
    release_point: str = None,
    broadcast_threshold: int = 100000
) -> DataFrame:
    """
    Perform join with left-side salting for skew handling.

    This function analyzes the left DataFrame for skewed keys, applies salting
    to the left side only, replicates the right DataFrame accordingly, performs
    the inner join, and cleans up salt columns from the result.

    Args:
        spark_ctx: Spark context with persist_manager for lifecycle management
        left_df: Left DataFrame (will be analyzed for skew)
        right_df: Right DataFrame (will be replicated if skew detected)
        join_columns: Columns to join on
        skew_threshold: Number of rows per key to consider as skewed (default: 1000)
        salt_buckets: Maximum number of salt buckets to create (default: 100)
        salt_column_name: Name for the salt column (default: "salt_value")
        release_point: Release point for persist_manager (required)
        broadcast_threshold: Threshold for broadcasting skew info (default: 100000)

    Returns:
        Joined DataFrame without salt columns

    Example:
        >>> result = left_salted_join(
        ...     spark_ctx=spark_ctx,
        ...     left_df=parent_df,
        ...     right_df=edge_df,
        ...     join_columns=['node_id', 'node_type', 'date'],
        ...     release_point='hop_1_done'
        ... )
    """
    logger.info(f"Performing left-salted inner join on columns: {join_columns}")

    # Step 1: Salt left DataFrame to handle skew
    left_salted, left_skewed_keys = salt_skewed_keys(
        spark_ctx=spark_ctx,
        df=left_df,
        join_columns=join_columns,
        skew_threshold=skew_threshold,
        salt_buckets=salt_buckets,
        salt_column_name=salt_column_name,
        release_point=release_point,
        broadcast_threshold=broadcast_threshold
    )

    salt_columns = []

    # Step 2: Replicate right DataFrame if skew detected
    if left_skewed_keys is not None:
        logger.info("  Skew detected in left DataFrame, replicating right DataFrame")
        right_replicated = replicate_for_salted_join(
            spark_ctx=spark_ctx,
            df=right_df,
            join_columns=join_columns,
            target_key_salts=left_skewed_keys,
            target_salt_column=salt_column_name,
            release_point=release_point,
            broadcast_threshold=broadcast_threshold
        )
        salt_columns.append(salt_column_name)
    else:
        logger.info("  No skew detected, performing regular join")
        right_replicated = right_df

    # Step 3: Perform join with salt columns
    result = left_salted.join(
        right_replicated,
        on=join_columns + salt_columns,
        how="inner"
    )

    # Step 4: Clean up salt columns from result
    if salt_columns:
        result = result.drop(*salt_columns)

    logger.info(f"Left-salted join completed")

    return result
