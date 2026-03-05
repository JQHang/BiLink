from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, lit, rand, floor, when, ceil,
    explode, sequence, least, broadcast
)
from typing import Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_num_salts_col(salt_column_name: str) -> str:
    """Get the num_salts column name for a given salt column."""
    return f"{salt_column_name}_num"


def salt_skewed_keys(
    spark_ctx,
    df: DataFrame,
    join_columns: List[str],
    skew_threshold: int,
    salt_buckets: int,
    salt_column_name: str,
    release_point: str,
    broadcast_threshold: int = 100000
) -> Tuple[DataFrame, Optional[DataFrame]]:
    """
    Identify skewed keys in a DataFrame and add salt values to distribute them.

    This function:
    1. Counts occurrences of each key combination
    2. Identifies keys exceeding the skew threshold
    3. Adds salt values to the original DataFrame
    4. Returns the salted DataFrame and skew information

    Args:
        spark_ctx: Spark context with persist_manager for lifecycle management
        df: Input DataFrame to analyze and salt
        join_columns: Column(s) to analyze for skew
        skew_threshold: Number of records per key to consider as skewed (must be > 0)
        salt_buckets: Maximum number of salt buckets to create (0 = unlimited, >0 = capped)
        salt_column_name: Name for the salt column (e.g., "salt_value", "left_salt")
        release_point: Release point name for persist_manager (required)
        broadcast_threshold: Threshold for broadcasting the skew info

    Returns:
        Tuple containing:
        - DataFrame with salt column added (0 for non-skewed keys)
        - Skewed keys DataFrame (managed by persist_manager) with num_salts info, None if no skew

    Example:
        >>> salted_df, skewed_keys = salt_skewed_keys(
        ...     spark_ctx=spark_ctx,
        ...     df=left_df,
        ...     join_columns=["user_id"],
        ...     skew_threshold=10000,
        ...     salt_buckets=100,
        ...     salt_column_name="left_salt",
        ...     release_point="stage_done"
        ... )
    """
    # Validate parameters
    assert skew_threshold > 0, f"skew_threshold must be > 0, got {skew_threshold}"
    assert salt_buckets >= 0, f"salt_buckets must be >= 0, got {salt_buckets}"

    logger.info(f"Identifying skewed keys and adding {salt_column_name} to DataFrame")

    # Persist input df since it will be used multiple times
    spark_ctx.persist_manager.persist(
        df,
        release_point=release_point,
        name=f'input_df_{salt_column_name}'
    )

    # Count occurrences of each key combination
    key_counts = (
        df
        .select(*join_columns)  # Reduce shuffle by selecting only needed columns
        .dropna(subset=join_columns)  # Filter out rows with null values in join columns
        .groupBy(*join_columns)
        .agg(count("*").alias("key_count"))
        .filter(col("key_count") > skew_threshold)
    )
    
    # Calculate required salt buckets for each skewed key
    num_salts_col = get_num_salts_col(salt_column_name)
    skewed_key_salts = key_counts.withColumn(
        num_salts_col,
        ceil(col("key_count") / skew_threshold)
    ).drop("key_count")
    
    # Cache skewed keys info for reuse
    spark_ctx.persist_manager.persist(
        skewed_key_salts,
        release_point=release_point,
        name=f'skewed_keys_{salt_column_name}'
    )
    skewed_keys_count = skewed_key_salts.count()
    
    logger.info(f"Found {skewed_keys_count:,} skewed key combination(s) for {salt_column_name}")
    
    if skewed_keys_count == 0:
        # No skew detected, return original DataFrame
        logger.info(f"No skewed keys found, skipping salting for {salt_column_name}")
        return df, None
    else:
        sample_data = skewed_key_salts.limit(5).collect()
        sample_str = "\n".join([f"  {row.asDict()}" for row in sample_data])
        logger.info(f"Sample skewed keys:\n{sample_str}")
    
    # Join with skew info to identify which rows need salting
    df_with_skew_info = df.join(
        broadcast(skewed_key_salts) if skewed_keys_count < broadcast_threshold else skewed_key_salts,
        join_columns,
        "left"
    )
    
    # Add salt values: random integer [0, num_salts) for skewed keys, 0 for non-skewed
    salted_df = (
        df_with_skew_info
        .withColumn(
            salt_column_name,
            when(
                col(num_salts_col).isNotNull(),
                floor(rand() * col(num_salts_col)).cast("int")
            ).otherwise(lit(0))
        )
        .drop(num_salts_col)
    )

    # Apply bucket limit and cap num_salts if salt_buckets > 0 (unlimited mode when salt_buckets == 0)
    if salt_buckets > 0:
        salted_df = salted_df.filter(col(salt_column_name) < salt_buckets)
        skewed_key_salts = skewed_key_salts.withColumn(
            num_salts_col,
            least(col(num_salts_col), lit(salt_buckets))
        )

    return salted_df, skewed_key_salts


def replicate_for_salted_join(
    spark_ctx,
    df: DataFrame,
    join_columns: List[str],
    target_key_salts: DataFrame,
    target_salt_column: str,
    release_point: str,
    own_key_salts: Optional[DataFrame] = None,
    own_salt_column: Optional[str] = None,
    own_key_count: Optional[int] = None,
    broadcast_threshold: int = 100000
) -> DataFrame:
    """
    Replicate a DataFrame based on the target side's skew for salted join.

    This function creates multiple copies of records based on salt mapping,
    optionally considering the DataFrame's own salt distribution to avoid
    aggregating skewed data during replication.

    Args:
        spark_ctx: Spark context with persist_manager for lifecycle management
        df: DataFrame to replicate (may already have its own salt column)
        join_columns: Column(s) to join on
        target_key_salts: Skewed keys info from the target DataFrame to replicate for
        target_salt_column: Name of the salt column to add (from target side)
        release_point: Release point name for persist_manager (required)
        own_key_salts: Optional skewed keys info if this DataFrame is also skewed
        own_salt_column: Optional name of this DataFrame's own salt column
        own_key_count: Optional count of own skewed keys
        broadcast_threshold: Threshold for broadcasting salt mapping

    Returns:
        Replicated DataFrame with target salt column added

    Example:
        >>> # Simple replication (one-sided skew)
        >>> replicated_df = replicate_for_salted_join(
        ...     spark_ctx=spark_ctx,
        ...     df=right_df,
        ...     join_columns=["user_id"],
        ...     target_key_salts=left_key_salts,
        ...     target_salt_column="left_salt",
        ...     release_point="stage_done"
        ... )

        >>> # Cross-replication (two-sided skew)
        >>> replicated_df = replicate_for_salted_join(
        ...     spark_ctx=spark_ctx,
        ...     df=right_with_salt,
        ...     join_columns=["user_id"],
        ...     target_key_salts=left_key_salts,
        ...     target_salt_column="left_salt",
        ...     own_key_salts=right_key_salts,
        ...     own_salt_column="right_salt",
        ...     own_key_count=100,
        ...     release_point="stage_done"
        ... )
    """
    
    logger.info(f"Replicating DataFrame for {target_salt_column}")
    
    # Generate salt mapping based on target skew
    target_num_salts_col = get_num_salts_col(target_salt_column)
    
    salt_mapping = (
        target_key_salts
        .withColumn(
            target_salt_column,
            explode(
                sequence(
                    lit(0),
                    col(target_num_salts_col) - 1
                )
            )
        )
        .drop(target_num_salts_col)
    )
    
    # If this DataFrame also has skew, need to consider its salt distribution
    # to avoid aggregating all records to single partition during replication
    if own_key_salts is not None:
        assert own_salt_column is not None, "Missing own_salt_column when own_key_salts is provided"
        assert own_key_count is not None, "Missing own_key_count when own_key_salts is provided"
        
        logger.info(f"Considering own salt distribution ({own_salt_column}) during replication")
        
        own_num_salts_col = get_num_salts_col(own_salt_column)

        assert set(own_key_salts.columns) == set(join_columns + [own_num_salts_col]), \
               f"Unexpected columns in own_key_salts: {own_key_salts.columns}"
        
        # Add own salt distribution to mapping for cross-product of salt values
        salt_mapping = (
            salt_mapping
            .join(
                broadcast(own_key_salts) if own_key_count < broadcast_threshold else own_key_salts,
                join_columns,
                "left"
            )
            .fillna(1, subset=[own_num_salts_col])
            .withColumn(
                own_salt_column,
                explode(
                    sequence(
                        lit(0),
                        col(own_num_salts_col) - 1
                    )
                )
            )
            .drop(own_num_salts_col)
        )
        
        # Join using both key and own salt column for partition-aware replication
        join_columns = join_columns + [own_salt_column]
    
    # Persist salt mapping for reuse
    spark_ctx.persist_manager.persist(
        salt_mapping,
        release_point=release_point,
        name=f'salt_mapping_{target_salt_column}'
    )
    salt_mapping_count = salt_mapping.count()
    
    logger.info(f"Generated {salt_mapping_count:,} salt mapping(s) for {target_salt_column}")

    sample_data = salt_mapping.limit(5).collect()
    sample_str = "\n".join([f"  {row.asDict()}" for row in sample_data])
    logger.info(f"Sample salt mapping:\n{sample_str}")
    
    # Apply salt mapping to DataFrame
    replicated_df = (
        df
        .join(
            broadcast(salt_mapping) if salt_mapping_count < broadcast_threshold else salt_mapping,
            join_columns,
            "left"
        )
        .fillna(0, subset=[target_salt_column])  # Non-skewed keys get target_salt=0
    )

    return replicated_df
    