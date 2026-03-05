"""
Aggregation pipeline execution for element table construction.

This module provides functions to execute multi-stage aggregation pipelines
with mapping column handling (drop/keep/pivot).
"""

from typing import Dict, List, Any
from pyspark.sql import DataFrame

from joinminer.spark.operations import aggregate


def execute_aggregation_pipeline(
    df: DataFrame,
    mapping_info: List[Dict[str, Any]],
    original_features: List[str],
    pipeline_config: List[Dict[str, Any]]
) -> DataFrame:
    """
    Execute multi-stage aggregation pipeline.

    Main entry point for executing the complete aggregation pipeline.
    Processes stages sequentially, handling mapping columns based on mapping_info.

    Args:
        df: Input DataFrame with ID columns, feature columns, and optional _mapping_id
        mapping_info: Mapping information from context_mapping stage (unified format)
        original_features: Original feature column names from context table
        pipeline_config: List of stage configurations from YAML config

    Returns:
        Aggregated DataFrame

    Example:
        aggregated_df = execute_aggregation_pipeline(
            df=context_df,
            mapping_info=[{'id': 'window_0', 'config': {...}}, ...],
            original_features=['citation_count', 'paper_count'],
            pipeline_config=[
                {'group_by': ['author_id', 'date'], 'mapping': 'pivot', 'functions': ['sum']}
            ]
        )
    """
    validate_pipeline_config(pipeline_config)

    current_df = df
    current_features = original_features

    for stage_config in pipeline_config:
        current_df, current_features = _execute_stage(
            current_df,
            current_features,
            mapping_info,
            stage_config
        )

    return current_df


def validate_pipeline_config(pipeline_config: List[Dict[str, Any]]) -> None:
    """
    Validate aggregation pipeline configuration.

    Args:
        pipeline_config: List of stage configurations

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(pipeline_config, list):
        raise ValueError("aggregation_pipeline must be a list")

    if not pipeline_config:
        raise ValueError("aggregation_pipeline cannot be empty")

    for i, stage_config in enumerate(pipeline_config):
        try:
            _validate_stage_config(stage_config)
        except ValueError as e:
            raise ValueError(f"Stage {i} invalid: {e}")


def _validate_stage_config(config: Dict[str, Any]) -> None:
    """Validate single aggregation stage configuration."""
    if 'group_by' not in config:
        raise ValueError("Missing required field: group_by")

    if not isinstance(config['group_by'], list):
        raise ValueError("group_by must be a list")

    if 'mapping' in config and config['mapping'] not in ['drop', 'keep', 'pivot']:
        raise ValueError("mapping must be 'drop', 'keep', or 'pivot'")

    if 'functions' in config and not isinstance(config['functions'], list):
        raise ValueError("functions must be a list")


def _execute_stage(
    df: DataFrame,
    current_features: List[str],
    mapping_info: List[Dict[str, Any]],
    stage_config: Dict[str, Any]
) -> tuple[DataFrame, List[str]]:
    """
    Execute a single aggregation stage.

    Handles mapping column behaviors:
    - drop: Aggregate across all mapping values, drop _mapping_id
    - keep: Include _mapping_id in group_by
    - pivot: Transform _mapping_id values into column suffixes

    Args:
        df: Input DataFrame
        current_features: Current feature column names
        mapping_info: Mapping information from context_mapping
        stage_config: Stage configuration

    Returns:
        Tuple of (aggregated DataFrame, updated feature list)
    """
    group_by = stage_config['group_by']
    mapping_behavior = stage_config.get('mapping', 'drop')
    functions = stage_config.get('functions', [])
    group_count = stage_config.get('group_count', False)

    # Identify feature columns to aggregate
    # These are the columns tracked as features (not ID or group columns)
    all_columns = set(df.columns)
    stable_columns = set(group_by) | {'_mapping_id'}
    candidate_columns = all_columns - stable_columns

    # Filter to only include columns that are either:
    # 1. In current_features list
    # 2. Aggregated columns (start with function name prefix)
    agg_function_prefixes = ('count_', 'sum_', 'mean_', 'max_', 'min_', 'first_', 'last_', 'stddev_', 'variance_')

    feature_columns = [
        col for col in candidate_columns
        if col in current_features or col.startswith(agg_function_prefixes)
    ]

    # Handle pivot separately
    if mapping_behavior == 'pivot':
        result_df, new_features = _execute_pivot_stage(
            df, feature_columns, mapping_info, group_by,
            functions, group_count
        )
    else:
        result_df, new_features = _execute_normal_stage(
            df, feature_columns, group_by, functions,
            group_count, mapping_behavior
        )

    return result_df, new_features


def _execute_normal_stage(
    df: DataFrame,
    feature_columns: List[str],
    group_by: List[str],
    functions: List[str],
    group_count: bool,
    mapping_behavior: str
) -> tuple[DataFrame, List[str]]:
    """
    Execute normal aggregation stage (drop or keep mapping).

    Args:
        df: Input DataFrame
        feature_columns: Feature column names to aggregate
        group_by: Grouping columns
        functions: Aggregation functions
        group_count: Whether to include group count
        mapping_behavior: 'drop' or 'keep'

    Returns:
        Tuple of (aggregated DataFrame, new feature names)
    """
    # Determine effective grouping columns
    effective_group_by = list(group_by)
    if mapping_behavior == 'keep' and '_mapping_id' in df.columns:
        effective_group_by.append('_mapping_id')

    # Build aggregations
    aggregations = []
    if group_count:
        aggregations.append({'functions': ['count'], 'columns': ['*']})
    for func in functions:
        aggregations.append({'functions': [func], 'columns': feature_columns})

    # Execute aggregation
    result_df = aggregate(
        df,
        group_columns=effective_group_by,
        aggregation_expressions=aggregations
    )

    # Generate new feature names (function_featurename format)
    new_features = []
    for func in functions:
        for feat in feature_columns:
            new_features.append(f'{func}_{feat}')

    if group_count:
        new_features.append('count')

    # Drop _mapping_id if mapping='drop'
    if mapping_behavior == 'drop' and '_mapping_id' in result_df.columns:
        result_df = result_df.drop('_mapping_id')

    return result_df, new_features


def _execute_pivot_stage(
    df: DataFrame,
    feature_columns: List[str],
    mapping_info: List[Dict[str, Any]],
    group_by: List[str],
    functions: List[str],
    group_count: bool
) -> tuple[DataFrame, List[str]]:
    """
    Execute aggregation stage with pivot on _mapping_id.

    Uses the native pivot functionality from spark/operations/aggregate.py
    for efficient pivoting without manual column manipulation.

    Args:
        df: Input DataFrame
        feature_columns: Feature column names to aggregate
        mapping_info: Mapping information from context_mapping
        group_by: Grouping columns (not including _mapping_id)
        functions: Aggregation functions
        group_count: Whether to include group count

    Returns:
        Tuple of (pivoted DataFrame, new feature names)
    """
    # Extract mapping IDs from mapping_info
    mapping_values = [m['id'] for m in mapping_info]

    if not mapping_values or '_mapping_id' not in df.columns:
        # No pivot needed, fall back to normal aggregation
        return _execute_normal_stage(
            df, feature_columns, group_by, functions,
            group_count, mapping_behavior='drop'
        )

    # Build aggregations
    aggregations = []
    if group_count:
        aggregations.append({'functions': ['count'], 'columns': ['*']})
    for func in functions:
        aggregations.append({'functions': [func], 'columns': feature_columns})

    # Execute aggregation with native pivot support
    result_df = aggregate(
        df,
        group_columns=group_by,
        aggregation_expressions=aggregations,
        pivot={'column': '_mapping_id', 'values': mapping_values}
    )

    # Generate new feature names after pivot
    # Format: function_feature_mappingid
    new_features = []
    for func in functions:
        for feat in feature_columns:
            for mapping_id in mapping_values:
                new_features.append(f'{func}_{feat}_{mapping_id}')

    # Add group_count features (one per mapping)
    if group_count:
        for mapping_id in mapping_values:
            new_features.append(f'count_{mapping_id}')

    return result_df, new_features
