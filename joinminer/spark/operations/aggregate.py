from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import logging

logger = logging.getLogger(__name__)


def aggregate(df, group_columns, aggregation_expressions, pivot=None):
    """
    Aggregate a PySpark DataFrame with flexible configuration and optional pivot.

    Args:
        df: PySpark DataFrame
        group_columns: Columns to group by (str or list)
        aggregation_expressions: List of aggregation configurations, each containing:
            - functions: List of aggregation functions to apply
            - columns: List of column names (must be explicit column names, not config specs)
        pivot: Optional pivot configuration with:
            - column: Column to pivot on
            - values: Optional list of values to pivot

    Returns:
        Aggregated DataFrame

    Note:
        The 'columns' field in aggregation_expressions must be a list of explicit column names.
        Column parsing from config specs should be done at the element layer before calling this function.
    """
    # Normalize group_columns to list
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    elif not isinstance(group_columns, list):
        raise ValueError(f"group_columns must be str or list, got {type(group_columns)}")

    # Map of function names to PySpark functions
    func_map = {
        'count': F.count,
        'mean': F.mean,
        'avg': F.mean,
        'sum': F.sum,
        'max': F.max,
        'min': F.min,
        'first': F.first,
        'last': F.last,
        'stddev': F.stddev,
        'variance': F.variance,
    }

    # Build aggregation expressions
    agg_expressions = []
    agg_aliases = []

    for agg_spec in aggregation_expressions:
        if not isinstance(agg_spec, dict):
            raise ValueError("Each aggregation expression must be a dict")

        functions = agg_spec.get('functions', [])
        columns = agg_spec.get('columns', [])

        # Columns must be explicit column names (list of strings)
        if not isinstance(columns, list):
            raise ValueError(f"'columns' must be a list of column names, got {type(columns)}")
        
        # Create aggregation expressions
        for func_name in functions:
            if func_name not in func_map:
                raise ValueError(
                    f"Unsupported aggregation function: {func_name}. "
                    f"Supported functions: {list(func_map.keys())}"
                )
                                
            for col_name in columns:
                # Get the aggregation function
                agg_func = func_map[func_name]
                
                # Create expression with alias
                alias = f"{func_name}_{col_name}"
                expr = agg_func(col_name).alias(alias)

                # Store expression and alias
                agg_expressions.append(expr)
                agg_aliases.append(alias)
                
    if not agg_expressions:
        # No aggregation expressions - perform distinct on group columns instead
        logger.info(f"No aggregation expressions found. Performing distinct on group columns: {group_columns}")
        result_df = df.select(*group_columns).distinct()
        return result_df

    logger.info(f"Aggregating with {len(agg_expressions)} expressions, grouped by {group_columns}")
    
    # Handle pivot if configured
    if pivot:
        pivot_column = pivot['column']
        pivot_values = pivot['values']
        
        # Create pivot operation
        grouped_df = df.groupBy(group_columns)
        pivoted_df = grouped_df.pivot(pivot_column, pivot_values)
        
        # Apply aggregations
        result_df = pivoted_df.agg(*agg_expressions)

        # Rename columns to ensure consistent format: aggregation_alias_pivot_value
        # This is needed because Spark's default pivot format varies based on number of aggregations
        select_exprs = []

        # Keep group columns as is
        for col_name in group_columns:
            select_exprs.append(F.col(col_name))

        # Rename pivot columns in logical order: alias_1_window_0, alias_1_window_1, ..., alias_2_window_0, ...
        # This ensures features are grouped by aggregation function first, then by window
        for alias in agg_aliases:
            for pivot_value in pivot_values:
                # Spark pivot creates columns in format:
                # - Single agg: "pivot_value"
                # - Multi agg: "pivot_value_alias"
                if len(agg_expressions) == 1:
                    spark_col_name = str(pivot_value)
                else:
                    spark_col_name = f"{pivot_value}_{alias}"

                # Rename to our standard format: alias_pivot_value
                new_col_name = f"{alias}_{pivot_value}"
                select_exprs.append(F.col(spark_col_name).alias(new_col_name))

        result_df = result_df.select(*select_exprs)
        
    else:
        # Standard aggregation without pivot
        result_df = df.groupBy(group_columns).agg(*agg_expressions)

    logger.info(f"Aggregation completed. Result has {len(result_df.columns)} columns")
    
    return result_df
    