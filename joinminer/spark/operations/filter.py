from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def filter(df, condition):
    """
    Filter a PySpark DataFrame based on structured conditions.

    Args:
        df: PySpark DataFrame to filter
        condition: Filter condition, can be:
            - str: Direct SQL expression like "age > 18"
            - dict: Structured condition with 'and'/'or' operators

    Returns:
        Filtered PySpark DataFrame
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Expected PySpark DataFrame, got {type(df)}")
    
    # Build the filter expression
    filter_expr = _build_filter_expression(condition)
    
    logger.info(f"Applying filter: {filter_expr}")
    
    # Apply the filter
    return df.filter(filter_expr)


def _build_filter_expression(condition):
    """
    Recursively build filter expression from structured condition.
    
    Args:
        condition: The condition to parse (str or dict)
        
    Returns:
        str: SQL filter expression
    """
    if isinstance(condition, str):
        # Base case: string is a direct SQL expression
        return f"({condition})"
    
    elif isinstance(condition, dict):
        if len(condition) != 1:
            raise ValueError(f"Condition dict should have exactly one key, got: {list(condition.keys())}")
        
        operator, operands = next(iter(condition.items()))
        
        if operator not in ['and', 'or']:
            raise ValueError(f"Unknown operator: {operator}. Supported: 'and', 'or'")
        
        if not isinstance(operands, list):
            raise ValueError(f"Operator '{operator}' expects a list of conditions")
        
        if len(operands) == 0:
            raise ValueError(f"Operator '{operator}' requires at least one condition")
        
        # Recursively build expressions for each operand
        sub_expressions = [_build_filter_expression(op) for op in operands]
        
        # Join with the operator
        if operator == 'and':
            return f"({' AND '.join(sub_expressions)})"
        else:  # or
            return f"({' OR '.join(sub_expressions)})"
    
    else:
        raise TypeError(f"Condition must be str or dict, got {type(condition)}")

def filter_by_columns(
    df: DataFrame,
    columns: List[str],
    values: List[List[Any]]
) -> DataFrame:
    """
    Filter a PySpark DataFrame based on column value combinations.
    
    Args:
        df: PySpark DataFrame to filter
        columns: List of column names to filter on
                 e.g., ['year', 'month', 'day']
        values: List of value combinations, each inner list corresponds to columns
                e.g., [[2024, 1, 15], [2024, 1, 16], [2024, 2, 1]]
                
    Returns:
        Filtered PySpark DataFrame
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Expected PySpark DataFrame, got {type(df)}")
    
    if not columns:
        raise ValueError("columns cannot be empty")
    
    if not values:
        logger.warning("No values provided, returning original DataFrame")
        return df
    
    # Validate that all value lists have correct length
    expected_len = len(columns)
    for i, val_list in enumerate(values):
        if len(val_list) != expected_len:
            raise ValueError(
                f"Values at index {i} has {len(val_list)} items, "
                f"expected {expected_len} to match columns"
            )
    
    # Check if all columns exist in DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    logger.info(f"Filtering DataFrame for {len(values)} value combinations on columns: {columns}")
    
    # Build filter conditions
    conditions = []
    for value_list in values:
        # Create AND condition for each value combination
        condition = reduce(
            lambda a, b: a & b,
            [F.col(col) == val for col, val in zip(columns, value_list)]
        )
        conditions.append(condition)
    
    # Combine all conditions with OR
    final_condition = reduce(lambda a, b: a | b, conditions)
    
    return df.filter(final_condition)
    