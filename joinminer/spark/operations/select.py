from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr
from pyspark.sql.types import NumericType
import re
import logging

logger = logging.getLogger(__name__)


def select(df, select, distinct=False):
    """
    Select columns from a PySpark DataFrame with aliasing support.
    
    Args:
        df: PySpark DataFrame
        select: Select configuration, can be:
            - str: Single column name
            - list: List of column specifications, where each item can be:
                - str: Column name
                - dict: Column specification with optional alias
        distinct: bool, whether to apply distinct() to the result (default: False)
            
    Returns:
        DataFrame with selected columns
    """
    # Normalize input to list
    if isinstance(select, str):
        select = [select]
    elif not isinstance(select, list):
        raise ValueError(f"Select must be str or list, got {type(select)}")
    
    # Parse select specifications
    select_expressions = []
    
    for item in select:
        expressions = _parse_select_item(df, item)
        select_expressions.extend(expressions)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expressions = []
    for expr in select_expressions:
        expr_str = str(expr)
        if expr_str not in seen:
            seen.add(expr_str)
            unique_expressions.append(expr)
    
    logger.info(f"Selecting {len(unique_expressions)} columns")
    
    # Select columns
    result_df = df.select(*unique_expressions)
    
    # Apply distinct if requested
    if distinct:
        logger.info("Applying distinct to result")
        result_df = result_df.distinct()
    
    return result_df


def _parse_select_item(df, item):
    """Parse a single select item and return list of column expressions."""
    
    if isinstance(item, str):
        # Simple column name
        return [col(item)]
    
    elif isinstance(item, dict):
        # Get the column specification
        column_spec = item.get('column')
        alias = item.get('alias')
        
        # Handle different column specifications
        if isinstance(column_spec, str):
            # Single column
            if alias:
                return [col(column_spec).alias(alias)]
            else:
                return [col(column_spec)]
        
        elif isinstance(column_spec, list):
            # List of columns
            if alias:
                raise ValueError("Cannot apply single alias to multiple columns")
            return [col(c) for c in column_spec]
        
        # Pattern-based selection
        elif item.get('pattern'):
            pattern = item['pattern']
            regex_mode = item.get('regex', False)
            
            if regex_mode:
                compiled = re.compile(pattern)
                matching_cols = [c for c in df.columns if compiled.match(c)]
            else:
                matching_cols = [c for c in df.columns if pattern in c]
            
            if alias:
                raise ValueError("Cannot apply single alias to pattern-matched columns")
            return [col(c) for c in matching_cols]
        
        # All columns except specified
        elif item.get('exclude'):
            exclude = item['exclude']
            if isinstance(exclude, str):
                exclude = [exclude]
            
            selected_cols = [c for c in df.columns if c not in exclude]
            
            if alias:
                raise ValueError("Cannot apply single alias to multiple columns")
            return [col(c) for c in selected_cols]
        
        # Expression
        elif item.get('expr'):
            expr_str = item['expr']
            if alias:
                return [expr(expr_str).alias(alias)]
            else:
                return [expr(expr_str)]
        
        else:
            raise ValueError(f"Invalid select item configuration: {item}")
    
    else:
        raise ValueError(f"Select item must be str or dict, got {type(item)}")
