"""
DataFrame size estimation utilities.

Provides functions to estimate the storage size of Spark DataFrames
based on schema and row counts. This is used primarily for optimizing
file splitting in write operations.
"""

import logging
from typing import Dict, Optional, Union
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

# Size estimates in bytes for common data types
TYPE_SIZES = {
    "int": 4,
    "bigint": 8,
    "double": 8,
    "float": 4,
    "date": 4,
    "boolean": 1,
    "string": 50,
    "default": 300,  # Fallback for unknown types
}


def estimate_vector_size(element_count: int, element_bytes: int = 4) -> int:
    """
    Estimate byte size for a vector column.

    Args:
        element_count: Number of elements in the vector
        element_bytes: Bytes per element (default: 4 for float32)

    Returns:
        Estimated byte size per row
    """
    return element_count * element_bytes


def fill_column_sizes(
    df: DataFrame,
    col_sizes: Optional[Dict[str, Union[int, str]]] = None
) -> Dict[str, Union[int, str]]:
    """
    Fill column sizes for all columns in DataFrame.

    For columns not specified in col_sizes, fills with type-based defaults
    from TYPE_SIZES.

    Args:
        df: Input DataFrame
        col_sizes: User-specified sizes. Can be:
                   - int: fixed byte size per row
                   - str: Spark SQL expression for variable-length columns
                          (e.g., "(hop_k + 1) * 8" for array columns)

    Returns:
        Dict mapping column names to size (int) or expression (str)
        Example: {
            "id": 8,
            "name": 50,
            "node_ids": "(hop_k + 1) * 8"
        }
    """
    if col_sizes is None:
        col_sizes = {}

    result = {}

    for field in df.schema.fields:
        col_name = field.name
        col_type = field.dataType.simpleString()

        if col_name in col_sizes:
            # User specified: keep as-is (int or str expression)
            result[col_name] = col_sizes[col_name]
        elif col_type in TYPE_SIZES:
            # Type default
            result[col_name] = TYPE_SIZES[col_type]
        else:
            # Fallback
            result[col_name] = TYPE_SIZES["default"]
            logger.warning(
                f"Unknown type {col_type} for column {col_name}, using default size"
            )

    return result


__all__ = [
    'TYPE_SIZES',
    'estimate_vector_size',
    'fill_column_sizes',
]
