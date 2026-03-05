"""DataFrame display utilities."""

import logging
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def show_table(
    df: DataFrame,
    n: int = 5,
    truncate: int = 30,
    cols_per_group: int = 10
) -> None:
    """
    Show first n rows of DataFrame through logger, grouped by columns.

    Args:
        df: DataFrame to display
        n: Number of rows to show
        truncate: Maximum width for each column
        cols_per_group: Number of columns per group (default 10)
    """
    columns = df.columns
    num_groups = (len(columns) + cols_per_group - 1) // cols_per_group

    for group_idx in range(num_groups):
        start = group_idx * cols_per_group
        end = min(start + cols_per_group, len(columns))
        group_cols = columns[start:end]

        logger.info(f"Columns {start+1}-{end} of {len(columns)}:")
        df_group = df.select(*group_cols)
        show_str = df_group._jdf.showString(n, truncate, False)
        for line in show_str.strip().split('\n'):
            logger.info(line)
        logger.info("")


__all__ = ['show_table']
