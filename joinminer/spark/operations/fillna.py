from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, udf
from pyspark.sql.types import NumericType
from pyspark.ml.linalg import SparseVector, VectorUDT
import logging

logger = logging.getLogger(__name__)


def fillna(df, columns, vector_column_length={}, add_mark=True, fill_value=0):
    """
    Fill null values in specified columns of a PySpark DataFrame.

    Args:
        df: PySpark DataFrame
        columns: List of column names to fill (must be explicit column names, not config specs)
        vector_column_length: Dict mapping vector column names to vector length
        add_mark: Whether to add null mark columns (0 for null, 1 for not null)
        fill_value: Value to fill for non-vector columns (default: 0)

    Returns:
        DataFrame with null values filled

    Note:
        The 'columns' parameter must be a list of explicit column names.
        Column parsing from config specs should be done at the element layer before calling this function.
    """
    # Columns must be explicit column names (list of strings)
    if not isinstance(columns, list):
        raise ValueError(f"'columns' must be a list of column names, got {type(columns)}")

    columns_to_fill = columns
    columns_to_fill_display = columns_to_fill if len(columns_to_fill) <= 5 else columns_to_fill[:5] + ['...']
    logger.info(f"Filling null values in {len(columns_to_fill)} columns: {columns_to_fill_display}")
    
    # Get schema info for validation
    schema_dict = {f.name: f.dataType for f in df.schema.fields}
    
    # Validate columns and separate by type
    numeric_columns = []
    vector_columns = []
    
    for col_name in columns_to_fill:        
        dtype = schema_dict[col_name]
        
        if isinstance(dtype, VectorUDT):
            # Vector column - check if source is provided
            if col_name not in vector_column_length:
                raise ValueError(
                    f"Vector column '{col_name}' requires vector length in vector_column_length"
                )
            vector_columns.append(col_name)
        elif isinstance(dtype, NumericType):
            # Numeric column
            numeric_columns.append(col_name)
        else:
            raise ValueError(
                f"Column '{col_name}' has type {dtype}, but only numeric and vector types are supported"
            )
    
    # Prepare all transformations
    fill_dict = {}  # For numeric columns fillna
    new_columns = []  # For mark columns and vector fills
    
    # Prepare numeric column fills
    if numeric_columns:
        fill_dict = {col_name: fill_value for col_name in numeric_columns}
        
        if add_mark:
            # Add mark columns for numeric columns
            for col_name in numeric_columns:
                mark_col_name = f"null_mark_of_{col_name}"
                new_columns.append((
                    mark_col_name,
                    when(isnull(col(col_name)) | isnan(col(col_name)), 0).otherwise(1)
                ))
    
    # Prepare vector column transformations
    for col_name in vector_columns:
        vector_length = vector_column_length[col_name]
        
        # Create UDF for this specific vector
        fill_vector_udf = udf(
            lambda v: v if v is not None else SparseVector(vector_length, []), 
            VectorUDT()
        )
        
        # Add vector fill transformation
        new_columns.append((col_name, fill_vector_udf(col(col_name))))
        
        if add_mark:
            # Add mark column for vector
            mark_col_name = f"null_mark_of_{col_name}"
            new_columns.append((
                mark_col_name,
                when(isnull(col(col_name)), 0).otherwise(1)
            ))
    
    # Apply all transformations efficiently
    # First, add all new/modified columns in one select
    if new_columns:
        # Build select expression preserving all original columns
        select_expr = []
        modified_cols = {name for name, _ in new_columns}
        
        # Keep all original columns that aren't being modified
        for col_name in df.columns:
            if col_name not in modified_cols:
                select_expr.append(col(col_name))
        
        # Add all new/modified columns
        for col_name, expr in new_columns:
            select_expr.append(expr.alias(col_name))
        
        df = df.select(*select_expr)
    
    # Then apply numeric fillna if needed
    if fill_dict:
        df = df.fillna(fill_dict)
    
    logger.info(
        f"Filled {len(numeric_columns)} numeric columns and {len(vector_columns)} vector columns"
    )
    
    return df

