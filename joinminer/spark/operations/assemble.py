from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorSizeHint, VectorAssembler
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import NumericType
import logging

logger = logging.getLogger(__name__)


def assemble(df, input_columns, output_column, stable_columns=None, vector_columns_size=None):
    """
    Assemble multiple columns into a single vector column using VectorAssembler.

    Args:
        df: PySpark DataFrame
        input_columns: List of column names to assemble (must be explicit column names, not config specs)
        output_column: Name of the output vector column
        stable_columns: Columns to keep in addition to the vector column (str or list)
        vector_columns_size: Dict mapping vector column names to their sizes (int)
            Example: {'feat_vector_0': 50, 'feat_vector_1': 30}
            Only required when assembling existing vector columns.

    Returns:
        DataFrame: DataFrame with vector column (only stable_columns + output_column)

    Note:
        The 'input_columns' parameter must be a list of explicit column names.
        Column parsing from config specs should be done at the element layer before calling this function.
    """
    # Normalize stable_columns to list
    if stable_columns is None:
        stable_columns = []
    elif isinstance(stable_columns, str):
        stable_columns = [stable_columns]
    elif not isinstance(stable_columns, list):
        raise ValueError(f"stable_columns must be str, list, or None, got {type(stable_columns)}")

    # Input columns must be explicit column names (list of strings)
    if not isinstance(input_columns, list):
        raise ValueError(f"'input_columns' must be a list of column names, got {type(input_columns)}")

    columns_to_assemble = input_columns

    if not columns_to_assemble:
        raise ValueError("No columns were selected for assembly")

    # Get schema info for validation
    schema_dict = {f.name: f.dataType for f in df.schema.fields}

    # Validate and prepare vector columns
    vector_columns_size = vector_columns_size or {}
    total_length = 0

    for col_name in columns_to_assemble:
        dtype = schema_dict[col_name]

        if isinstance(dtype, VectorUDT):
            # Vector column - require explicit size
            if col_name not in vector_columns_size:
                raise ValueError(
                    f"Vector column '{col_name}' requires size specification in vector_columns_size. "
                    f"Example: vector_columns_size={{'{col_name}': 50}}"
                )

            size = vector_columns_size[col_name]
            total_length += size

            # Apply VectorSizeHint to validate vector size
            hint = VectorSizeHint(
                inputCol=col_name,
                size=size,
                handleInvalid="error"
            )
            df = hint.transform(df)

        elif isinstance(dtype, NumericType):
            # Numeric column - counts as 1 feature
            total_length += 1
        else:
            raise ValueError(
                f"Column '{col_name}' has type {dtype}, but only numeric and vector types are supported"
            )
    
    logger.info(
        f"Assembling {len(columns_to_assemble)} columns "
        f"({total_length} total features) into vector '{output_column}'"
    )

    # Create VectorAssembler
    assembler = VectorAssembler(
        inputCols=columns_to_assemble,
        outputCol=output_column
    )

    # Apply transformation
    assembled_df = assembler.transform(df)

    # Select only stable columns and the new vector column
    select_columns = stable_columns + [output_column]
    result_df = assembled_df.select(*select_columns)

    return result_df
