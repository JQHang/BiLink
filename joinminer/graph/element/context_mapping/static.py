"""
Static context mapper for profile/metadata tables.
"""

from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from joinminer.spark import SparkContext
from .base import ContextMapper
from joinminer.spark.io import read_table


class StaticContextMapper(ContextMapper):
    """
    Maps static context tables to element partitions.

    Static mapping is used for profile or metadata tables where each entity
    (e.g., author) has features that don't change based on the partition
    (e.g., h_index on a specific date).

    Process:
    1. Read context table's _metadata.json
    2. Read context table (with optional partition_mapping for pruning)
    3. Select and rename ID columns
    4. Cross-join with element partitions to replicate rows
    5. Initialize MetadataTracker with id_mapping='static'

    Example configuration:
        context_mapping:
          type: static
          context_table: author_profiles
          id_columns: [author_id]
          partition_mapping:  # Optional
            date: date        # element_partition -> context_partition
    """

    @classmethod
    def get_metadata_fields(cls) -> List[str]:
        """
        Return metadata fields for static mapper.

        Static mapper uses only base fields, no additional type-specific fields.

        Returns:
            ['type', 'id_columns', 'context_table', 'dir']
        """
        # Static mapper doesn't need additional fields beyond base
        return super().get_metadata_fields()

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate static context_mapping configuration.

        Required fields:
        - type: must be 'static'
        - context_table: context table name
        - id_columns: list of ID column specs

        Optional fields:
        - partition_mapping: dict mapping element partition cols to context partition cols
        - dir: override default context table directory

        Raises:
            ValueError: If required fields missing or invalid
        """
        if config.get('type') != 'static':
            raise ValueError(f"Expected type='static', got '{config.get('type')}'")

        if 'context_table' not in config:
            raise ValueError("Missing required field: context_table")

        if 'id_columns' not in config:
            raise ValueError("Missing required field: id_columns")

        if not isinstance(config['id_columns'], list):
            raise ValueError("id_columns must be a list")

    def read_context_table(
        self,
        spark_ctx,
        config: Dict[str, Any],
        element_partition_columns: List[str],
        element_partition_instances: List[List[str]]
    ) -> Tuple[DataFrame, List[Dict[str, Any]], List[str]]:
        """
        Read static context table and replicate to element partitions.

        Args:
            spark_ctx: SparkRunner instance
            config: context_mapping configuration
            element_partition_columns: Element partition columns (e.g., ['date'])
            element_partition_instances: Element partition instances (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            Tuple of (DataFrame, mapping_info, original_features)
            - DataFrame: Context data replicated to all element partitions
            - mapping_info: [{'id': 'static', 'config': {}}]
            - original_features: List of feature column names
        """
        # Validate configuration
        self.validate_config(config)

        # Get context table path
        context_table_name = config['context_table']
        context_table_dir = config.get('dir')
        if context_table_dir is None:
            # Use default directory from spark_ctx or config
            # This would be passed from ElementBuilder
            raise ValueError("context_table directory must be specified")

        # Read context _context_info.json
        context_metadata = self._read_context_metadata(
            spark_ctx.fileio,
            context_table_dir
        )

        # Read context table with optional partition mapping
        context_df = self._read_with_partition_mapping(
            spark_ctx,
            context_table_dir,
            context_metadata.get('partition_columns', []),
            config.get('partition_mapping'),
            element_partition_columns,
            element_partition_instances
        )

        # Select and rename ID columns
        context_df, final_id_columns = self._select_and_rename_columns(
            context_df,
            config['id_columns']
        )

        # Get feature columns from context metadata (authoritative source)
        # This ensures only columns defined in _context_info.json are treated as features
        feature_columns_in_df = list(context_metadata['feature_columns'].keys())

        # Validate that expected feature columns exist in the DataFrame
        missing_features = set(feature_columns_in_df) - set(context_df.columns)
        if missing_features:
            raise ValueError(
                f"Expected feature columns {missing_features} not found in context table. "
                f"Available columns: {context_df.columns}"
            )

        # Select ID columns and feature columns only
        select_columns = final_id_columns + feature_columns_in_df
        context_df = context_df.select(*select_columns)

        # Replicate to element partitions
        replicated_df = self._replicate_to_partitions(
            context_df,
            element_partition_columns,
            element_partition_instances
        )

        # Create unified mapping_info format for static mapping
        mapping_info = [{'id': 'static', 'config': {}}]

        # Return original feature column names
        original_features = feature_columns_in_df

        return replicated_df, mapping_info, original_features

    def _read_with_partition_mapping(
        self,
        spark_ctx,
        context_table_dir: str,
        context_partition_columns: List[str],
        partition_mapping: Dict[str, str],
        element_partition_columns: List[str],
        element_partition_instances: List[List[str]]
    ) -> DataFrame:
        """
        Read context table with optional partition pruning.

        If partition_mapping is specified, only read context partitions that
        match the element partition values.

        Args:
            spark_ctx: SparkRunner instance
            context_table_dir: Path to context table
            context_partition_columns: Context table's partition columns
            partition_mapping: Dict mapping element_col -> context_col
            element_partition_columns: Element partition columns
            element_partition_instances: Element partition instances to build

        Returns:
            DataFrame with context table data
        """
        if partition_mapping:
            # Convert partition instances to dict for easier filtering
            # This builds a dict like {'date': ['2021-01-01', '2021-01-02']}
            element_partition_dict = {}
            for col_idx, col_name in enumerate(element_partition_columns):
                element_partition_dict[col_name] = [
                    instance[col_idx] for instance in element_partition_instances
                ]

            # Build partition filter
            context_partition_instances = []
            for element_col, context_col in partition_mapping.items():
                if element_col in element_partition_dict:
                    # Map element partition values to context partition values
                    # For simplicity, assume 1:1 mapping
                    context_partition_instances = [
                        [val] for val in element_partition_dict[element_col]
                    ]

            # Read with partition filter
            df = read_table(
                spark_ctx,
                context_table_dir,
                partition_columns=context_partition_columns,
                partition_instances=context_partition_instances
            )
        else:
            # Read entire table
            df = read_table(
                spark_ctx,
                context_table_dir,
                partition_columns=context_partition_columns
            )

        return df

    def _replicate_to_partitions(
        self,
        df: DataFrame,
        partition_columns: List[str],
        partition_instances: List[List[str]]
    ) -> DataFrame:
        """
        Replicate rows to all element partition values.

        For static mapping, each row in the context table needs to appear
        in each element partition (e.g., author profile copied to each date).

        Args:
            df: Context table DataFrame
            partition_columns: Element partition columns (e.g., ['date'])
            partition_instances: Element partition instances (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            DataFrame with partition columns added and rows replicated

        Example:
            Input: df with [author_id, h_index]
            partition_instances: [['2021-01-01'], ['2021-01-02']]

            Output: df with [author_id, h_index, date]
            - Each row duplicated for each date
        """
        if not partition_columns:
            return df

        # Create partition values DataFrame
        spark = df.sparkSession

        # Convert partition instances to tuples for createDataFrame
        # partition_instances is List[List[str]], convert to List[Tuple[str, ...]]
        partition_values = [tuple(instance) for instance in partition_instances]

        # Create DataFrame with partition values
        partition_df = spark.createDataFrame(
            partition_values,
            schema=partition_columns
        )

        # Cross join to replicate rows
        result_df = df.crossJoin(partition_df)

        return result_df
