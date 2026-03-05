"""
Abstract base class for context mapping strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame

from joinminer.spark import SparkContext


class ContextMapper(ABC):
    """
    Abstract base class for context mapping strategies.

    Context mappers handle reading context tables and performing ID mapping
    to prepare data for aggregation. Different strategies (static, time_window)
    are implemented as subclasses.

    The mapper is responsible for:
    1. Reading the context table's _context_info.json metadata
    2. Reading and filtering the context table data
    3. Applying ID mapping (static replication or time window mapping)
    4. Returning unified mapping_info format for downstream processing
    """

    @classmethod
    def get_metadata_fields(cls) -> List[str]:
        """
        Return the list of configuration fields needed for metadata generation.

        This method declares which fields from the mapping_config should be
        included in the metadata. Subclasses should override to add their
        specific fields.

        Returns:
            List of field names (e.g., ['type', 'id_columns'])

        Example:
            StaticContextMapper.get_metadata_fields()
            => ['type', 'id_columns', 'context_table', 'dir']

            TimeWindowContextMapper.get_metadata_fields()
            => ['type', 'id_columns', 'context_table', 'dir',
                'time_column', 'time_format', 'windows']
        """
        # Base fields required by all mappers
        return ['type', 'id_columns', 'context_table', 'dir']

    @classmethod
    def normalize_config(cls, mapping_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and standardize configuration fields for metadata.

        This method creates a normalized configuration dict containing only
        the fields declared in get_metadata_fields(). It handles:
        - Missing optional fields (skips them)
        - Type-specific transformations (overridden by subclasses if needed)

        Args:
            mapping_config: Raw configuration dict from YAML

        Returns:
            Normalized configuration dict for metadata

        Example:
            mapping_config = {
                'type': 'time_window',
                'context_table': 'works_author',
                'id_columns': ['author_id'],
                'time_column': 'date',
                'time_format': 'yyyy-MM-dd',
                'windows': [{'offset': -600, 'length': 12}, ...],
                'extra_field': 'ignored'
            }

            TimeWindowContextMapper.normalize_config(mapping_config)
            => {
                'type': 'time_window',
                'context_table': 'works_author',
                'dir': '...',  # from mapping_config['dir']
                'id_columns': ['author_id'],
                'time_column': 'date',
                'time_format': 'yyyy-MM-dd',
                'windows': {'offset': -600, 'length': 12}  # First window only
            }
        """
        metadata_fields = cls.get_metadata_fields()
        normalized = {}

        for field in metadata_fields:
            if field in mapping_config:
                normalized[field] = mapping_config[field]

        return normalized

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate context_mapping configuration.

        Args:
            config: The context_mapping configuration dict

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def read_context_table(
        self,
        spark_ctx: SparkContext,
        config: Dict[str, Any],
        element_partition_columns: List[str],
        element_partition_instances: List[List[str]]
    ) -> Tuple[DataFrame, List[Dict[str, Any]], List[str]]:
        """
        Read context table and apply ID mapping.

        This is the main method that orchestrates the context mapping process:
        1. Read context table _context_info.json
        2. Read context table data (with optional partition pruning)
        3. Select and rename ID columns
        4. Apply ID mapping strategy (static/time_window)
        5. Return mapping information in unified format

        Args:
            spark_ctx: SparkContext instance providing Spark session and FileIO
            config: The context_mapping configuration dict with fields:
                   - type: 'static' or 'time_window'
                   - context_table: table name
                   - id_columns: list of column names or dicts with original/alias
                   - (type-specific fields)
            element_partition_columns: Element table's partition columns (e.g., ['date'])
            element_partition_instances: Element partition instances to build
                                        (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            Tuple of three elements:
            1. DataFrame: Processed context data with:
                * ID columns (renamed if aliases specified)
                * Feature columns
                * Partition columns matching element table
                * Optional _mapping_id column (for time_window)

            2. mapping_info: List of mapping configurations in unified format:
                For static: [{'id': 'static', 'config': {}}]
                For time_window: [
                    {'id': 'window_0', 'config': {'offset': -600, 'length': 12, ...}},
                    {'id': 'window_1', 'config': {'offset': -588, 'length': 12, ...}},
                    ...
                ]

            3. original_features: List of original feature column names from context table
                (after renaming, but before any aggregation)

        Raises:
            ValueError: If configuration or data is invalid
        """
        pass

    def _read_context_metadata(
        self,
        fileio,
        context_table_dir: str
    ) -> Dict[str, Any]:
        """
        Read context table's _context_info.json.

        Args:
            fileio: FileIO instance
            context_table_dir: Full path to context table directory

        Returns:
            Dict containing partition_columns and feature_columns

        Raises:
            FileNotFoundError: If _context_info.json doesn't exist
        """
        metadata_path = f"{context_table_dir}/_context_info.json"
        return fileio.read_json(metadata_path)

    def _select_and_rename_columns(
        self,
        df: DataFrame,
        id_columns_config: List[Any]
    ) -> Tuple[DataFrame, List[str]]:
        """
        Select ID columns and apply aliases if specified.

        This method handles column name conflicts by:
        1. Identifying which columns to keep and which to drop
        2. Dropping conflicting columns before renaming
        3. Only selecting the columns specified in id_columns_config

        Args:
            df: Input DataFrame
            id_columns_config: List of column specs, each can be:
                              - str: column name (no rename)
                              - dict: {'original': 'old_name', 'alias': 'new_name'}

        Returns:
            Tuple of:
            - DataFrame: With columns renamed and conflicts resolved
            - List[str]: Final ID column names (after renaming)

        Example:
            Input: id_columns_config = ['author_id', {'original': 'pub_date', 'alias': 'date'}]
            Output: df with 'pub_date' renamed to 'date', id_columns = ['author_id', 'date']

            Conflict handling:
            If df has columns [work_id, cited_work_id, date] and config is:
            [{'original': 'cited_work_id', 'alias': 'work_id'}, 'date']
            Then:
            1. Drop the original 'work_id' column (it conflicts with the alias)
            2. Rename 'cited_work_id' to 'work_id'
            3. Keep 'date'
            Result: df has [work_id, date] where work_id is from cited_work_id
        """
        # First pass: identify columns to keep and aliases
        columns_to_keep = []  # Original column names to keep
        columns_to_drop = set()  # Original column names to drop
        rename_map = {}  # {original_name: alias_name}
        final_id_columns = []  # Final column names after rename

        # Identify all alias targets
        alias_targets = set()
        for col_spec in id_columns_config:
            if isinstance(col_spec, dict):
                alias_targets.add(col_spec['alias'])

        # Process each column spec
        for col_spec in id_columns_config:
            if isinstance(col_spec, str):
                # Simple column name, no rename
                # But check if this column will be overwritten by an alias
                if col_spec not in alias_targets:
                    columns_to_keep.append(col_spec)
                else:
                    # This column will be overwritten by an alias, drop it
                    columns_to_drop.add(col_spec)
                final_id_columns.append(col_spec)
            elif isinstance(col_spec, dict):
                # Rename: original -> alias
                original = col_spec['original']
                alias = col_spec['alias']
                columns_to_keep.append(original)
                rename_map[original] = alias
                final_id_columns.append(alias)

                # If the alias name already exists as a column, mark it for dropping
                if alias in df.columns and alias != original:
                    columns_to_drop.add(alias)
            else:
                raise ValueError(f"Invalid id_columns spec: {col_spec}")

        # Drop conflicting columns first
        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Apply renames
        for original, alias in rename_map.items():
            df = df.withColumnRenamed(original, alias)

        return df, final_id_columns
