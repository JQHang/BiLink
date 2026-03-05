"""
Time window context mapper for temporal aggregation.
"""

from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from .base import ContextMapper
from joinminer.spark.io import read_table
from joinminer.spark.io.partition import parse_partition_spec
from joinminer.spark.operations import join


class TimeWindowContextMapper(ContextMapper):
    """
    Maps context tables through time windows for temporal aggregation.

    Time window mapping is used for event tables where we want to aggregate
    historical data over multiple time windows (e.g., citation counts over
    past 12 months, 24 months, etc.).

    Process:
    1. Read context table's _metadata.json
    2. Calculate time windows based on element partition dates
    3. Build time mapping table (date -> mapped_date, _mapping_id)
    4. Read context table with time-based partition pruning
    5. Join context data with time mapping
    6. Initialize MetadataTracker with id_mapping='time_window'

    Example configuration:
        context_mapping:
          type: time_window
          context_table: works_author
          id_columns: [author_id, work_id, date]
          time_column: date
          time_format: yyyy-MM-dd
          windows:
            - unit: month
              offset: -600
              count: 50
              length: 12
          partition_mapping:  # Optional
            time_partition: date
    """

    @classmethod
    def get_metadata_fields(cls) -> List[str]:
        """
        Return metadata fields for time_window mapper.

        Time window mapper needs additional temporal fields beyond base fields.

        Returns:
            ['type', 'id_columns', 'context_table', 'dir',
             'time_column', 'time_format', 'windows']
        """
        base_fields = super().get_metadata_fields()
        return base_fields + ['time_column', 'time_format', 'windows']

    @classmethod
    def normalize_config(cls, mapping_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and standardize configuration for metadata.

        For time_window mapper, we take the first window configuration
        from the windows list as the representative window config.

        Args:
            mapping_config: Raw configuration from YAML

        Returns:
            Normalized configuration with first window config

        Example:
            Input:
                {'type': 'time_window', 'windows': [
                    {'offset': -600, 'length': 12, 'unit': 'month', 'count': 50},
                    {'offset': -30, 'length': 1, 'unit': 'year', 'count': 30}
                ], ...}

            Output:
                {'type': 'time_window', 'windows': {
                    'offset': -600, 'length': 12, 'unit': 'month'
                }, ...}
                # Note: 'count' is removed as it's not needed in metadata
        """
        # Use base implementation to extract fields
        normalized = super().normalize_config(mapping_config)

        # Special handling for windows: take first window config
        if 'windows' in normalized and isinstance(normalized['windows'], list):
            if normalized['windows']:
                # Take first window config, excluding 'count' field
                first_window = normalized['windows'][0].copy()
                # Remove 'count' as it's not needed in metadata (it's used during mapping generation)
                first_window.pop('count', None)
                normalized['windows'] = first_window
            else:
                # Empty windows list - this should have been caught by validate_config
                normalized['windows'] = {}

        return normalized

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate time_window context_mapping configuration.

        Required fields:
        - type: must be 'time_window'
        - context_table: context table name
        - id_columns: list of ID column specs
        - time_column: name of time column in context table
        - time_format: Spark date format (e.g., 'yyyy-MM-dd')
        - windows: list of window configurations

        Each window config requires:
        - unit: 'year' | 'month' | 'day'
        - offset: starting offset (negative = past)
        - count: number of windows
        - length: window length in units

        Optional fields:
        - partition_mapping: {time_partition: 'column_name'}
        - dir: override default context table directory

        Raises:
            ValueError: If configuration is invalid
        """
        if config.get('type') != 'time_window':
            raise ValueError(f"Expected type='time_window', got '{config.get('type')}'")

        required_fields = ['context_table', 'id_columns', 'time_column', 'time_format', 'windows']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(config['windows'], list) or not config['windows']:
            raise ValueError("windows must be a non-empty list")

        # Validate each window config
        for i, window_config in enumerate(config['windows']):
            required_window_fields = ['unit', 'offset', 'count', 'length']
            for field in required_window_fields:
                if field not in window_config:
                    raise ValueError(f"Window config {i} missing required field: {field}")

            if window_config['unit'] not in ['year', 'month', 'day']:
                raise ValueError(f"Window config {i}: unit must be 'year', 'month', or 'day'")

    def read_context_table(
        self,
        spark_ctx,
        config: Dict[str, Any],
        element_partition_columns: List[str],
        element_partition_instances: List[List[str]]
    ) -> Tuple[DataFrame, List[Dict[str, Any]], List[str]]:
        """
        Read context table with time window mapping.

        Args:
            spark_ctx: SparkRunner instance
            config: context_mapping configuration
            element_partition_columns: Element partition columns (e.g., ['date'])
            element_partition_instances: Element partition instances (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            Tuple of (DataFrame, mapping_info, original_features)
            - DataFrame: Context data with _mapping_id column
            - mapping_info: List of window configurations
            - original_features: List of feature column names
        """
        # Validate configuration
        self.validate_config(config)

        # Get context table path
        context_table_name = config['context_table']
        context_table_dir = config.get('dir')
        if context_table_dir is None:
            raise ValueError("context_table directory must be specified")

        # Read context _context_info.json
        context_metadata = self._read_context_metadata(
            spark_ctx.fileio,
            context_table_dir
        )

        # Build time window mapping
        time_map_df, mapping_info, distinct_mapped_dates = self._build_time_window_mapping(
            spark_ctx.spark,
            config,
            element_partition_columns,
            element_partition_instances
        )

        # Read context table with time-based partition filtering
        context_df = self._read_with_time_filter(
            spark_ctx,
            context_table_dir,
            context_metadata.get('partition_columns', []),
            config.get('partition_mapping'),
            distinct_mapped_dates,
            config['time_column']
        )

        # Select and rename ID columns
        context_df, final_id_columns = self._select_and_rename_columns(
            context_df,
            config['id_columns']
        )

        # Join with time mapping
        # Note: join requires same column names in both DataFrames
        # The time_map_df has: [date, mapped_date, _mapping_id]
        # We need to join context's time_column with time_map's mapped_date
        time_col_name = config['time_column']

        # Rename context's time_column to 'mapped_date' for join
        context_df_renamed = context_df.withColumnRenamed(time_col_name, 'mapped_date')

        # Check for duplicate columns (excluding join column)
        # This can happen if time_column appears in both context and time_map_df
        # (e.g., both have 'date' column)
        time_map_cols = set(time_map_df.columns) - {'mapped_date'}  # Exclude join column
        context_cols = set(context_df_renamed.columns) - {'mapped_date'}
        duplicate_cols = time_map_cols & context_cols

        if duplicate_cols:
            # Drop duplicate columns from context_df before join
            # We'll use the versions from time_map_df (which has the element partition values)
            for col in duplicate_cols:
                context_df_renamed = context_df_renamed.drop(col)
                # Also remove from final_id_columns to avoid selecting duplicate columns later
                if col in final_id_columns:
                    final_id_columns.remove(col)

        joined_df = join(
            context_df_renamed,
            time_map_df,
            join_columns='mapped_date',
            join_type='inner'
        )

        # Drop the 'mapped_date' column (redundant after join)
        # We now have time_map's partition columns (e.g., 'date') in the result
        joined_df = joined_df.drop('mapped_date')

        # Get feature columns from context metadata (authoritative source)
        # This ensures only columns defined in _context_info.json are treated as features
        feature_columns_in_df = list(context_metadata['feature_columns'].keys())

        # Validate that expected feature columns exist in the DataFrame
        missing_features = set(feature_columns_in_df) - set(joined_df.columns)
        if missing_features:
            raise ValueError(
                f"Expected feature columns {missing_features} not found in context table "
                f"after join. Available columns: {joined_df.columns}"
            )

        # Select final columns: ID + partition + _mapping_id + features
        select_columns = final_id_columns + element_partition_columns + ['_mapping_id'] + feature_columns_in_df
        result_df = joined_df.select(*select_columns)

        # Return original feature column names
        original_features = feature_columns_in_df

        return result_df, mapping_info, original_features

    def _build_time_window_mapping(
        self,
        spark,
        config: Dict[str, Any],
        element_partition_columns: List[str],
        element_partition_instances: List[List[str]]
    ) -> Tuple[DataFrame, List[Dict[str, Any]], List[str]]:
        """
        Build time window mapping DataFrame using native PySpark operations.

        This implementation uses PySpark's native sequence() and explode() functions
        to avoid Python loops and driver-side data generation, providing better
        scalability and distributed processing.

        Args:
            spark: Spark session
            config: context_mapping configuration
            element_partition_columns: Element partition columns
            element_partition_instances: Element partition instances

        Returns:
            Tuple of:
            - time_map_df: DataFrame with [date, mapped_date, _mapping_id]
            - mapping_info: List of window configurations in format:
                [{'id': 'unit_offset_length', 'config': {...}}, ...]
            - distinct_mapped_dates: List of distinct mapped dates for filtering

        Example output:
            date       | mapped_date | _mapping_id
            2021-01-01 | 2020-01-01  | month_-12_3
            2021-01-01 | 2019-01-01  | month_-9_3
            2021-01-02 | 2020-02-02  | month_-12_3
            2021-01-02 | 2019-01-02  | month_-9_3

        Implementation Strategy:
            1. Create target dates DataFrame from partition instances
            2. Create window configurations DataFrame
            3. Use sequence() to generate offset values for each window
            4. Explode offsets to create individual windows
            5. Cross join targets with windows
            6. Calculate window start dates
            7. Use sequence() again to generate date arrays within windows
            8. Explode dates to final mapping rows
        """
        # Extract target dates from partition instances
        target_dates = [instance[0] for instance in element_partition_instances]

        # Get time configuration
        time_format = config['time_format']
        windows = config['windows']

        # Assert data size to ensure single partition processing is appropriate
        assert len(target_dates) < 10000, \
            f"target_dates size ({len(target_dates)}) exceeds 10000, single partition may not be appropriate"
        assert len(windows) < 10000, \
            f"windows size ({len(windows)}) exceeds 10000, single partition may not be appropriate"

        # Step 1: Create target dates DataFrame
        from pyspark.sql.types import StructType, StructField, StringType as ST
        target_schema = StructType([StructField('date', ST(), False)])
        target_df = spark.createDataFrame(
            [(d,) for d in target_dates],
            schema=target_schema
        )

        # Convert string dates to date type for calculation
        target_df = target_df.withColumn(
            'target_date',
            F.to_date(F.col('date'), time_format)
        ).coalesce(1)  # Force single partition for small data

        # Step 2: Create window configurations DataFrame
        window_configs_df = spark.createDataFrame(windows).coalesce(1)  # Force single partition for small data

        # Step 3: Generate offset sequences using sequence() and explode()
        # For each window config with count=N, generate N offsets
        # offset_list = [offset, offset+length, offset+2*length, ..., offset+(count-1)*length]
        windows_df = window_configs_df.withColumn(
            'offset_list',
            F.expr('sequence(offset, offset + (count - 1) * length, length)')
        ).withColumn(
            'actual_offset',
            F.explode(F.col('offset_list'))
        ).drop('offset_list', 'offset', 'count').coalesce(1)  # Force single partition after explode

        # Step 4: Create mapping_id as combination of unit_offset_length
        windows_df = windows_df.withColumn(
            '_mapping_id',
            F.concat_ws('_', F.col('unit'), F.col('actual_offset'), F.col('length'))
        )

        # Step 5: Cross join target dates with windows (Cartesian product)
        cross_df = target_df.crossJoin(windows_df).coalesce(1)  # Force single partition after crossJoin

        # Step 6: Calculate window start date based on unit
        # For 'year': add_months(date, offset * 12)
        # For 'month': add_months(date, offset)
        # For 'day': date_add(date, offset)
        # Note: Cast to int because add_months and date_add require integer type
        cross_df = cross_df.withColumn(
            'window_start',
            F.when(F.col('unit') == 'year',
                   F.add_months(F.col('target_date'), (F.col('actual_offset') * 12).cast('int'))
            ).when(F.col('unit') == 'month',
                   F.add_months(F.col('target_date'), F.col('actual_offset').cast('int'))
            ).when(F.col('unit') == 'day',
                   F.date_add(F.col('target_date'), F.col('actual_offset').cast('int'))
            )
        )

        # Step 7: Generate date sequences within each window
        # Use expr() with CASE to handle different units
        # sequence(start, end, step) where step is INTERVAL
        # Note: Cast length to int for add_months and date_add compatibility
        cross_df = cross_df.withColumn(
            'date_sequence',
            F.expr("""
                CASE
                    WHEN unit = 'year' THEN
                        sequence(window_start, add_months(window_start, CAST((length - 1) * 12 AS INT)), interval 1 year)
                    WHEN unit = 'month' THEN
                        sequence(window_start, add_months(window_start, CAST(length - 1 AS INT)), interval 1 month)
                    WHEN unit = 'day' THEN
                        sequence(window_start, date_add(window_start, CAST(length - 1 AS INT)), interval 1 day)
                END
            """)
        )

        # Step 8: Explode date sequences to individual rows
        result_df = cross_df.withColumn(
            'mapped_date_obj',
            F.explode(F.col('date_sequence'))
        ).coalesce(1)  # Force single partition after second explode

        # Step 9: Convert date back to string format
        result_df = result_df.withColumn(
            'mapped_date',
            F.date_format(F.col('mapped_date_obj'), time_format)
        )

        # Step 10: Select final columns
        time_map_df = result_df.select('date', 'mapped_date', '_mapping_id')

        # Step 11: Get distinct mapped dates for partition filtering
        distinct_mapped_dates = sorted([
            row.mapped_date
            for row in time_map_df.select('mapped_date').distinct().collect()
        ])

        # Step 12: Build mapping_info for compatibility
        # Collect window metadata from windows_df
        mapping_info = []
        for row in windows_df.select('_mapping_id', 'unit', 'actual_offset', 'length').distinct().collect():
            mapping_info.append({
                'id': row._mapping_id,
                'config': {
                    'unit': row.unit,
                    'offset': row.actual_offset,
                    'length': row.length
                }
            })

        # 以后再加入对time_map_df的persist方案
        return time_map_df, mapping_info, distinct_mapped_dates

    def _read_with_time_filter(
        self,
        spark_ctx,
        context_table_dir: str,
        context_partition_columns: List[str],
        partition_mapping: Dict[str, str],
        mapped_dates: List[str],
        time_column: str
    ) -> DataFrame:
        """
        Read context table with time-based partition filtering.

        If partition_mapping specifies time_partition, use mapped dates to
        filter partitions. Otherwise, filter after reading using time_column.

        Args:
            spark_ctx: SparkRunner instance
            context_table_dir: Path to context table
            context_partition_columns: Context table's partition columns
            partition_mapping: Optional dict with 'time_partition' key
            mapped_dates: List of dates needed from mapping
            time_column: Time column name in context table

        Returns:
            DataFrame with context table data filtered to relevant dates
        """
        if partition_mapping and 'time_partition' in partition_mapping:
            # Use partition pruning
            time_partition_col = partition_mapping['time_partition']

            if time_partition_col in context_partition_columns:
                # Read with partition filter
                # Convert dict format to partition_instances format
                partition_spec = {time_partition_col: mapped_dates}
                partition_instances = parse_partition_spec(context_partition_columns, partition_spec)
                df = read_table(
                    spark_ctx,
                    context_table_dir,
                    partition_columns=context_partition_columns,
                    partition_instances=partition_instances
                )
            else:
                # Column not a partition, read all and filter
                df = read_table(
                    spark_ctx,
                    context_table_dir,
                    partition_columns=context_partition_columns
                )
                df = df.filter(F.col(time_column).isin(mapped_dates))
        else:
            # No partition mapping, read all and filter by time_column
            df = read_table(
                spark_ctx,
                context_table_dir,
                partition_columns=context_partition_columns
            )
            df = df.filter(F.col(time_column).isin(mapped_dates))

        return df
