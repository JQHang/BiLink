"""
Element table builder for graph construction.

This module provides the main ElementBuilder class that orchestrates the creation
of element tables from multiple sources.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
from pyspark.sql import DataFrame, functions as F

from joinminer.spark import SparkContext
from joinminer.spark.operations import join, assemble, fillna
from joinminer.spark.io import write_table, read_table
from joinminer.spark.io.partition import parse_partition_spec
from .source import SourceProcessor

# Module-level logger
logger = logging.getLogger(__name__)


class ElementBuilder:
    """
    Builder for a single element table.

    Each ElementBuilder instance is dedicated to building one specific element.
    The builder encapsulates all configuration and logic for that element.

    Element tables are the building blocks for graph construction, containing:
    - Entity ID columns
    - Partition columns (e.g., date)
    - Feature vectors (always generated)

    This builder:
    1. Processes one or more sources (temporal, static, or custom)
    2. Merges multiple sources via outer join
    3. Merges multiple feature vectors if present
    4. Cleans up intermediate source tables
    """

    def __init__(
        self,
        spark_ctx: SparkContext,
        element_name: str,
        element_config: Dict,
        context_table_config: Dict,
        element_table_config: Dict
    ):
        """
        Initialize the builder for a single element.

        All configuration is prepared and validated during initialization.
        The build() method only needs partition values.

        Args:
            spark_ctx: SparkContext instance (contains spark, fileio, persist_manager, table_state)
            element_name: Name of the element (e.g., 'work', 'author')
            element_config: Configuration for this specific element:
                {
                    'element_id': ['work_id', 'date'],
                    'partition_spec': 'date_partitions',  # optional
                    'sources': [...]
                }
            context_table_config: Global context table configuration:
                {
                    'dir': 'file:///path/to/context',
                    'format': 'parquet',
                    'partition_columns': ['date']
                }
            element_table_config: Global element table configuration:
                {
                    'dir': 'file:///path/to/element',
                    'format': 'parquet',
                    'partition_columns': ['date']
                }
        """
        self.spark_ctx = spark_ctx
        self.element_name = element_name

        # Store configurations
        self.element_config = element_config
        self.context_table_config = context_table_config
        self.element_table_config = element_table_config

        # Parse element configuration
        self.element_id = element_config['element_id']
        self.partition_spec_name = element_config.get('partition_spec')

        # Build output path
        self.output_path = f"{element_table_config['dir']}/{element_name}"

        # Store raw source configs (no preprocessing needed)
        self.source_configs = element_config['sources']

        # Validate basic configuration
        self._validate_basic_config()

    def build(self, partition_spec: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Build this element table with unified path strategy.

        Flow:
        1. Check element-level completion
        2. Determine strategy based on source count:
           - Single source: Direct save to element path
           - Multiple sources: Stage -> Merge -> Cleanup
        3. Mark element complete

        Args:
            partition_spec: User-provided partition specification to build.
                Format: {'date': ['2021-01-01', '2021-01-02']}
                For non-partitioned elements, pass None.

        Examples:
            >>> # Build partitioned element
            >>> work_builder = ElementBuilder(
            ...     spark_runner, 'work', element_config,
            ...     context_table_config, element_table_config
            ... )
            >>> work_builder.build({'date': ['2021-01-01']})

            >>> # Build non-partitioned element
            >>> static_builder = ElementBuilder(...)
            >>> static_builder.build(None)
        """
        # 1. Resolve partition configuration
        partition_columns, partition_instances = self._resolve_partitions(partition_spec)

        # 2. Check element-level completion
        is_complete, missing_partition_instances = self.spark_ctx.table_state.check_complete(
            self.output_path, partition_columns, partition_instances
        )
        if is_complete:
            logger.info(f"Element '{self.element_name}' already complete, skipping")
            return

        # 3. Determine strategy based on source count
        num_sources = len(self.source_configs)

        if num_sources == 0:
            raise ValueError(f"Element '{self.element_name}' has no sources configured")

        logger.info(
            f"Building element '{self.element_name}' from {num_sources} source(s) "
            f"for {len(missing_partition_instances)} partition(s)"
        )

        if num_sources == 1:
            # Single source: Direct save to element path
            self._process_single_source_direct(
                partition_columns, missing_partition_instances
            )
        else:
            # Multiple sources: Stage -> Merge -> Cleanup
            self._process_multi_source_with_merge(
                partition_columns, missing_partition_instances
            )

        # 4. Mark element complete (both strategies need this)
        self.spark_ctx.table_state.mark_complete(
            self.output_path, partition_columns, missing_partition_instances
        )

        logger.info(f"Successfully built element table: {self.element_name}")

    def _resolve_partitions(
        self,
        partition_spec: Optional[Dict[str, List[str]]]
    ) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
        """
        Resolve partition specification into standardized format.

        Args:
            partition_spec: User-provided partition specification dict
                Example: {'date': ['2021-01-01', '2021-01-02']}

        Returns:
            Tuple of (partition_columns, partition_instances):
            - partition_columns: ['date'] or None for non-partitioned
            - partition_instances: [['2021-01-01'], ['2021-01-02']] or None

        Raises:
            ValueError: If partition configuration is invalid
        """
        if not self.partition_spec_name:
            # Non-partitioned element
            if partition_spec:
                logger.warning(
                    f"Element '{self.element_name}' is not partitioned, "
                    f"but partition_spec was provided (will be ignored)"
                )
            return None, None

        # Partitioned element requires partition spec
        if not partition_spec:
            raise ValueError(
                f"Element '{self.element_name}' requires partition_spec "
                f"for partition_spec '{self.partition_spec_name}'"
            )

        # Get partition columns from element_table config
        partition_columns = self.element_table_config.get('partition_columns')
        if not partition_columns:
            raise ValueError(
                f"Element '{self.element_name}': element_table.partition_columns "
                f"must be defined for partitioned elements"
            )

        # Parse partition specification into standardized format
        partition_instances = parse_partition_spec(partition_columns, partition_spec)

        return partition_columns, partition_instances

    def _validate_basic_config(self) -> None:
        """
        Validate basic element configuration during initialization.

        Source-level validation is deferred to SourceProcessor initialization
        during processing (lazy validation).

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate element_id
        if not self.element_id:
            raise ValueError(f"Element '{self.element_name}': element_id cannot be empty")
        if not isinstance(self.element_id, list):
            raise ValueError(f"Element '{self.element_name}': element_id must be a list")

        # Validate sources exist
        if not self.source_configs:
            raise ValueError(f"Element '{self.element_name}': requires at least one source")

        # Source-specific validation happens during processing when SourceProcessor is initialized

    def _process_single_source_direct(
        self,
        partition_columns: Optional[List[str]],
        partition_instances: Optional[List[List[str]]]
    ) -> None:
        """
        Process single source and save directly to element path.

        This is the optimized path for single-source elements:
        - No staging needed
        - Direct save to final element location
        - Source processor handles completion checking and saving

        Args:
            partition_columns: Element partition columns
            partition_instances: List of partition instances to process
        """
        source_config = self.source_configs[0]
        source_name = source_config.get('name', 'source_0')

        logger.info(f"Processing single source '{source_name}' directly to element path")

        # Prepare element configuration for source processor
        element_config = {
            'element_id': self.element_id,
            'context_table_dir': self.context_table_config['dir'],
            'context_format': self.context_table_config.get('format', 'parquet'),
            'context_partition_columns': self.context_table_config.get('partition_columns', [])
        }

        # Create source processor with element path (direct save!)
        processor = SourceProcessor(
            source_config=source_config,
            element_config=element_config,
            spark_ctx=self.spark_ctx,
            save_path=self.output_path  # Direct save to element path!
        )

        # Process and save (no return value)
        processor.process(partition_columns, partition_instances)

        logger.info(f"Single source '{source_name}' processed and saved to {self.output_path}")

    def _process_multi_source_with_merge(
        self,
        partition_columns: Optional[List[str]],
        partition_instances: Optional[List[List[str]]]
    ) -> None:
        """
        Process multiple sources with staging, merging, and cleanup.

        Flow:
        1. Process each source to staging path (_sources/source_{idx})
        2. Read back and merge all sources
        3. Write merged result to element path
        4. Cleanup staging directories

        Args:
            partition_columns: Element partition columns
            partition_instances: List of partition instances to process
        """
        # Prepare element configuration (shared by all sources)
        element_config = {
            'element_id': self.element_id,
            'context_table_dir': self.context_table_config['dir'],
            'context_format': self.context_table_config.get('format', 'parquet'),
            'context_partition_columns': self.context_table_config.get('partition_columns', [])
        }

        # Step 1: Process all sources to staging paths
        staging_paths = []

        for idx, source_config in enumerate(self.source_configs):
            source_name = source_config.get('name', f'source_{idx}')
            staging_path = f"{self.output_path}/_sources/source_{idx}"
            staging_paths.append(staging_path)

            logger.info(f"Processing source {idx}: '{source_name}' to staging path: {staging_path}")

            # Create source processor with staging path
            processor = SourceProcessor(
                source_config=source_config,
                element_config=element_config,
                spark_ctx=self.spark_ctx,
                save_path=staging_path  # Staging path!
            )

            # Process and save (no return value)
            processor.process(partition_columns, partition_instances)

        # Step 2: Merge staged sources
        self._merge_staged_sources(staging_paths, partition_columns, partition_instances)

        # Step 3: Cleanup staging data (always cleanup temporary staging directory)
        logger.info(f"Cleaning up staging directory: {self.output_path}/_sources")
        self.spark_ctx.fileio.delete(f"{self.output_path}/_sources", recursive=True)

    def _merge_staged_sources(
        self,
        staging_paths: List[str],
        partition_columns: Optional[List[str]],
        partition_instances: Optional[List[List[str]]]
    ) -> None:
        """
        Read staged sources from disk and merge into final element table.

        This is a pure read-and-merge operation with no reprocessing.

        Args:
            staging_paths: List of staging directory paths
            partition_columns: Element partition columns
            partition_instances: List of partition instances
        """
        logger.info(f"Merging {len(staging_paths)} staged sources")

        # Read all sources and metadata
        source_data = []

        for idx, path in enumerate(staging_paths):
            # Read DataFrame
            df = read_table(self.spark_ctx, path)

            # Read metadata
            metadata_path = f"{path}/_element_info.json"
            if self.spark_ctx.fileio.exists(metadata_path):
                metadata = self.spark_ctx.fileio.read_json(metadata_path)
            else:
                metadata = {}

            source_name = path.split('/')[-1]  # e.g., "source_0"
            source_data.append({
                'df': df,
                'metadata': metadata,
                'name': source_name,
                'index': idx
            })

        # Merge DataFrames (pass source_data to access metadata)
        merged_df = self._merge_dataframes(source_data)

        # Merge metadata
        merged_metadata = self._merge_metadata(source_data, partition_columns)

        # Calculate col_sizes from merged feature vector dimensions
        col_sizes = {}
        if merged_metadata.get('features'):
            feature_count = len(merged_metadata['features'])
            col_sizes['feat_vector'] = feature_count
            logger.info(
                f"Using col_sizes hint for merged feat_vector: {feature_count} features"
            )

        # Write to element path
        logger.info(f"Writing merged data to element path: {self.output_path}")
        write_table(
            self.spark_ctx,
            merged_df,
            self.output_path,
            partition_columns=partition_columns,
            partition_instances=partition_instances,
            col_sizes=col_sizes
        )

        # Write merged metadata (always create file to preserve partition info)
        metadata_path = f"{self.output_path}/_element_info.json"
        self.spark_ctx.fileio.write_json(metadata_path, merged_metadata)
        logger.info(f"Wrote merged metadata to {metadata_path}")

    def _merge_dataframes(self, source_data: List[Dict]) -> DataFrame:
        """
        Merge multiple DataFrames via outer join on element_id.

        Handles mixed scenarios:
        - All sources have features (feat_vector column)
        - All sources have no features (pure ID tables)
        - Mixed: some sources have features, others don't

        Args:
            source_data: List of dicts with 'df', 'metadata', 'name', 'index' keys

        Returns:
            Merged DataFrame with combined feat_vector (or pure ID table if no features)
        """
        # Extract DataFrames
        dfs = [s['df'] for s in source_data]

        # Get join keys (element_id already includes partition columns if any)
        join_keys = self.element_id

        # Separate sources by whether they have feat_vector column
        sources_with_vector = []
        sources_without_vector = []

        for i, source in enumerate(source_data):
            df = source['df']
            if 'feat_vector' in df.columns:
                sources_with_vector.append({'index': i, 'source': source, 'df': df})
            else:
                sources_without_vector.append({'index': i, 'source': source, 'df': df})

        logger.info(
            f"Merging {len(dfs)} DataFrames with outer join on {join_keys}: "
            f"{len(sources_with_vector)} with features, "
            f"{len(sources_without_vector)} without features"
        )

        # If no sources have features, join and return pure ID table
        if not sources_with_vector:
            logger.info("No sources have features, joining and returning pure ID table")
            merged = dfs[0]
            for df in dfs[1:]:
                merged = join(
                    left_df=merged,
                    right_df=df,
                    join_columns=join_keys,
                    join_type='outer'
                )
            return merged.select(*join_keys)

        # Rename vector columns BEFORE joining (to avoid column conflicts)
        vector_columns = []
        for item in sources_with_vector:
            i = item['index']
            vec_col = f"feat_vector_{i}"
            dfs[i] = dfs[i].withColumnRenamed('feat_vector', vec_col)
            vector_columns.append(vec_col)

        # Outer join all DataFrames (ID-level merge)
        merged = dfs[0]
        for df in dfs[1:]:
            merged = join(
                left_df=merged,
                right_df=df,
                join_columns=join_keys,
                join_type='outer'
            )

        # Build vector_columns_size mapping from metadata
        vector_columns_size = {}
        total_features = 0

        for item in sources_with_vector:
            i = item['index']
            source = item['source']
            vec_col = f"feat_vector_{i}"
            metadata = source['metadata']

            if metadata and 'features' in metadata:
                size = len(metadata['features'])
                vector_columns_size[vec_col] = size
                total_features += size
                logger.debug(
                    f"Source {i} ('{source['name']}'): {size} features in {vec_col}"
                )
            else:
                # This should not happen: if df has feat_vector, it should have metadata
                raise ValueError(
                    f"Source {i} ('{source['name']}') has feat_vector column "
                    f"but no feature metadata. Check SourceProcessor output."
                )

        # Fill null vectors after outer join
        # Outer join creates null vectors for rows that don't exist in all sources
        logger.info("Filling null vectors with zero vectors after outer join")
        merged = fillna(
            df=merged,
            columns=list(vector_columns_size.keys()),
            vector_column_length=vector_columns_size,
            add_mark=False,  # Don't add null marker columns
            fill_value=0     # Not used for vectors, but required parameter
        )

        # Combine vectors: assemble if multiple, rename if single
        if len(vector_columns) > 1:
            # Multiple vectors: assemble
            merged = assemble(
                df=merged,
                input_columns=vector_columns,
                output_column='feat_vector',
                stable_columns=join_keys,
                vector_columns_size=vector_columns_size
            )
            logger.info(
                f"Successfully merged {len(vector_columns)} vectors: "
                f"{total_features} total features"
            )
        else:
            # Single vector: direct rename (no assembly needed)
            vec_col = vector_columns[0]
            merged = merged.withColumnRenamed(vec_col, 'feat_vector')
            logger.info(
                f"Renamed {vec_col} -> feat_vector "
                f"({total_features} features, no assembly needed)"
            )

        return merged

    def _merge_metadata(
        self,
        source_data: List[Dict],
        partition_columns: Optional[List[str]]
    ) -> Dict:
        """
        Merge metadata from sources with features.

        Sources without features (pure ID tables) are automatically skipped.

        Args:
            source_data: List of dicts with 'metadata', 'name', 'index' keys
            partition_columns: Element partition columns

        Returns:
            Merged metadata dict with keys:
            - partition_columns: List of partition column names (may be empty)
            - features: List of feature metadata (may be empty for pure ID tables)
        """
        # Merge features from sources that have them
        merged_features = []
        current_index = 0
        sources_with_features = 0
        sources_without_features = 0

        for source in source_data:
            metadata = source['metadata']

            # Skip sources without features (pure ID tables)
            if not metadata or 'features' not in metadata:
                sources_without_features += 1
                logger.debug(
                    f"Source '{source['name']}' has no features (pure ID table), "
                    f"skipping metadata merge"
                )
                continue

            # Add features with updated indices
            sources_with_features += 1
            for feature in metadata['features']:
                feature_copy = feature.copy()
                feature_copy['index'] = current_index
                merged_features.append(feature_copy)
                current_index += 1

        # Return empty features if no features at all (preserve partition info)
        if not merged_features:
            logger.info(
                f"No features in any source ({sources_without_features} pure ID tables), "
                f"element table will have metadata with empty features"
            )
            return {
                'partition_columns': partition_columns if partition_columns else [],
                'features': []
            }

        logger.info(
            f"Merged {current_index} features from {sources_with_features} source(s) "
            f"({sources_without_features} source(s) without features skipped)"
        )

        return {
            'partition_columns': partition_columns,
            'features': merged_features
        }
