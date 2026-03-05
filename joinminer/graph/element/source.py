"""
Source processor for element table construction.

This processor orchestrates context mapping and aggregation pipeline to build
element tables with complete feature provenance.
"""

import logging
from typing import Dict, Tuple, Any, List, Optional
from pyspark.sql import DataFrame

from joinminer.spark import SparkContext
from .context_mapping import StaticContextMapper, TimeWindowContextMapper
from .aggregation_pipeline import execute_aggregation_pipeline, validate_pipeline_config
from .feature_metadata import FeatureMetadataGenerator
from joinminer.spark.operations import assemble
from joinminer.spark.io import write_table
from joinminer.spark.io.size_estimator import estimate_vector_size

logger = logging.getLogger(__name__)


class SourceProcessor:
    """
    Source processor that orchestrates context mapping and aggregation.

    Each SourceProcessor instance is dedicated to processing ONE source.
    The processor is initialized with source-specific configuration and
    processes it when needed (lazy initialization pattern).

    This processor handles all source types by:
    1. Using appropriate context mapper (static or time_window)
    2. Executing aggregation pipeline
    3. Vectorizing features with metadata tracking

    Configuration structure:
        context_mapping:
          type: static | time_window
          context_table: table_name
          id_columns: [...]
          # ... type-specific fields

        aggregation_pipeline:
          - group_by: [...]
            mapping: drop | keep | pivot
            group_count: true | false
            functions: [...]
    """

    # Mapper registry for extensibility
    # To add new mapper types, register them here instead of modifying code
    MAPPER_REGISTRY = {
        'static': StaticContextMapper,
        'time_window': TimeWindowContextMapper,
    }

    def __init__(self,
                 source_config: Dict,
                 element_config: Dict,
                 spark_ctx: SparkContext,
                 save_path: str):
        """
        Initialize processor for a single source.

        Args:
            source_config: This source's configuration (context_mapping, aggregation_pipeline, etc.)
            element_config: Element-level shared config containing:
                - element_id: List of ID columns
                - context_table_dir: Base directory for context tables
                - context_format: File format (parquet, etc.)
                - context_partition_columns: Partition columns for context tables
            spark_ctx: SparkContext instance (contains spark, fileio, persist_manager, table_state)
            save_path: Path where this source's output will be saved
                      For single source: element table path
                      For multi-source: staging path (_sources/source_{idx})

        Raises:
            ValueError: If configuration is invalid
        """
        self.source_config = source_config
        self.element_config = element_config
        self.spark_ctx = spark_ctx
        self.save_path = save_path

        # Extract commonly used fields
        self.source_name = source_config.get('name', 'unnamed_source')
        self.element_id_columns = element_config['element_id']

        # Validate configuration immediately (fail-fast)
        # Note: Context mapper will be created lazily in process()
        self._validate_config()

    def _has_group_count(self, aggregation_pipeline: List[Dict]) -> bool:
        """
        Check if any aggregation stage has group_count enabled.

        Args:
            aggregation_pipeline: List of aggregation stage configurations

        Returns:
            True if any stage has group_count: true
        """
        return any(stage.get('group_count', False) for stage in aggregation_pipeline)

    def _create_context_mapper(self, mapping_config: Dict):
        """
        Create appropriate context mapper for the mapping type.

        This is a factory method that instantiates the right mapper class
        based on the 'type' field in mapping_config, using the MAPPER_REGISTRY.

        Args:
            mapping_config: The context_mapping configuration dict

        Returns:
            ContextMapper instance (StaticContextMapper or TimeWindowContextMapper)

        Raises:
            ValueError: If mapper type is unsupported
        """
        mapping_type = mapping_config.get('type', 'static')

        mapper_class = self.MAPPER_REGISTRY.get(mapping_type)
        if mapper_class is None:
            raise ValueError(
                f"Source '{self.source_name}': Unsupported context_mapping type: {mapping_type}. "
                f"Available types: {list(self.MAPPER_REGISTRY.keys())}"
            )

        return mapper_class()

    def _validate_config(self) -> None:
        """
        Validate this source's configuration.

        Validates both context_mapping and aggregation_pipeline blocks.
        Creates a temporary mapper to validate context_mapping configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        config = self.source_config

        # Check required blocks
        if 'context_mapping' not in config:
            raise ValueError(f"Source '{self.source_name}': Missing required block: context_mapping")

        if 'aggregation_pipeline' not in config:
            raise ValueError(f"Source '{self.source_name}': Missing required block: aggregation_pipeline")

        # Validate context_mapping
        mapping_config = config['context_mapping']
        if 'type' not in mapping_config:
            raise ValueError(f"Source '{self.source_name}': context_mapping missing required field: type")

        # Create temporary mapper to validate configuration
        temp_mapper = self._create_context_mapper(mapping_config)
        temp_mapper.validate_config(mapping_config)

        # Validate aggregation_pipeline
        validate_pipeline_config(config['aggregation_pipeline'])

    def process(self,
                partition_columns: Optional[List[str]],
                partition_instances: Optional[List[List[str]]]) -> None:
        """
        Process and save this source to the configured save_path.

        This is the main entry point for source processing. It:
        1. Checks if the source is already complete for the given partitions
        2. Processes only missing partitions using _process_internal()
        3. Atomically saves: data -> metadata -> completion marker

        Args:
            partition_columns: Element partition columns (e.g., ['date'])
            partition_instances: Element partition instances (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            None - All data is persisted to self.save_path

        Raises:
            ValueError: If configuration or processing fails
        """
        # 1. Check if already complete
        is_complete, missing_partitions = self.spark_ctx.table_state.check_complete(
            self.save_path,
            partition_columns,
            partition_instances
        )

        if is_complete:
            logger.info(
                f"Source '{self.source_name}' at {self.save_path} already complete, skipping"
            )
            return

        logger.info(
            f"Processing source '{self.source_name}' for {len(missing_partitions)} "
            f"partition(s): {missing_partitions}"
        )

        # 2. Process missing partitions
        df, element_info = self._process_internal(partition_columns, missing_partitions)

        # 3. Atomic save: data -> metadata -> completion marker
        logger.info(f"Saving source '{self.source_name}' to {self.save_path}")

        # Calculate col_sizes from feature vector dimensions
        col_sizes = {}
        if element_info.get('features'):
            feature_count = len(element_info['features'])
            col_sizes['feat_vector'] = estimate_vector_size(feature_count)
            logger.info(f"Using col_sizes hint for feat_vector: {feature_count} features")

        # Write data
        write_table(
            spark_ctx=self.spark_ctx,
            df=df,
            path=self.save_path,
            partition_columns=partition_columns,
            partition_instances=missing_partitions,
            col_sizes=col_sizes
        )

        # Write metadata (always create file to preserve partition info)
        metadata_path = f"{self.save_path}/_element_info.json"
        self.spark_ctx.fileio.write_json(metadata_path, element_info)
        logger.info(f"Wrote metadata to {metadata_path}")

        # Mark complete
        self.spark_ctx.table_state.mark_complete(
            self.save_path,
            partition_columns,
            missing_partitions
        )

        logger.info(f"Source '{self.source_name}' successfully saved and marked complete")

    def _process_internal(
        self,
        partition_columns: Optional[List[str]],
        partition_instances: List[List[str]]
    ) -> Tuple[DataFrame, Dict]:
        """
        Internal processing logic: context_mapping + aggregation_pipeline + vectorization.

        Args:
            partition_columns: Element partition columns (e.g., ['date'])
            partition_instances: Element partition instances to process (e.g., [['2021-01-01'], ['2021-01-02']])

        Returns:
            Tuple of:
            - DataFrame: With columns (element_id + partition_columns + feat_vector or pure IDs)
            - element_info: Dict with keys (always non-None):
                * partition_columns: List of partition column names (may be empty)
                * features: List of feature metadata (may be empty for pure ID tables)

        Raises:
            ValueError: If configuration or processing fails
        """
        # Use instance attributes (no need to validate again, done in __init__)
        config = self.source_config
        spark_ctx = self.spark_ctx
        element_id = self.element_id_columns

        # Add default context_table directory if not specified
        mapping_config = config['context_mapping'].copy()
        if 'dir' not in mapping_config:
            # Use context_table_dir from element_config
            context_table_name = mapping_config['context_table']
            context_table_dir = self.element_config.get('context_table_dir')
            if not context_table_dir:
                raise ValueError(f"Source '{self.source_name}': context_table_dir not provided in element_config")
            mapping_config['dir'] = f"{context_table_dir}/{context_table_name}"

        # ========================================
        # Stage 1: Context Mapping
        # ========================================
        # Create mapper on-demand (lazy initialization)
        mapper = self._create_context_mapper(mapping_config)

        context_df, mapping_info, original_features = mapper.read_context_table(
            spark_ctx=spark_ctx,
            config=mapping_config,
            element_partition_columns=partition_columns,
            element_partition_instances=partition_instances
        )

        # Read context metadata for feature descriptions
        context_metadata = mapper._read_context_metadata(
            spark_ctx.fileio,
            mapping_config['dir']
        )
        context_feature_descriptions = context_metadata.get('feature_columns', {})

        # Build normalized context_mapping configuration for metadata
        # Delegate to mapper to extract relevant fields (no type-specific logic here)
        context_mapping_config = mapper.normalize_config(mapping_config)

        # Initialize feature metadata from context mapping
        # At this stage, create one entry per original feature (not per window)
        feature_metadata = FeatureMetadataGenerator.initialize_from_context(
            original_features=original_features,
            context_table_name=mapping_config['context_table'],
            context_mapping_config=context_mapping_config,
            context_feature_descriptions=context_feature_descriptions,
            context_dir=mapping_config['dir']  # Pass context_dir
        )

        # Special case: if no original features but aggregation has group_count,
        # create a template feature metadata to carry context_mapping info
        if not feature_metadata and self._has_group_count(config.get('aggregation_pipeline', [])):
            # Use normalized context_mapping_config directly (no type-specific logic needed)
            # Mapper's normalize_config already extracted all relevant fields
            template_meta = {
                'index': 0,
                'context_mapping': dict(context_mapping_config),  # Copy all fields
                'aggregations': []
            }
            # Ensure context_table and context_dir are present
            # (these may not be in context_mapping_config from normalize_config)
            template_meta['context_mapping']['context_table'] = mapping_config['context_table']
            template_meta['context_mapping']['context_dir'] = mapping_config['dir']

            feature_metadata = [template_meta]

        # ========================================
        # Stage 2: Aggregation Pipeline
        # ========================================
        aggregated_df = execute_aggregation_pipeline(
            df=context_df,
            mapping_info=mapping_info,
            original_features=original_features,
            pipeline_config=config['aggregation_pipeline']
        )

        # Update feature metadata for each aggregation stage
        for stage_config in config['aggregation_pipeline']:
            feature_metadata = FeatureMetadataGenerator.apply_aggregation_stage(
                feature_metadata=feature_metadata,
                aggregation_config=stage_config,
                mapping_info=mapping_info
            )

        # ========================================
        # Stage 2.5: Fill NULL values from aggregation
        # ========================================
        # Aggregation functions (sum, mean, etc.) produce NULL when:
        # - All values in a group are NULL
        # - Group has no matching rows
        # We need to fill these NULLs before vectorization to prevent VectorAssembler errors
        final_stage = config['aggregation_pipeline'][-1]
        stable_columns = final_stage['group_by']

        # Get feature columns (exclude ID/partition columns)
        all_columns = set(aggregated_df.columns)
        feature_columns_to_fill = list(all_columns - set(stable_columns))

        # Fill NULL values with 0 for all aggregated feature columns
        # Note: null_mark not needed since NULLs are handled in build_context_tables.py
        # and group_count already provides count tracking
        from joinminer.spark.operations import fillna
        aggregated_df = fillna(
            df=aggregated_df,
            columns=feature_columns_to_fill,
            vector_column_length={},  # No vectors at this stage
            add_mark=False,  # No need for null_mark columns
            fill_value=0
        )

        # ========================================
        # Stage 3: Vectorization
        # ========================================
        # Determine stable columns (element_id columns from final stage)
        stable_columns = final_stage['group_by']

        # Validate stable columns contain element_id
        element_id_set = set(element_id)
        stable_set = set(stable_columns)
        if not element_id_set.issubset(stable_set):
            missing = element_id_set - stable_set
            raise ValueError(
                f"Final stage group_by {stable_columns} must contain all element_id columns. "
                f"Missing: {missing}"
            )

        # Get feature columns (all columns except stable columns)
        # IMPORTANT: Preserve DataFrame column order - don't use set operations!
        feature_columns = [col for col in aggregated_df.columns if col not in stable_set]

        # Handle case with no features (pure relationship table)
        if not feature_columns:
            logger.info(
                "No feature columns found after aggregation. "
                "This is a pure relationship table (edge table) with no features."
            )
            # Just keep the stable columns (IDs + partition columns)
            vector_df = aggregated_df.select(*stable_columns)

            # Return metadata with empty features list (preserve partition info)
            element_info = {
                'partition_columns': partition_columns if partition_columns else [],
                'features': []
            }
        else:
            # Assemble feature vector
            vector_df = assemble(
                aggregated_df,
                input_columns=feature_columns,
                output_column='feat_vector',
                stable_columns=stable_columns
            )

            # Generate final element_info using FeatureMetadataGenerator
            element_info = FeatureMetadataGenerator.generate_element_info(
                feature_metadata=feature_metadata,
                partition_columns=partition_columns
            )

        return vector_df, element_info
