"""
Path Feature Preparation

This module provides functions for preparing node and edge features for bipath instances.
It handles feature extraction, normalization, and numerical ID assignment for efficient
model training.

Key Functions:
- extract_unique_elements: Extract unique nodes/edges from bipath instances
- load_element_features: Load and join features from graph element tables
- normalize_features: Apply MinMaxScaler normalization to feature vectors
- prepare_path_features: Main orchestration function for feature preparation
"""

import logging
from typing import List, Tuple
from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import MinMaxScaler, MinMaxScalerModel
from pyspark.ml.functions import vector_to_array

from joinminer.spark.io import read_table, write_table
from joinminer.spark.io.size_estimator import estimate_vector_size
from joinminer.spark.operations.row import add_row_number
from joinminer.spark.operations.fillna import fillna
from joinminer.graph.join_edges.bipath_types import get_forward_paths, get_backward_paths
from joinminer.graph.join_edges.add_path import _get_path_type_columns

# Module-level logger
logger = logging.getLogger("joinminer")


def extract_unique_elements(
    spark_ctx,
    exploration_path: str,
    target_paths_df: DataFrame,
    max_hop: int,
    partition_columns: List[str]
) -> Tuple[DataFrame, DataFrame]:
    """
    Extract unique nodes and edges from exploration paths.

    Reads exploration path tables (hop_0 to explore_hop) and extracts unique nodes and edges
    for feature preparation. Uses array operations for efficient extraction.

    The function combines forward and backward path types upfront (distinguished by seed_node_side),
    then uses a single broadcast join per hop to filter paths matching target bipath patterns.
    This approach is more efficient than separate forward/backward filtering and union operations.

    Args:
        spark_ctx: Spark context with persist_manager
        exploration_path: Base path containing exploration paths (hop_0, hop_1/seed_node_side=0/1, ...)
        target_paths_df: DataFrame with target path types for filtering
        max_hop: Maximum hop number (determines explore_hop = (max_hop + 1) // 2)
        partition_columns: Graph partition columns (e.g., ['date'])

    Returns:
        Tuple of (unique_node_df, unique_edge_df) where:
        - unique_node_df: DataFrame(node_id, node_type, *partition_columns)
        - unique_edge_df: DataFrame(u_node_id, v_node_id, edge_type, *partition_columns)
    """
    logger.info("\n" + "="*70)
    logger.info("Extracting unique nodes and edges from exploration paths")
    logger.info("="*70)

    explore_hop = (max_hop + 1) // 2
    node_df_list = []
    path_df_list = []

    # Step 1: Process hop_0 (seed nodes only, no edges)
    hop_0_path = f"{exploration_path}/hop_0"
    logger.info(f"Reading hop_0 from: {hop_0_path}")
    hop_0_df = read_table(spark_ctx, hop_0_path)

    # Hop_0 only has node_0, extract directly
    hop_0_nodes = hop_0_df.select(
        F.col('node_0_id').alias('node_id'),
        F.col('node_0_type').alias('node_type'),
        *[F.col(c) for c in partition_columns]
    ).distinct()
    node_df_list.append(hop_0_nodes)
    logger.info(f"Extracted nodes from hop_0")

    # Step 2: Prepare combined path types (forward + backward) with seed_node_side
    # Extract forward path types (from u_node, seed_node_side=0)
    forward_path_types = get_forward_paths(spark_ctx.spark, target_paths_df)
    # Extract backward path types (from v_node, seed_node_side=1)
    backward_path_types = get_backward_paths(spark_ctx.spark, target_paths_df)

    logger.info(f"Forward path types: {forward_path_types.count()} types")
    logger.info(f"Backward path types: {backward_path_types.count()} types")

    # Add seed_node_side column to distinguish forward (0) from backward (1) paths
    forward_path_types = forward_path_types.withColumn('seed_node_side', F.lit(0))
    backward_path_types = backward_path_types.withColumn('seed_node_side', F.lit(1))

    # Combine into single DataFrame for efficient filtering
    combined_path_types = forward_path_types.unionByName(backward_path_types)

    # Persist once for reuse across all hops (instead of filtering forward/backward separately each hop)
    combined_path_types = spark_ctx.persist_manager.persist(
        combined_path_types,
        release_point='path_feature_done',
        name='combined_path_types'
    )
    logger.info("Combined path types prepared with seed_node_side distinction")

    # Step 3: Process each explore_hop (1 to explore_hop)
    for hop_k in range(1, explore_hop + 1):
        hop_path = f"{exploration_path}/hop_{hop_k}"
        logger.info(f"Reading hop_{hop_k} from: {hop_path}")
        path_df = read_table(spark_ctx, hop_path)
        # Note: path_df includes seed_node_side as a partition column from the file system

        # Build join columns: path type columns + seed_node_side
        path_type_cols = _get_path_type_columns(hop_k)
        join_cols = path_type_cols + ['seed_node_side']

        # Get path types for this hop_k (includes both forward and backward)
        hop_path_types = combined_path_types.filter(F.col('hop_k') == hop_k).select(join_cols).distinct()

        # Single broadcast inner join filters all paths (forward + backward) in one operation
        # This replaces the old approach of separate forward/backward joins + union
        path_df = path_df.join(F.broadcast(hop_path_types), on=join_cols, how='inner')

        # Build array columns for node_id, node_type, edge_type, u_index, v_index
        node_id_cols = [F.col(f'node_{i}_id') for i in range(hop_k + 1)]
        node_type_cols = [F.col(f'node_{i}_type') for i in range(hop_k + 1)]
        edge_type_cols = [F.col(f'edge_{i}_type') for i in range(hop_k)]
        u_index_cols = [F.col(f'u_index_of_edge_{i}') for i in range(hop_k)]
        v_index_cols = [F.col(f'v_index_of_edge_{i}') for i in range(hop_k)]

        path_df = path_df.select(
            F.array(*node_id_cols).alias('node_id_array'),
            F.array(*node_type_cols).alias('node_type_array'),
            F.array(*edge_type_cols).alias('edge_type_array'),
            F.array(*u_index_cols).alias('u_index_array'),
            F.array(*v_index_cols).alias('v_index_array'),
            *[F.col(c) for c in partition_columns]
        )

        path_df_list.append(path_df)
        logger.info(f"Processed hop_{hop_k}")

    # Step 4: Union all path DataFrames
    all_paths_df = path_df_list[0]
    for df in path_df_list[1:]:
        all_paths_df = all_paths_df.unionByName(df)

    # Persist for reuse
    all_paths_df = spark_ctx.persist_manager.persist(
        all_paths_df,
        release_point='path_feature_done',
        name='all_paths_arrays'
    )
    logger.info(f"Combined paths from {explore_hop} hops")

    # Step 5: Extract unique nodes using arrays_zip + explode
    nodes_from_paths = all_paths_df.select(
        F.explode(F.arrays_zip('node_id_array', 'node_type_array')).alias('node_struct'),
        *partition_columns
    ).select(
        F.col('node_struct.node_id_array').alias('node_id'),
        F.col('node_struct.node_type_array').alias('node_type'),
        *partition_columns
    ).distinct()

    # Merge with hop_0 nodes
    unique_node_df = node_df_list[0].unionByName(nodes_from_paths).distinct()

    # Persist for reuse
    unique_node_df = spark_ctx.persist_manager.persist(
        unique_node_df,
        release_point='path_feature_done',
        name='unique_node_df'
    )

    node_count = unique_node_df.count()
    logger.info(f"Extracted {node_count} unique nodes")

    # Step 6: Extract unique edges using arrays_zip + explode + element_at
    unique_edge_df = all_paths_df.select(
        'node_id_array',
        F.explode(F.arrays_zip('edge_type_array', 'u_index_array', 'v_index_array')).alias('edge_struct'),
        *partition_columns
    ).select(
        F.element_at('node_id_array', F.col('edge_struct.u_index_array') + 1).alias('u_node_id'),
        F.element_at('node_id_array', F.col('edge_struct.v_index_array') + 1).alias('v_node_id'),
        F.col('edge_struct.edge_type_array').alias('edge_type'),
        *partition_columns
    ).distinct()

    # Persist for reuse
    unique_edge_df = spark_ctx.persist_manager.persist(
        unique_edge_df,
        release_point='path_feature_done',
        name='unique_edge_df'
    )

    edge_count = unique_edge_df.count()
    logger.info(f"Extracted {edge_count} unique edges")

    return unique_node_df, unique_edge_df


def normalize_features(
    spark_ctx,
    features_df: DataFrame,
    scaler_path: str,
    load_scaler: bool = False
) -> DataFrame:
    """
    Apply MinMaxScaler normalization to feature vectors.

    Args:
        spark_ctx: Spark context with SparkSession
        features_df: DataFrame with 'feat_vector' column (VectorUDT)
        scaler_path: Path to save/load scaler model
        load_scaler: If True, load existing scaler from scaler_path
                     If False, fit new scaler and save to scaler_path

    Returns:
        DataFrame with normalized 'feat_vector' column as array type (original replaced)

    Note:
        Always loads scaler from disk before transform to verify save/load works correctly.
        Original feat_vector is dropped and replaced with normalized version as array type.
        Vector type is only needed for MinMaxScaler; converted to array after normalization.
    """
    logger.info(f"  Normalizing features (load_scaler={load_scaler})")

    if not load_scaler:
        # Fit new scaler and save
        logger.info("    Fitting new MinMaxScaler...")
        scaler = MinMaxScaler(inputCol='feat_vector', outputCol='normalized_features')
        scaler_model = scaler.fit(features_df)

        logger.info(f"    Saving scaler to: {scaler_path}")
        scaler_model.write().overwrite().save(scaler_path)

    # Always load from disk (verifies save/load works correctly)
    logger.info(f"    Loading scaler from: {scaler_path}")
    scaler_model = MinMaxScalerModel.load(scaler_path)

    # Apply normalization
    normalized_df = scaler_model.transform(features_df)

    # Replace original feat_vector with normalized version and convert to array
    normalized_df = normalized_df.drop('feat_vector') \
                                 .withColumnRenamed('normalized_features', 'feat_vector') \
                                 .withColumn('feat_vector', vector_to_array('feat_vector'))

    logger.info("    Normalization completed (converted to array)")

    return normalized_df


def prepare_path_features(
    spark_ctx,
    graph,
    output_path: str,
    max_hop: int,
    target_paths_df: DataFrame,
    path_node_type_indices: set,
    path_edge_type_indices: set,
    scaler_base_path: str,
    load_scaler: bool,
    element_partition_instances: List[List[str]] = None
) -> None:
    """
    Main orchestration function for path feature preparation.

    Prepares node and edge features for bipath instances by:
    1. Extracting unique elements (nodes/edges) from exploration paths
    2. Reading element tables and inner joining with unique elements
    3. Checking element counts (raises error if 0 elements found)
    4. Assembling and normalizing features (if element type has features)
    5. Assigning sequential feat_id to all elements
    6. Saving prepared features with per-element-type persist/release

    Processing Flow per Element Type:
    - Read element table based on graph config's table_path
    - Inner join with unique elements (on node_id/edge endpoints + partition_columns)
    - Persist for element-specific processing
    - Count elements and raise error if 0 (indicates missing data)
    - Check if has features using graph config's feature_count
    - If has features: assemble feature vector from feature_0, feature_1, ... columns
    - If has features: normalize using MinMaxScaler (load or generate based on mode)
    - Always assign feat_id using row numbering
    - Save results and release element-specific persist

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with nodes/edges configs (must have table_path, feature_count)
        output_path: Base output path (features saved to {output_path}/features/)
        max_hop: Maximum hop number in bipaths
        target_paths_df: DataFrame with target path types for filtering exploration paths
        path_node_type_indices: Set of node type indices to process
        path_edge_type_indices: Set of edge type indices to process
        scaler_base_path: Base path to save/load scalers (required)
                         Should point to the /scalers directory (e.g., {output_path}/scalers)
        load_scaler: If False, fit new scalers and save to scaler_base_path
                     If True, load existing scalers from scaler_base_path
        element_partition_instances: Optional partition instances to read from element tables
                                     (e.g., [['2024-01-01'], ['2024-01-02']])
                                     If provided, only these partitions are read
                                     If None, all partitions are read
                                     Should match date_partitions from bipath collection

    Output Structure:
        {output_path}/
        ├── features/
        │   ├── nodes/
        │   │   └── {node_type}/
        │   │       ├── _SUCCESS
        │   │       └── part-*.parquet
        │   └── edges/
        │       └── {edge_type}/
        │           ├── _SUCCESS
        │           └── part-*.parquet
        └── scalers/
            ├── nodes/
            │   └── {node_type}/  (MinMaxScaler model, only if has features)
            └── edges/
                └── {edge_type}/  (MinMaxScaler model, only if has features)

    Output Columns:
        - Elements with features: element_id(s), feature_*, features, normalized_features, feat_id, partition_columns
        - Elements without features: element_id(s), feat_id, partition_columns

    Raises:
        ValueError: If no elements found for any element type (indicates data issue)
    """
    logger.info("\n" + "="*70)
    logger.info("Preparing Path Features")
    logger.info("="*70)

    features_base_path = f"{output_path}/features"
    exploration_path = f"{output_path}/path_exploration"

    # Log scaler mode
    if load_scaler:
        logger.info(f"Scaler mode: Load from {scaler_base_path}")
    else:
        logger.info(f"Scaler mode: Generate new and save to {scaler_base_path}")

    # Check if features are already prepared
    is_complete, _ = spark_ctx.table_state.check_complete(features_base_path)
    if is_complete:
        logger.info(f"Features already prepared at {features_base_path}, skipping")
        return

    # Step 1: Extract unique elements
    logger.info("\nStep 1: Extract unique elements from exploration paths")
    logger.info("-"*70)

    unique_node_df, unique_edge_df = extract_unique_elements(
        spark_ctx=spark_ctx,
        exploration_path=exploration_path,
        target_paths_df=target_paths_df,
        max_hop=max_hop,
        partition_columns=graph.partition_columns
    )

    # Step 2: Process node features
    logger.info("\nStep 2: Process node features")
    logger.info("-"*70)

    for node_type_index in path_node_type_indices:
        node_type_name = graph.node_index_to_type[node_type_index]
        node_config = graph.nodes[node_type_name]
        node_feature_count = node_config['feature_count']
        logger.info(f"\nProcessing node type: {node_type_name} (index: {node_type_index})")

        # Construct output path first
        node_output_path = f"{features_base_path}/nodes/{node_type_name}"

        # Check if this node type already processed
        is_complete, _ = spark_ctx.table_state.check_complete(node_output_path)
        if is_complete:
            logger.info(f"  Node type '{node_type_name}' already processed, skipping")
            continue

        # Filter unique_node_df to this node type
        type_nodes_df = unique_node_df.filter(F.col('node_type') == node_type_index).select(
            'node_id', *graph.partition_columns
        )

        # Read element table directly (only target partitions if specified)
        element_path = node_config['table_path']
        logger.info(f"  Reading element table from: {element_path}")
        element_df = read_table(
            spark_ctx,
            element_path,
            partition_columns=graph.partition_columns,
            partition_instances=element_partition_instances
        )

        # Rename original id column to standardized 'node_id'
        id_column = node_config['id_column']
        element_df = element_df.withColumnRenamed(id_column, 'node_id')

        # Left join to preserve all nodes from paths (even if not in element table)
        element_df = type_nodes_df.join(
            element_df,
            on=type_nodes_df.columns,
            how='left'
        )

        # Fill NULL feat_vector BEFORE persist (if has features)
        if node_feature_count:
            element_df = fillna(
                df=element_df,
                columns=['feat_vector'],
                vector_column_length={'feat_vector': node_feature_count},
                add_mark=False,
                fill_value=0
            )

        # Persist for this element type processing (caches fillna'd result if has features)
        node_release_point = f'node_{node_type_name}_done'
        element_df = spark_ctx.persist_manager.persist(
            element_df,
            release_point=node_release_point,
            name=f'node_{node_type_name}_elements'
        )

        # Check if any elements exist
        element_count = element_df.count()
        logger.info(f"  Found {element_count} elements for node type '{node_type_name}'")
        if element_count == 0:
            raise ValueError(
                f"No elements found for node type '{node_type_name}'. "
                f"This indicates the element table is empty or has no matching elements."
            )

        # Normalize features if this node type has features
        if node_feature_count:
            logger.info(f"  Node type has features (feature_count: {node_feature_count})")

            # Construct scaler path
            scaler_path = f"{scaler_base_path}/nodes/{node_type_name}"

            # Normalize features - MinMaxScaler directly on feat_vector
            element_df = normalize_features(
                spark_ctx=spark_ctx,
                features_df=element_df,
                scaler_path=scaler_path,
                load_scaler=load_scaler
            )
        else:
            logger.info(f"  Node type has no features")

        # Always assign feature IDs regardless of whether features exist
        logger.info("  Assigning feature IDs...")
        final_df = add_row_number(
            spark_ctx.spark,
            element_df,
            row_num_col='feat_id'
        )

        # Prepare col_sizes for write optimization
        col_sizes = {}
        if node_feature_count:
            col_sizes['feat_vector'] = estimate_vector_size(node_feature_count)

        # Save results
        logger.info(f"  Saving to: {node_output_path}")
        write_table(
            spark_ctx=spark_ctx,
            df=final_df,
            path=node_output_path,
            mode='overwrite',
            col_sizes=col_sizes
        )

        # Mark this element type as complete
        spark_ctx.table_state.mark_complete(node_output_path)

        # Release persisted data for this element type
        spark_ctx.persist_manager.mark_released(node_release_point)
        logger.info(f"  Completed processing for node type '{node_type_name}'")

    # Step 3: Process edge features
    logger.info("\nStep 3: Process edge features")
    logger.info("-"*70)

    for edge_type_index in path_edge_type_indices:
        edge_type_name = graph.edge_index_to_type[edge_type_index]
        edge_config = graph.edges[edge_type_name]
        edge_feature_count = edge_config['feature_count']
        logger.info(f"\nProcessing edge type: {edge_type_name} (index: {edge_type_index})")

        # Construct output path first
        edge_output_path = f"{features_base_path}/edges/{edge_type_name}"

        # Check if this edge type already processed
        is_complete, _ = spark_ctx.table_state.check_complete(edge_output_path)
        if is_complete:
            logger.info(f"  Edge type '{edge_type_name}' already processed, skipping")
            continue

        # Filter unique_edge_df to this edge type
        type_edges_df = unique_edge_df.filter(F.col('edge_type') == edge_type_index).select(
            'u_node_id', 'v_node_id', *graph.partition_columns
        )

        # Read element table directly (only target partitions if specified)
        element_path = edge_config['table_path']
        logger.info(f"  Reading element table from: {element_path}")
        element_df = read_table(
            spark_ctx,
            element_path,
            partition_columns=graph.partition_columns,
            partition_instances=element_partition_instances
        )

        # Rename original id columns to standardized 'u_node_id', 'v_node_id'
        u_id_column = edge_config['u_node']['id_column']
        v_id_column = edge_config['v_node']['id_column']
        element_df = element_df.withColumnRenamed(u_id_column, 'u_node_id') \
                               .withColumnRenamed(v_id_column, 'v_node_id')

        # Inner join to get only existing edges
        element_df = element_df.join(
            type_edges_df,
            on=type_edges_df.columns,
            how='inner'
        )

        # Persist for this element type processing
        edge_release_point = f'edge_{edge_type_name}_done'
        element_df = spark_ctx.persist_manager.persist(
            element_df,
            release_point=edge_release_point,
            name=f'edge_{edge_type_name}_elements'
        )

        # Check if any elements exist
        element_count = element_df.count()
        logger.info(f"  Found {element_count} elements for edge type '{edge_type_name}'")
        if element_count == 0:
            raise ValueError(
                f"No elements found for edge type '{edge_type_name}'. "
                f"This indicates the element table is empty or has no matching elements."
            )

        # Normalize features if this edge type has features
        if edge_feature_count:
            logger.info(f"  Edge type has features (feature_count: {edge_feature_count})")

            # No fillna needed for edges - all edges in paths exist in element table
            # feat_vector already exists from element table

            # Construct scaler path
            scaler_path = f"{scaler_base_path}/edges/{edge_type_name}"

            # Normalize features - MinMaxScaler directly on feat_vector
            element_df = normalize_features(
                spark_ctx=spark_ctx,
                features_df=element_df,
                scaler_path=scaler_path,
                load_scaler=load_scaler
            )
        else:
            logger.info(f"  Edge type has no features")

        # Always assign feature IDs regardless of whether features exist
        logger.info("  Assigning feature IDs...")
        final_df = add_row_number(
            spark_ctx.spark,
            element_df,
            row_num_col='feat_id'
        )

        # Prepare col_sizes for write optimization
        col_sizes = {}
        if edge_feature_count:
            col_sizes['feat_vector'] = estimate_vector_size(edge_feature_count)

        # Save results
        logger.info(f"  Saving to: {edge_output_path}")
        write_table(
            spark_ctx=spark_ctx,
            df=final_df,
            path=edge_output_path,
            mode='overwrite',
            col_sizes=col_sizes
        )

        # Mark this element type as complete
        spark_ctx.table_state.mark_complete(edge_output_path)

        # Release persisted data for this element type
        spark_ctx.persist_manager.mark_released(edge_release_point)
        logger.info(f"  Completed processing for edge type '{edge_type_name}'")

    # Release persisted DataFrames
    spark_ctx.persist_manager.mark_released('path_feature_done')

    # Mark features preparation as complete
    spark_ctx.table_state.mark_complete(features_base_path)
    logger.info("\n" + "="*70)
    logger.info("Path feature preparation completed successfully!")
    logger.info("="*70)
