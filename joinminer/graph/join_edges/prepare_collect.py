"""
Prepare Collection Data for Batch Processing

This module provides functions to pre-compute data that can be processed in full
(by seed node dimension) to avoid redundant computation during batch processing.

Key Functions:
- prepare_collect: Main orchestration function for preparing collection data
- get_seed_node_elements: Extract elements per seed node
- get_seed_node_paths: Assemble paths per seed node (without joining to pairs)
"""

import logging
from typing import Set
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

from joinminer.spark.io import read_table, write_table
from joinminer.graph.join_edges.bipath_collect_feat import (
    create_feat_id_mappings,
    extract_path_elements
)

# Module-level logger
logger = logging.getLogger("joinminer")


def get_seed_node_elements(
    spark_ctx,
    graph,
    output_path: str,
    path_nodes: DataFrame,
    path_edges: DataFrame,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> None:
    """
    Extract elements (feat_ids) collected by each seed node, grouped by element_type.

    Includes seed nodes themselves plus path_nodes and path_edges, deduplicated
    by (node_0_id, seed_node_side, element_type, feat_id).

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with partition columns
        output_path: Base output path (saves to /seed_node_elements)
        path_nodes: Path nodes DataFrame with feat_ids
        path_edges: Path edges DataFrame with feat_ids
        path_node_type_indices: Set of node type indices in target paths
        path_edge_type_indices: Set of edge type indices in target paths

    Output:
        Saves to {output_path}/seed_node_elements with schema:
        - node_0_id, seed_node_side, element_type, feat_id, partition_columns
    """
    seed_node_elements_output = f"{output_path}/seed_node_elements"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(seed_node_elements_output)
    if is_complete:
        logger.info(f"Seed node elements already exists at {seed_node_elements_output}, skipping")
        return

    logger.info("\n" + "="*70)
    logger.info("Extracting seed node elements")
    logger.info("="*70)

    # Filter element types with features
    feat_node_type_indices = {
        idx for idx in path_node_type_indices
        if graph.nodes[graph.node_index_to_type[idx]]['feature_count'] > 0
    }
    feat_edge_type_indices = {
        idx for idx in path_edge_type_indices
        if graph.edges[graph.edge_index_to_type[idx]]['feature_count'] > 0
    }

    logger.info(f"  Node types with features: {len(feat_node_type_indices)}/{len(path_node_type_indices)}")
    logger.info(f"  Edge types with features: {len(feat_edge_type_indices)}/{len(path_edge_type_indices)}")

    if not feat_node_type_indices and not feat_edge_type_indices:
        logger.info("  No element types with features, skipping seed node elements")
        return

    # Step 1: Read seed node elements from saved file
    logger.info("\nStep 1: Reading seed node elements")
    logger.info("-"*70)

    seed_node_feat_path = f"{output_path}/seed_node_feat"
    seed_node_feat_df = read_table(spark_ctx, seed_node_feat_path)

    seed_node_feat_ids = seed_node_feat_df.select(
        'node_0_id',
        'seed_node_side',
        F.concat(F.lit('0_'), F.col('node_type')).alias('element_type'),
        'feat_id',
        *graph.partition_columns
    )

    element_dfs = [seed_node_feat_ids]
    logger.info(f"  Read seed node elements")

    # Step 2: Extract elements from path_nodes and path_edges
    logger.info("\nStep 2: Extracting elements from paths")
    logger.info("-"*70)

    # From path_nodes: keep node_0_id, seed_node_side, element_type, feat_id
    if feat_node_type_indices:
        path_node_elements = path_nodes.filter(
            F.col('node_type').isin(list(feat_node_type_indices))
        ).select(
            F.col('node_0_id'),
            F.col('seed_node_side'),
            F.concat(F.lit('0_'), F.col('node_type')).alias('element_type'),
            F.col('feat_id'),
            *[F.col(c) for c in graph.partition_columns]
        )
        element_dfs.append(path_node_elements)

    # From path_edges: keep node_0_id, seed_node_side, element_type, feat_id
    if feat_edge_type_indices:
        path_edge_elements = path_edges.filter(
            F.col('edge_type').isin(list(feat_edge_type_indices))
        ).select(
            F.col('node_0_id'),
            F.col('seed_node_side'),
            F.concat(F.lit('1_'), F.col('edge_type')).alias('element_type'),
            F.col('feat_id'),
            *[F.col(c) for c in graph.partition_columns]
        )
        element_dfs.append(path_edge_elements)

    # Union all elements
    all_elements = element_dfs[0]
    for df in element_dfs[1:]:
        all_elements = all_elements.unionByName(df)

    logger.info(f"  Extracted elements from paths")

    # Step 3: Deduplicate by (node_0_id, seed_node_side, element_type, feat_id)
    logger.info("\nStep 3: Deduplicating elements")
    logger.info("-"*70)

    seed_node_elements = all_elements.select(
        'node_0_id', 'seed_node_side', 'element_type', 'feat_id', *graph.partition_columns
    ).distinct()

    logger.info(f"  Deduplicated elements")

    # Step 4: Save output
    logger.info("\nStep 4: Saving seed node elements")
    logger.info("-"*70)

    write_table(spark_ctx, seed_node_elements, seed_node_elements_output, mode='overwrite')

    logger.info(f"  Saved to {seed_node_elements_output}")

    # Mark as complete
    spark_ctx.table_state.mark_complete(seed_node_elements_output)

    logger.info(f"\n{'='*70}")
    logger.info(f"Seed node elements extraction completed")
    logger.info(f"{'='*70}")


def get_seed_node_paths(
    spark_ctx,
    graph,
    output_path: str,
    path_nodes: DataFrame,
    path_edges: DataFrame
) -> None:
    """
    Assemble paths per seed node without joining to pairs.

    Groups path nodes and edges back into arrays, merges them to create
    the paths with feat_ids for each seed node.

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with partition columns
        output_path: Base output path (saves to /seed_node_paths)
        path_nodes: Path nodes DataFrame with feat_ids (from extract_path_elements)
        path_edges: Path edges DataFrame with feat_ids (from extract_path_elements)

    Output:
        Saves to {output_path}/seed_node_paths with schema:
        - node_0_id, seed_node_side, path_id, hop_k, partition_columns
        - node_type_array, node_feat_id_array
        - edge_type_array, u_index_array, v_index_array, edge_feat_id_array
    """
    seed_node_paths_output = f"{output_path}/seed_node_paths"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(seed_node_paths_output)
    if is_complete:
        logger.info(f"Seed node paths already exists at {seed_node_paths_output}, skipping")
        return

    logger.info("\n" + "="*70)
    logger.info("Assembling seed node paths")
    logger.info("="*70)

    # Step 1: Collect nodes back to arrays grouped by path
    logger.info("\nStep 1: Collecting nodes into arrays")
    logger.info("-"*70)

    node_feat_id_df = path_nodes.groupBy(
        'node_0_id', 'seed_node_side', 'path_id', 'hop_k', *graph.partition_columns
    ).agg(
        F.sort_array(F.collect_list(F.struct('pos', 'node_type', 'feat_id'))).alias('node_structs')
    ).select(
        'node_0_id',
        'seed_node_side',
        'path_id',
        'hop_k',
        F.col('node_structs.node_type').alias('node_type_array'),
        F.col('node_structs.feat_id').alias('node_feat_id_array'),
        *graph.partition_columns
    )

    node_feat_id_df = spark_ctx.persist_manager.persist(
        node_feat_id_df,
        release_point='seed_node_paths_done',
        name='node_feat_id_df'
    )
    logger.info(f"  Collected nodes into arrays")

    # Step 2: Collect edges back to arrays grouped by path
    logger.info("\nStep 2: Collecting edges into arrays")
    logger.info("-"*70)

    edge_feat_id_df = path_edges.groupBy(
        'node_0_id', 'seed_node_side', 'path_id', 'hop_k', *graph.partition_columns
    ).agg(
        F.sort_array(F.collect_list(F.struct('pos', 'edge_type', 'u_index', 'v_index', 'feat_id'))).alias('edge_structs')
    ).select(
        'node_0_id',
        'seed_node_side',
        'path_id',
        'hop_k',
        F.col('edge_structs.edge_type').alias('edge_type_array'),
        F.col('edge_structs.u_index').alias('u_index_array'),
        F.col('edge_structs.v_index').alias('v_index_array'),
        F.col('edge_structs.feat_id').alias('edge_feat_id_array'),
        *graph.partition_columns
    )

    edge_feat_id_df = spark_ctx.persist_manager.persist(
        edge_feat_id_df,
        release_point='seed_node_paths_done',
        name='edge_feat_id_df'
    )
    logger.info(f"  Collected edges into arrays")

    # Step 3: Merge node and edge arrays
    logger.info("\nStep 3: Merging node and edge arrays")
    logger.info("-"*70)

    merge_cols = ['node_0_id', 'seed_node_side', 'path_id', 'hop_k'] + graph.partition_columns
    seed_node_paths = node_feat_id_df.join(
        edge_feat_id_df,
        on=merge_cols,
        how='inner'
    ).drop('path_id')

    logger.info(f"  Merged arrays")

    # Step 4: Save output
    logger.info("\nStep 4: Saving seed node paths")
    logger.info("-"*70)

    col_sizes = {
        'node_type_array': 'hop_k * 4',
        'node_feat_id_array': 'hop_k * 8',
        'edge_type_array': 'hop_k * 4',
        'u_index_array': 'hop_k * 4',
        'v_index_array': 'hop_k * 4',
        'edge_feat_id_array': 'hop_k * 8'
    }
    write_table(spark_ctx, seed_node_paths, seed_node_paths_output, mode='overwrite', col_sizes=col_sizes)

    logger.info(f"  Saved to {seed_node_paths_output}")

    # Mark as complete
    spark_ctx.table_state.mark_complete(seed_node_paths_output)

    # Release persisted DataFrames
    spark_ctx.persist_manager.mark_released('seed_node_paths_done')

    logger.info(f"\n{'='*70}")
    logger.info(f"Seed node paths assembly completed")
    logger.info(f"{'='*70}")


def get_pair_bipaths(
    spark_ctx,
    graph,
    output_path: str,
    max_hop: int,
    nodes_feat_id_df: DataFrame,
    edges_feat_id_df: DataFrame
) -> None:
    """
    Assemble pair-to-bipaths mapping from bipath instances with feat_ids.

    Reads bipaths from {output_path}/bipaths/hop_{hop_k} for each hop, converts
    to array format, maps intermediate nodes and edges to feat_ids, and saves
    the result.

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with partition columns
        output_path: Base output path (reads from /bipaths, saves to /pair_bipaths)
        max_hop: Maximum hop number in bipaths
        nodes_feat_id_df: Node feat_id mapping (node_id, node_type, feat_id, partition_columns)
        edges_feat_id_df: Edge feat_id mapping (u_node_id, v_node_id, edge_type, feat_id, partition_columns)

    Output:
        Saves to {output_path}/pair_bipaths with schema:
        - u_node_id, v_node_id, hop_k, path_id
        - node_type_array, node_feat_id_array (intermediate nodes, may be empty for 1-hop)
        - edge_type_array, u_index_array, v_index_array, edge_feat_id_array
        - partition_columns
    """
    final_output_path = f"{output_path}/pair_bipaths"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(final_output_path)
    if is_complete:
        logger.info(f"Pair bipaths already exists at {final_output_path}, skipping")
        return

    logger.info("\n" + "="*70)
    logger.info("Assembling pair-to-bipaths mapping")
    logger.info("="*70)

    bipaths_base_path = f"{output_path}/bipaths"

    # Step 1: Read bipaths and convert to array format
    logger.info("\nStep 1: Reading bipaths and converting to array format")
    logger.info("-"*70)

    bipath_df_list = []

    # Bipaths start from hop_k=1
    for hop_k in range(1, max_hop + 1):
        hop_path = f"{bipaths_base_path}/hop_{hop_k}"
        logger.info(f"  Reading hop_{hop_k} from: {hop_path}")

        # Check if this hop exists
        if not spark_ctx.fileio.exists(hop_path):
            logger.info(f"  Skipping hop_{hop_k} (not found)")
            continue

        bipath_df = read_table(spark_ctx, hop_path)

        # Convert columnar format to arrays
        node_id_cols = [F.col(f'node_{i}_id') for i in range(hop_k + 1)]
        node_type_cols = [F.col(f'node_{i}_type') for i in range(hop_k + 1)]
        edge_type_cols = [F.col(f'edge_{i}_type') for i in range(hop_k)]
        u_index_cols = [F.col(f'u_index_of_edge_{i}') for i in range(hop_k)]
        v_index_cols = [F.col(f'v_index_of_edge_{i}') for i in range(hop_k)]

        bipath_df = bipath_df.select(
            'node_0_id',
            F.col(f'node_{hop_k}_id').alias('node_k_id'),
            F.lit(hop_k).alias('hop_k'),
            F.array(*node_id_cols).alias('node_id_array'),
            F.array(*node_type_cols).alias('node_type_array'),
            F.array(*edge_type_cols).alias('edge_type_array'),
            F.array(*u_index_cols).alias('u_index_array'),
            F.array(*v_index_cols).alias('v_index_array'),
            *[F.col(c) for c in graph.partition_columns]
        )

        bipath_df_list.append(bipath_df)
        logger.info(f"  Processed hop_{hop_k}")

    if not bipath_df_list:
        logger.warning("No bipaths found, skipping merge")
        return

    # Union all hops
    all_bipaths_df = bipath_df_list[0]
    for df in bipath_df_list[1:]:
        all_bipaths_df = all_bipaths_df.unionByName(df)

    logger.info(f"Combined {len(bipath_df_list)} hops")

    # Step 2: Add path_id
    logger.info("\nStep 2: Adding path_id")
    logger.info("-"*70)

    window_spec = Window.partitionBy('node_0_id', 'node_k_id', *graph.partition_columns).orderBy(F.rand())
    all_bipaths_df = all_bipaths_df.withColumn('path_id', F.row_number().over(window_spec) - 1)

    all_bipaths_df = spark_ctx.persist_manager.persist(
        all_bipaths_df,
        release_point='merge_bipaths_done',
        name='all_bipaths_df'
    )
    logger.info(f"Added path_id")

    # Key columns for grouping
    key_cols = ['node_0_id', 'node_k_id', 'path_id', 'hop_k'] + graph.partition_columns

    # Step 3: Extract intermediate nodes and map to feat_ids
    logger.info("\nStep 3: Extracting intermediate nodes and mapping to feat_ids")
    logger.info("-"*70)

    # Slice arrays to get positions [1, hop_k-1] (exclude position 0=u_node and position hop_k=v_node)
    sliced_nodes_df = all_bipaths_df.select(
        *key_cols,
        F.slice('node_id_array', 2, F.col('hop_k') - 1).alias('node_id_array'),
        F.slice('node_type_array', 2, F.col('hop_k') - 1).alias('node_type_array')
    )

    # Explode and extract node elements
    path_nodes = sliced_nodes_df.select(
        *key_cols,
        F.posexplode(F.arrays_zip('node_id_array', 'node_type_array')).alias('pos', 'node_struct')
    ).select(
        *key_cols, 'pos',
        F.col('node_struct.node_id_array').alias('node_id'),
        F.col('node_struct.node_type_array').alias('node_type')
    )

    # Map nodes to feat_ids
    node_join_cols = ['node_id', 'node_type'] + graph.partition_columns
    path_nodes = path_nodes.join(
        nodes_feat_id_df,
        on=node_join_cols,
        how='left'
    ).drop('node_id')

    path_nodes = spark_ctx.persist_manager.persist(
        path_nodes,
        release_point='merge_bipaths_done',
        name='path_nodes_bipath'
    )
    logger.info(f"Extracted and mapped intermediate nodes")

    # Step 4: Extract edges and map to feat_ids
    logger.info("\nStep 4: Extracting edges and mapping to feat_ids")
    logger.info("-"*70)

    path_edges = all_bipaths_df.select(
        *key_cols,
        'node_id_array',
        F.posexplode(F.arrays_zip('edge_type_array', 'u_index_array', 'v_index_array')).alias('pos', 'edge_struct')
    ).select(
        *key_cols, 'pos',
        F.element_at('node_id_array', F.col('edge_struct.u_index_array') + 1).alias('u_node_id'),
        F.element_at('node_id_array', F.col('edge_struct.v_index_array') + 1).alias('v_node_id'),
        F.col('edge_struct.edge_type_array').alias('edge_type'),
        F.col('edge_struct.u_index_array').alias('u_index'),
        F.col('edge_struct.v_index_array').alias('v_index')
    )

    # Map edges to feat_ids
    edge_join_cols = ['u_node_id', 'v_node_id', 'edge_type'] + graph.partition_columns
    path_edges = path_edges.join(
        edges_feat_id_df,
        on=edge_join_cols,
        how='inner'
    ).drop('u_node_id', 'v_node_id')

    path_edges = spark_ctx.persist_manager.persist(
        path_edges,
        release_point='merge_bipaths_done',
        name='path_edges_bipath'
    )
    logger.info(f"Extracted and mapped edges")

    # Step 5: Collect back to arrays and merge
    logger.info("\nStep 5: Collecting back to arrays and merging")
    logger.info("-"*70)

    # Collect nodes back to arrays
    bipath_node_arrays = path_nodes.groupBy(*key_cols).agg(
        F.sort_array(F.collect_list(F.struct('pos', 'node_type', 'feat_id'))).alias('node_structs')
    ).select(
        *key_cols,
        F.col('node_structs.node_type').alias('node_type_array'),
        F.col('node_structs.feat_id').alias('node_feat_id_array')
    )
    logger.info(f"Collected nodes into arrays")

    # Collect edges back to arrays
    bipath_edge_arrays = path_edges.groupBy(*key_cols).agg(
        F.sort_array(F.collect_list(F.struct('pos', 'edge_type', 'u_index', 'v_index', 'feat_id'))).alias('edge_structs')
    ).select(
        *key_cols,
        F.col('edge_structs.edge_type').alias('edge_type_array'),
        F.col('edge_structs.u_index').alias('u_index_array'),
        F.col('edge_structs.v_index').alias('v_index_array'),
        F.col('edge_structs.feat_id').alias('edge_feat_id_array')
    )
    logger.info(f"Collected edges into arrays")

    # Define output columns
    edge_array_cols = ['edge_type_array', 'u_index_array', 'v_index_array', 'edge_feat_id_array']

    # Merge and select final columns with path_type=2
    bipaths_with_feat_ids = bipath_edge_arrays.join(
        bipath_node_arrays, on=key_cols, how='left'
    ).select(
        F.col('node_0_id').alias('u_node_id'),
        F.col('node_k_id').alias('v_node_id'),
        *graph.partition_columns,
        'hop_k',
        *edge_array_cols,
        F.lit(2).alias('path_type'),
        F.coalesce(F.col('node_type_array'), F.array().cast('array<int>')).alias('node_type_array'),
        F.coalesce(F.col('node_feat_id_array'), F.array().cast('array<bigint>')).alias('node_feat_id_array')
    )

    logger.info(f"Merged node and edge arrays with path_type=2")

    # Step 6: Save output
    logger.info("\nStep 6: Saving pair-to-bipaths mapping")
    logger.info("-"*70)

    col_sizes = {
        'node_type_array': '(hop_k - 1) * 4',
        'node_feat_id_array': '(hop_k - 1) * 8',
        'edge_type_array': 'hop_k * 4',
        'u_index_array': 'hop_k * 4',
        'v_index_array': 'hop_k * 4',
        'edge_feat_id_array': 'hop_k * 8'
    }
    write_table(spark_ctx, bipaths_with_feat_ids, final_output_path, mode='overwrite', col_sizes=col_sizes)

    logger.info(f"Saved to {final_output_path}")

    # Mark as complete
    spark_ctx.table_state.mark_complete(final_output_path)

    # Release persisted DataFrames
    spark_ctx.persist_manager.mark_released('merge_bipaths_done')

    logger.info(f"\n{'='*70}")
    logger.info(f"Pair-to-bipaths assembly completed")
    logger.info(f"{'='*70}")


def prepare_collect(
    spark_ctx,
    graph,
    bilink_config: dict,
    output_path: str,
    max_hop: int,
    target_paths_df: DataFrame,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> None:
    """
    Prepare data for batch bipath collection.

    Pre-computes data that can be processed in full (by seed node dimension)
    to avoid redundant computation during batch processing.

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with node/edge type mappings and partition columns
        bilink_config: BiLink configuration with target_edge info
        output_path: Base output path for prepared data
        max_hop: Maximum hop number in bipaths
        target_paths_df: DataFrame with selected target path types
        path_node_type_indices: Set of node type indices in target paths
        path_edge_type_indices: Set of edge type indices in target paths

    Output:
        {output_path}/
        ├── seed_node_elements/   (feat_id arrays per seed node + element_type)
        ├── seed_node_paths/      (path arrays per seed node, one row per path)
        └── pair_bipaths/         (bipath arrays per node pair)
    """
    final_output_path = f"{output_path}/pair_bipaths"

    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(final_output_path)
    if is_complete:
        logger.info(f"Prepare collect already complete (pair_bipaths exists at {final_output_path}), skipping")
        return

    logger.info("\n" + "="*70)
    logger.info("Preparing collection data")
    logger.info("="*70)

    # Step 1: Create feat_id mappings
    logger.info("\nStep 1: Creating feat_id mappings")
    logger.info("-"*70)

    nodes_feat_id_df, edges_feat_id_df = create_feat_id_mappings(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=output_path,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )

    logger.info("  Feat ID mappings created")

    # Step 2: Save seed node feature IDs
    logger.info("\nStep 2: Saving seed node feature IDs")
    logger.info("-"*70)

    seed_node_feat_path = f"{output_path}/seed_node_feat"

    is_complete, _ = spark_ctx.table_state.check_complete(seed_node_feat_path)
    if not is_complete:
        hop_0_path = f"{output_path}/path_exploration/hop_0"
        hop_0_df = read_table(spark_ctx, hop_0_path)

        seed_nodes = hop_0_df.select(
            F.col('node_0_id'),
            F.col('node_0_type').alias('node_type'),
            'seed_node_side',
            *graph.partition_columns
        )

        join_cols = ['node_0_id', 'node_type'] + graph.partition_columns
        seed_node_feat_df = seed_nodes.join(
            nodes_feat_id_df.withColumnRenamed('node_id', 'node_0_id'),
            on=join_cols,
            how='inner'
        )

        write_table(spark_ctx, seed_node_feat_df, seed_node_feat_path, mode='overwrite')
        spark_ctx.table_state.mark_complete(seed_node_feat_path)
        logger.info(f"  Saved to {seed_node_feat_path}")
    else:
        logger.info(f"  Already exists at {seed_node_feat_path}, skipping")

    # Step 3: Extract path elements and map to feat_ids
    logger.info("\nStep 3: Extracting path elements and mapping to feat_ids")
    logger.info("-"*70)

    path_nodes, path_edges = extract_path_elements(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=output_path,
        target_paths_df=target_paths_df,
        max_hop=max_hop,
        nodes_feat_id_df=nodes_feat_id_df,
        edges_feat_id_df=edges_feat_id_df
    )

    logger.info("  Path elements extracted and mapped to feat_ids")

    # Step 4: Get seed node elements
    logger.info("\nStep 4: Extracting seed node elements")
    logger.info("-"*70)

    get_seed_node_elements(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=output_path,
        path_nodes=path_nodes,
        path_edges=path_edges,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )

    logger.info("  Seed node elements extracted and saved")

    # Step 5: Get seed node paths
    logger.info("\nStep 5: Assembling seed node paths")
    logger.info("-"*70)

    get_seed_node_paths(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=output_path,
        path_nodes=path_nodes,
        path_edges=path_edges
    )

    logger.info("  Seed node paths assembled and saved")

    # Step 6: Get pair bipaths
    logger.info("\nStep 6: Assembling pair bipaths")
    logger.info("-"*70)

    get_pair_bipaths(
        spark_ctx=spark_ctx,
        graph=graph,
        output_path=output_path,
        max_hop=max_hop,
        nodes_feat_id_df=nodes_feat_id_df,
        edges_feat_id_df=edges_feat_id_df
    )

    logger.info("  Pair bipaths assembled and saved")

    logger.info("\n" + "="*70)
    logger.info("Prepare collect completed successfully")
    logger.info(f"Output saved to: {output_path}")
    logger.info("="*70)


__all__ = ['prepare_collect', 'get_seed_node_elements', 'get_seed_node_paths', 'get_pair_bipaths']
