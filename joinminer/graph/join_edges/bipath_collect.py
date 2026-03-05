"""
Bipath Collection with Pre-computed Data

This module provides functions for collecting bipaths using pre-computed
seed node data for efficient batch processing.

Key Functions:
- bipaths_collection: Main orchestration function for batch bipath collection
- _collect_pair_element_features: Collect element features per pair
- _collect_pair_all_paths: Collect all paths and bipaths per pair
- _assemble_final_output: Assemble final output with all features
"""

import logging
from typing import Set, Tuple
from pyspark.sql import DataFrame, functions as F

from joinminer.spark.io import write_table

# Module-level logger
logger = logging.getLogger("joinminer")


def _collect_pair_all_paths(
    spark_ctx,
    graph,
    batch_pairs_df: DataFrame,
    seed_node_paths_df: DataFrame,
    pair_bipaths_df: DataFrame
) -> Tuple[DataFrame, dict]:
    """
    Collect all paths and bipaths for each pair in the batch.

    Args:
        spark_ctx: Spark context with persist_manager
        graph: Graph instance with partition columns
        batch_pairs_df: Current batch pairs (u_node_id, v_node_id, partition_columns)
        seed_node_paths_df: Pre-computed seed node paths
        pair_bipaths_df: Pre-computed pair bipaths (with path_type=2)

    Returns:
        Tuple of (DataFrame, col_sizes dict) where DataFrame contains collected paths
        (u_node_id, v_node_id, partition_columns, path_count, path_type_array, ...)
    """
    logger.info("\nPhase 1: Collecting pair all paths")
    logger.info("-"*70)

    pair_cols = ['u_node_id', 'v_node_id'] + graph.partition_columns

    # Step 1: Expand batch_pairs to u (path_type=0) and v (path_type=1) sides
    logger.info("\n  Step 1.1: Expanding batch pairs with path_type")

    batch_pairs_u = batch_pairs_df.select(
        *pair_cols,
        F.lit(0).alias('path_type'),
        F.col('u_node_id').alias('node_0_id')
    )
    batch_pairs_v = batch_pairs_df.select(
        *pair_cols,
        F.lit(1).alias('path_type'),
        F.col('v_node_id').alias('node_0_id')
    )
    expanded_pairs = batch_pairs_u.unionByName(batch_pairs_v)

    # Step 2: Join with seed_node_paths (rename seed_node_side to path_type)
    logger.info("\n  Step 1.2: Joining with seed node paths")

    # seed_node_paths_df has: node_0_id, seed_node_side, hop_k, arrays..., partition_columns
    seed_paths_renamed = seed_node_paths_df.withColumnRenamed('seed_node_side', 'path_type')

    join_cols = ['node_0_id', 'path_type'] + graph.partition_columns
    pair_paths = expanded_pairs.join(seed_paths_renamed, on=join_cols, how='inner')
    pair_paths = pair_paths.drop('node_0_id')

    logger.info(f"    Joined pair paths")

    # Step 3: Filter pair_bipaths for current batch pairs
    logger.info("\n  Step 1.3: Filtering pair bipaths for current batch")

    batch_bipaths = batch_pairs_df.join(pair_bipaths_df, on=pair_cols, how='inner')

    logger.info(f"    Filtered batch bipaths")

    # Step 4: Union paths and bipaths
    logger.info("\n  Step 1.4: Unioning paths and bipaths")

    # Ensure same column order for unionByName
    path_data_cols = [
        'hop_k', 'path_type',
        'node_type_array', 'node_feat_id_array',
        'edge_type_array', 'u_index_array', 'v_index_array', 'edge_feat_id_array'
    ]

    pair_paths_selected = pair_paths.select(*pair_cols, *path_data_cols)
    batch_bipaths_selected = batch_bipaths.select(*pair_cols, *path_data_cols)

    all_paths = pair_paths_selected.unionByName(batch_bipaths_selected)

    logger.info(f"    Unioned paths and bipaths")

    # Step 5: Collect all paths by pair
    logger.info("\n  Step 1.5: Collecting paths by pair")

    collected_paths = all_paths.groupBy(*pair_cols).agg(
        F.count('*').alias('path_count'),
        F.collect_list(F.struct(*path_data_cols)).alias('path_struct_array')
    )

    # Extract individual arrays from struct array
    pair_all_paths = collected_paths.select(
        *pair_cols,
        'path_count',
        F.col('path_struct_array.path_type').alias('path_type_array'),
        F.col('path_struct_array.hop_k').alias('hop_k_array'),
        F.col('path_struct_array.node_type_array').alias('paths_node_type_array'),
        F.col('path_struct_array.node_feat_id_array').alias('paths_node_feat_id_array'),
        F.col('path_struct_array.edge_type_array').alias('paths_edge_type_array'),
        F.col('path_struct_array.u_index_array').alias('paths_u_index_array'),
        F.col('path_struct_array.v_index_array').alias('paths_v_index_array'),
        F.col('path_struct_array.edge_feat_id_array').alias('paths_edge_feat_id_array')
    )

    logger.info("  Phase 1 completed")

    col_sizes = {
        'path_type_array': 'path_count * 4',
        'hop_k_array': 'path_count * 4',
        'paths_node_type_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 4",
        'paths_node_feat_id_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 8",
        'paths_edge_type_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 4",
        'paths_u_index_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 4",
        'paths_v_index_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 4",
        'paths_edge_feat_id_array': "aggregate(hop_k_array, 0, (a,x) -> a+x) * 8",
    }

    return pair_all_paths, col_sizes

def _collect_pair_element_features(
    spark_ctx,
    graph,
    batch_pairs_df: DataFrame,
    seed_node_elements_df: DataFrame,
    element_feat_df: DataFrame,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> Tuple[DataFrame, dict]:
    """
    Collect element features for each pair in the batch.

    Args:
        spark_ctx: Spark context with persist_manager
        graph: Graph instance with node/edge type mappings
        batch_pairs_df: Current batch pairs (u_node_id, v_node_id, partition_columns)
        seed_node_elements_df: Pre-computed seed node elements
        element_feat_df: Pre-computed element feat_vector mapping (from create_feat_vector_mapping)
        path_node_type_indices: Set of node type indices in target paths
        path_edge_type_indices: Set of edge type indices in target paths

    Returns:
        Tuple of (DataFrame, col_sizes dict) where DataFrame contains pair element features
        (u_node_id, v_node_id, partition_columns, node_*_count/feat_id_array/feat_vector_array, ...)
    """
    logger.info("\nPhase 2: Collecting pair element features")
    logger.info("-"*70)

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

    # Step 1: Expand batch_pairs to u and v sides
    logger.info("\n  Step 2.1: Expanding batch pairs to u/v sides")

    pair_cols = ['u_node_id', 'v_node_id'] + graph.partition_columns

    batch_pairs_u = batch_pairs_df.select(
        *pair_cols,
        F.lit(0).alias('seed_node_side'),
        F.col('u_node_id').alias('node_0_id')
    )
    batch_pairs_v = batch_pairs_df.select(
        *pair_cols,
        F.lit(1).alias('seed_node_side'),
        F.col('v_node_id').alias('node_0_id')
    )
    expanded_pairs = batch_pairs_u.unionByName(batch_pairs_v)

    # Step 2: Join with seed_node_elements
    logger.info("\n  Step 2.2: Joining with seed node elements")

    join_cols = ['node_0_id', 'seed_node_side'] + graph.partition_columns
    pair_elements = expanded_pairs.join(seed_node_elements_df, on=join_cols, how='inner')
    pair_elements = pair_elements.drop('node_0_id', 'seed_node_side')

    logger.info(f"    Joined pair elements")

    # Step 3: Deduplicate by pair + element_type + feat_id
    logger.info("\n  Step 2.3: Deduplicating by pair dimension")

    dedup_cols = pair_cols + ['element_type', 'feat_id']
    pair_elements_dedup = pair_elements.select(dedup_cols).distinct()

    # Step 4: Join with feat_vector mapping
    logger.info("\n  Step 2.4: Joining with feat_vector mapping")

    feat_join_cols = ['element_type', 'feat_id'] + graph.partition_columns
    pair_elements_with_feat = pair_elements_dedup.join(
        element_feat_df,
        on=feat_join_cols,
        how='inner'
    )

    logger.info(f"    Joined with feat_vector")

    # Step 5: Aggregate and pivot by element type
    logger.info("\n  Step 2.5: Aggregating and pivoting by element type")

    pivot_values = []
    for node_type_index in sorted(feat_node_type_indices):
        pivot_values.append(f'0_{node_type_index}')
    for edge_type_index in sorted(feat_edge_type_indices):
        pivot_values.append(f'1_{edge_type_index}')

    pair_elem_feat = pair_elements_with_feat.groupBy(pair_cols).pivot(
        'element_type', pivot_values
    ).agg(
        F.count('feat_id').alias('count'),
        F.collect_list(F.struct('feat_id', 'feat_vector')).alias('feat_struct_array')
    )

    # Rename pivot columns
    select_exprs = [F.col(c) for c in pair_cols]
    col_sizes = {}

    for node_type_index in sorted(feat_node_type_indices):
        node_type_name = graph.node_index_to_type[node_type_index]
        feature_count = graph.nodes[node_type_name]['feature_count']

        # count column
        old_count = f'0_{node_type_index}_count'
        new_count = f'node_{node_type_name}_count'
        select_exprs.append(F.col(old_count).alias(new_count))

        # Extract from struct array (guarantees feat_id and feat_vector alignment)
        old_struct = f'0_{node_type_index}_feat_struct_array'
        select_exprs.append(F.col(f'{old_struct}.feat_id').alias(f'node_{node_type_name}_feat_id_array'))
        select_exprs.append(F.col(f'{old_struct}.feat_vector').alias(f'node_{node_type_name}_feat_vector_array'))

        col_sizes[f'node_{node_type_name}_feat_id_array'] = f'{new_count} * 8'
        col_sizes[f'node_{node_type_name}_feat_vector_array'] = f'{new_count} * {feature_count} * 8'

    for edge_type_index in sorted(feat_edge_type_indices):
        edge_type_name = graph.edge_index_to_type[edge_type_index]
        feature_count = graph.edges[edge_type_name]['feature_count']

        # count column
        old_count = f'1_{edge_type_index}_count'
        new_count = f'edge_{edge_type_name}_count'
        select_exprs.append(F.col(old_count).alias(new_count))

        # Extract from struct array (guarantees feat_id and feat_vector alignment)
        old_struct = f'1_{edge_type_index}_feat_struct_array'
        select_exprs.append(F.col(f'{old_struct}.feat_id').alias(f'edge_{edge_type_name}_feat_id_array'))
        select_exprs.append(F.col(f'{old_struct}.feat_vector').alias(f'edge_{edge_type_name}_feat_vector_array'))

        col_sizes[f'edge_{edge_type_name}_feat_id_array'] = f'{new_count} * 8'
        col_sizes[f'edge_{edge_type_name}_feat_vector_array'] = f'{new_count} * {feature_count} * 8'

    pair_elem_feat = pair_elem_feat.select(*select_exprs)

    # Fill NA for count columns
    count_cols = [f'node_{graph.node_index_to_type[i]}_count' for i in sorted(feat_node_type_indices)]
    count_cols += [f'edge_{graph.edge_index_to_type[i]}_count' for i in sorted(feat_edge_type_indices)]
    pair_elem_feat = pair_elem_feat.fillna(0, subset=count_cols)

    logger.info(f"    Aggregated and pivoted")

    logger.info("  Phase 2 completed")

    return pair_elem_feat, col_sizes

def _assemble_final_output(
    spark_ctx,
    graph,
    sample_config: dict,
    batch_pairs_df: DataFrame,
    sample_pairs_df: DataFrame,
    seed_node_feat_df: DataFrame,
    pair_elem_feat_df: DataFrame,
    pair_all_paths_df: DataFrame,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int]
) -> DataFrame:
    """
    Assemble final output by joining all features.

    Args:
        spark_ctx: Spark context
        graph: Graph instance with partition columns
        sample_config: Sample configuration with partition_columns info
        batch_pairs_df: Current batch pairs
        sample_pairs_df: Full sample pairs with labels
        seed_node_feat_df: Seed node feat_ids
        pair_elem_feat_df: Pair element features
        pair_all_paths_df: Pair collected paths
        path_node_type_indices: Set of node type indices
        path_edge_type_indices: Set of edge type indices

    Returns:
        Final assembled DataFrame
    """
    logger.info("\nPhase 3: Assembling final output")
    logger.info("-"*70)

    pair_cols = ['u_node_id', 'v_node_id'] + graph.partition_columns

    # Step 1: Check if sample_config has partition_columns
    sample_partition_cols = sample_config.get('partition_columns')
    if sample_partition_cols:
        # Join with sample_pairs to get neg_ratio, label, etc. (Cartesian effect)
        logger.info("\n  Step 3.1: Joining with sample pairs")
        result = batch_pairs_df.join(sample_pairs_df, on=pair_cols, how='left')
        logger.info(f"    Joined with sample pairs (partition_columns: {sample_partition_cols})")
    else:
        # No partition columns, skip join
        logger.info("\n  Step 3.1: No sample partition columns, using batch_pairs directly")
        result = batch_pairs_df

    # Step 2: Add u_node_feat_id and v_node_feat_id
    logger.info("\n  Step 3.2: Adding u/v node feat_ids")

    # u side
    u_feat = seed_node_feat_df.filter(F.col('seed_node_side') == 0).select(
        F.col('node_0_id').alias('u_node_id'),
        F.col('feat_id').alias('u_node_feat_id'),
        *graph.partition_columns
    )
    u_join_cols = ['u_node_id'] + graph.partition_columns
    result = result.join(u_feat, on=u_join_cols, how='left')

    # v side
    v_feat = seed_node_feat_df.filter(F.col('seed_node_side') == 1).select(
        F.col('node_0_id').alias('v_node_id'),
        F.col('feat_id').alias('v_node_feat_id'),
        *graph.partition_columns
    )
    v_join_cols = ['v_node_id'] + graph.partition_columns
    result = result.join(v_feat, on=v_join_cols, how='left')

    logger.info(f"    Added u/v node feat_ids")

    # Step 3: Add collected paths (smaller, join first)
    logger.info("\n  Step 3.3: Adding collected paths")

    result = result.join(pair_all_paths_df, on=pair_cols, how='left')

    # Fill path_count with 0
    result = result.withColumn(
        'path_count',
        F.coalesce(F.col('path_count'), F.lit(0))
    )

    logger.info(f"    Added collected paths")

    # Step 4: Add element features (larger, join later)
    logger.info("\n  Step 3.4: Adding element features")

    result = result.join(pair_elem_feat_df, on=pair_cols, how='left')

    # Fill count columns with 0
    feat_node_type_indices = {
        idx for idx in path_node_type_indices
        if graph.nodes[graph.node_index_to_type[idx]]['feature_count'] > 0
    }
    feat_edge_type_indices = {
        idx for idx in path_edge_type_indices
        if graph.edges[graph.edge_index_to_type[idx]]['feature_count'] > 0
    }

    count_cols = [f'node_{graph.node_index_to_type[i]}_count' for i in sorted(feat_node_type_indices)]
    count_cols += [f'edge_{graph.edge_index_to_type[i]}_count' for i in sorted(feat_edge_type_indices)]
    result = result.fillna(0, subset=count_cols)

    logger.info(f"    Added element features")

    logger.info("  Phase 3 completed")

    return result


def bipaths_collection(
    spark_ctx,
    graph,
    bilink_config: dict,
    sample_config: dict,
    batch_pairs_df: DataFrame,
    sample_pairs_df: DataFrame,
    seed_node_feat_df: DataFrame,
    seed_node_elements_df: DataFrame,
    seed_node_paths_df: DataFrame,
    pair_bipaths_df: DataFrame,
    path_node_type_indices: Set[int],
    path_edge_type_indices: Set[int],
    output_path: str,
    element_feat_df: DataFrame = None
) -> None:
    """
    Main orchestration function for collecting bipaths with pre-computed data.

    This function uses pre-computed seed node data to efficiently collect
    bipath features for a batch of pairs.

    Args:
        spark_ctx: Spark context with persist_manager and table_state
        graph: Graph instance with node/edge type mappings and partition columns
        bilink_config: BiLink configuration with target_edge info
        sample_config: Sample configuration with partition_columns info
        batch_pairs_df: Current batch pairs (u_node_id, v_node_id, partition_columns)
        sample_pairs_df: Full sample pairs with labels (neg_ratio, label, etc.)
        seed_node_feat_df: Pre-computed seed node feat_ids
        seed_node_elements_df: Pre-computed seed node elements
        seed_node_paths_df: Pre-computed seed node paths
        pair_bipaths_df: Pre-computed pair bipaths
        path_node_type_indices: Set of node type indices in target paths
        path_edge_type_indices: Set of edge type indices in target paths
        output_path: Output path for collected bipaths
        element_feat_df: Pre-computed element feat_vector mapping (optional, for reuse across batches)

    Output:
        Saves collected bipaths to output_path with all features
    """
    # Check if already complete
    is_complete, _ = spark_ctx.table_state.check_complete(output_path)
    if is_complete:
        logger.info(f"Bipaths collection already complete at {output_path}, skipping")
        return

    logger.info("\n" + "="*70)
    logger.info("Starting bipaths collection for batch")
    logger.info("="*70)

    # Phase 1: Collect pair all paths (smaller result, join first)
    pair_all_paths_df, paths_col_sizes = _collect_pair_all_paths(
        spark_ctx=spark_ctx,
        graph=graph,
        batch_pairs_df=batch_pairs_df,
        seed_node_paths_df=seed_node_paths_df,
        pair_bipaths_df=pair_bipaths_df
    )
    pair_all_paths_df = spark_ctx.persist_manager.persist(
        pair_all_paths_df,
        release_point='batch_collection_done',
        name='pair_all_paths_df'
    )

    # Phase 2: Collect pair element features (larger result, join later)
    pair_elem_feat_df, elem_col_sizes = _collect_pair_element_features(
        spark_ctx=spark_ctx,
        graph=graph,
        batch_pairs_df=batch_pairs_df,
        seed_node_elements_df=seed_node_elements_df,
        element_feat_df=element_feat_df,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )
    pair_elem_feat_df = spark_ctx.persist_manager.persist(
        pair_elem_feat_df,
        release_point='batch_collection_done',
        name='pair_elem_feat_df'
    )

    # Phase 3: Assemble final output
    result_df = _assemble_final_output(
        spark_ctx=spark_ctx,
        graph=graph,
        sample_config=sample_config,
        batch_pairs_df=batch_pairs_df,
        sample_pairs_df=sample_pairs_df,
        seed_node_feat_df=seed_node_feat_df,
        pair_elem_feat_df=pair_elem_feat_df,
        pair_all_paths_df=pair_all_paths_df,
        path_node_type_indices=path_node_type_indices,
        path_edge_type_indices=path_edge_type_indices
    )

    # Merge col_sizes from both phases
    col_sizes = {**paths_col_sizes, **elem_col_sizes}

    # Save output
    logger.info("\nSaving collected bipaths")
    logger.info("-"*70)

    sample_partition_cols = sample_config.get('partition_columns')
    if sample_partition_cols:
        write_table(spark_ctx, result_df, output_path, mode='overwrite',
                    col_sizes=col_sizes, partition_columns=sample_partition_cols)
        logger.info(f"Saved to {output_path} (partitioned by {sample_partition_cols})")
    else:
        write_table(spark_ctx, result_df, output_path, mode='overwrite', col_sizes=col_sizes)
        logger.info(f"Saved to {output_path}")

    # Mark as complete
    spark_ctx.table_state.mark_complete(output_path)

    # Release persisted DataFrames
    spark_ctx.persist_manager.mark_released('batch_collection_done')

    logger.info("\n" + "="*70)
    logger.info("Bipaths collection completed successfully")
    logger.info("="*70)


__all__ = ['bipaths_collection']
