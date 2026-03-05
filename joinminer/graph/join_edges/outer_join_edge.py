"""
Outer Join Edge Generation

This module generates outer join edges by sampling neighborhoods from inner join edges.
Each edge is transformed from two perspectives (u_node and v_node as join_node),
allowing efficient neighborhood exploration.

Schema:
    - join_node_id: Node used for joining
    - add_node_id: Neighboring node to add
    - join_node_type: Join node type index
    - add_node_type: Add node type index
    - edge_type: Edge type index
    - join_node_side: Which side the join_node comes from (0 for u_node, 1 for v_node)
    - partition_columns: Partition columns from graph
"""

import logging
from typing import List

from pyspark.sql.functions import col, lit

from joinminer.spark.io import read_table, write_table
from joinminer.spark.operations.sample import skewed_random_sample

logger = logging.getLogger(__name__)


def generate_outer_join_edge(
    spark_ctx,
    graph,
    join_edge_path: str,
    partition_instances: List[List[str]],
    max_neighbor: int = 100
) -> None:
    """
    Generate outer join edges by sampling neighborhoods from inner join edges.

    This function reads inner join edges and transforms them from two perspectives:
    - From u_node perspective: u_node becomes join_node, v_node becomes add_node
    - From v_node perspective: v_node becomes join_node, u_node becomes add_node

    For each (join_node_id, join_node_type, edge_type, join_node_side) group,
    up to max_neighbor neighbors are sampled.

    Args:
        spark_ctx: SparkContext instance for Spark operations
        graph: Graph object containing partition columns
        join_edge_path: Root directory for join edges
        partition_instances: List of partition instances to process
        max_neighbor: Maximum number of neighbors to sample per
                     (join_node_id, join_node_type, edge_type, join_node_side) group.
                     Default: 100

    Output:
        Saves sampled edges to: {join_edge_path}/outer/

    Schema:
        - join_node_id: Node used for joining
        - add_node_id: Neighboring node to add
        - join_node_type: Join node type index
        - add_node_type: Add node type index
        - edge_type: Edge type index
        - join_node_side: 0 (from u_node) or 1 (from v_node)
        - partition columns (if any)

    Example:
        >>> generate_outer_join_edge(
        ...     spark_ctx=spark_ctx,
        ...     graph=graph,
        ...     join_edge_path="hdfs:///data/join_edges",
        ...     partition_instances=[['2024-01-01']],
        ...     max_neighbor=100
        ... )
    """
    inner_path = f"{join_edge_path}/inner"
    output_path = f"{join_edge_path}/outer"
    partition_columns = graph.partition_columns

    # Check completion status
    is_complete, missing_partitions = spark_ctx.table_state.check_complete(
        output_path,
        partition_columns,
        partition_instances
    )

    if is_complete:
        logger.info(f"Outer join edges already complete at {output_path}")
        return

    logger.info(f"Generating outer join edges for {len(missing_partitions)} missing partition(s)")
    logger.info(f"Input path: {inner_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Max neighbors per group: {max_neighbor}")

    # Read inner join edges for all missing partitions
    df = read_table(
        spark_ctx,
        path=inner_path,
        partition_columns=partition_columns,
        partition_instances=missing_partitions
    )

    # Perspective 1: u_node as join_node
    logger.info("Transforming from u_node perspective...")
    df_u = df.select(
        col('u_node_id').alias('join_node_id'),
        col('v_node_id').alias('add_node_id'),
        col('u_node_type').alias('join_node_type'),
        col('v_node_type').alias('add_node_type'),
        col('edge_type'),
        lit(0).alias('join_node_side'),
        *[col(c) for c in partition_columns]
    )

    # Perspective 2: v_node as join_node
    logger.info("Transforming from v_node perspective...")
    df_v = df.select(
        col('v_node_id').alias('join_node_id'),
        col('u_node_id').alias('add_node_id'),
        col('v_node_type').alias('join_node_type'),
        col('u_node_type').alias('add_node_type'),
        col('edge_type'),
        lit(1).alias('join_node_side'),
        *[col(c) for c in partition_columns]
    )

    # Union both perspectives first
    logger.info("Unioning both perspectives...")
    union_df = df_u.union(df_v)

    # Then sample once with all necessary grouping columns
    logger.info("Sampling neighborhoods...")
    group_columns = ['join_node_id', 'join_node_type', 'edge_type', 'join_node_side'] + partition_columns
    sampled_df = skewed_random_sample(
        spark_ctx=spark_ctx,
        df=union_df,
        group_columns=group_columns,
        n=max_neighbor,
        release_point='outer_join_edge_sampling'
    )

    # Write to output
    write_table(
        spark_ctx,
        df=sampled_df,
        path=output_path,
        partition_columns=partition_columns,
        partition_instances=missing_partitions,
        mode='overwrite'
    )

    # Release persisted sampling data
    spark_ctx.persist_manager.mark_released('outer_join_edge_sampling')

    # Mark as complete
    spark_ctx.table_state.mark_complete(
        output_path,
        partition_columns,
        missing_partitions
    )

    logger.info(f"Outer join edge generation completed for {len(missing_partitions)} partition(s)")
