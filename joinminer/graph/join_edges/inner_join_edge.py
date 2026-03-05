"""
Inner Join Edge Generation

This module generates inner join edges by unifying multiple edge types into a
standardized schema. All edges are preserved (no sampling).

Schema:
    - u_node_id: Source node ID
    - v_node_id: Target node ID
    - u_node_type: Source node type index
    - v_node_type: Target node type index
    - edge_type: Edge type index
    - partition_columns: Partition columns from graph
"""

import logging
from typing import List
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

from joinminer.spark.io import read_table, write_table

logger = logging.getLogger(__name__)


def generate_inner_join_edge(
    spark_ctx,
    graph,
    join_edge_path: str,
    partition_instances: List[List[str]]
) -> None:
    """
    Generate inner join edges by unifying all edge types.

    This function reads all edge types from the graph and transforms them into
    a unified schema, then unions them together. No sampling is performed - all
    edges are preserved.

    Args:
        spark_ctx: SparkContext instance for Spark operations
        graph: Graph object containing edge definitions and type indices
        join_edge_path: Root directory for join edges
        partition_instances: List of partition instances to process

    Output:
        Saves unified edges to: {join_edge_path}/inner/

    Schema:
        - u_node_id: Source node ID
        - v_node_id: Target node ID
        - u_node_type: Source node type index (from graph.nodes[type]['type_index'])
        - v_node_type: Target node type index
        - edge_type: Edge type index (from graph.edges[type]['type_index'])
        - partition columns (if any)

    Example:
        >>> generate_inner_join_edge(
        ...     spark_ctx=spark_ctx,
        ...     graph=graph,
        ...     join_edge_path="hdfs:///data/join_edges",
        ...     partition_instances=[['2024-01-01']]
        ... )
    """
    output_path = f"{join_edge_path}/inner"
    partition_columns = graph.partition_columns

    # Check completion status
    is_complete, missing_partitions = spark_ctx.table_state.check_complete(
        output_path,
        partition_columns,
        partition_instances
    )

    if is_complete:
        logger.info(f"Inner join edges already complete at {output_path}")
        return

    logger.info(f"Generating inner join edges for {len(missing_partitions)} missing partition(s)")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Edge types to union: {list(graph.edges.keys())}")

    # Collect transformed edge DataFrames from all edge types
    edge_dfs = []

    # Process each edge type
    for edge_type, edge_info in graph.edges.items():
        # Read source edge table for all missing partitions
        df = read_table(
            spark_ctx,
            path=edge_info['table_path'],
            partition_columns=partition_columns,
            partition_instances=missing_partitions
        )

        # Get node type information
        u_node_type = edge_info['u_node']['type']
        v_node_type = edge_info['v_node']['type']
        u_node_id_col = edge_info['u_node']['id_column']
        v_node_id_col = edge_info['v_node']['id_column']

        # Get type indices
        u_node_type_index = graph.nodes[u_node_type]['type_index']
        v_node_type_index = graph.nodes[v_node_type]['type_index']
        edge_type_index = edge_info['type_index']

        # Transform to unified schema
        select_exprs = [
            col(u_node_id_col).alias('u_node_id'),
            col(v_node_id_col).alias('v_node_id'),
            lit(u_node_type_index).alias('u_node_type'),
            lit(v_node_type_index).alias('v_node_type'),
            lit(edge_type_index).alias('edge_type')
        ]

        # Add partition columns
        for part_col in partition_columns:
            select_exprs.append(col(part_col))

        df_transformed = df.select(*select_exprs)
        edge_dfs.append(df_transformed)

    # Check if we have any edge data
    if len(edge_dfs) == 0:
        raise ValueError(
            f"No edge data found for partitions {missing_partitions}. "
            f"Check if source edge tables exist and contain data."
        )

    # Union all edge DataFrames
    logger.info(f"Unioning {len(edge_dfs)} edge types")
    union_df = reduce(DataFrame.union, edge_dfs)

    # Write to output
    write_table(
        spark_ctx,
        df=union_df,
        path=output_path,
        partition_columns=partition_columns,
        partition_instances=missing_partitions,
        mode='overwrite'
    )

    # Mark as complete
    spark_ctx.table_state.mark_complete(
        output_path,
        partition_columns,
        missing_partitions
    )

    logger.info(f"Inner join edge generation completed for {len(missing_partitions)} partition(s)")
