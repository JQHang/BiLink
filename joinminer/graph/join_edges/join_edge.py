"""
Join Edge Generation

This module provides the main orchestrator function for generating join edges.
"""

import logging
from typing import List

from .inner_join_edge import generate_inner_join_edge
from .outer_join_edge import generate_outer_join_edge

logger = logging.getLogger(__name__)


def generate_join_edges(
    spark_ctx,
    graph,
    join_edge_path: str,
    partition_instances: List[List[str]],
    max_neighbor: int = 100
) -> None:
    """
    Generate join edges based on graph configuration.

    This function orchestrates the generation of both inner and outer join edges:
    1. Inner join edge: Union of all edge types into unified schema
    2. Outer join edge: Sampled neighborhoods from inner join edges

    Args:
        spark_ctx: SparkContext instance for Spark operations
        graph: Graph object containing edge definitions and type indices
        join_edge_path: Root directory for join edges (with URI scheme)
                       Results will be saved to:
                       - {join_edge_path}/inner/
                       - {join_edge_path}/outer/
        partition_instances: List of partition instances to process
                            e.g., [['2024-01-01'], ['2024-01-02']]
        max_neighbor: Maximum number of neighbors to sample per
                     (join_node, edge_type) group in outer join edge.
                     Default: 100

    Example:
        >>> from joinminer.graph.join_edges import generate_join_edges
        >>> generate_join_edges(
        ...     spark_ctx=spark_ctx,
        ...     graph=graph,
        ...     join_edge_path="hdfs:///data/join_edges",
        ...     partition_instances=[['2024-01-01']],
        ...     max_neighbor=100
        ... )

    Notes:
        - Inner join edge schema: u_node_id, v_node_id, u_node_type,
          v_node_type, edge_type, partition_columns
        - Outer join edge schema: join_node_id, add_node_id, join_node_type,
          add_node_type, edge_type, join_node_side, partition_columns
        - All type columns store type_index (integer) rather than type names
    """
    logger.info("=" * 70)
    logger.info("Starting join edge generation")
    logger.info(f"Output path: {join_edge_path}")
    logger.info(f"Partitions to process: {len(partition_instances)}")
    logger.info(f"Max neighbor sampling: {max_neighbor}")
    logger.info("=" * 70)

    # Step 1: Generate inner join edge (union all edges)
    logger.info("\n[Step 1/2] Generating inner join edges...")
    generate_inner_join_edge(
        spark_ctx=spark_ctx,
        graph=graph,
        join_edge_path=join_edge_path,
        partition_instances=partition_instances
    )
    logger.info("Inner join edges generated successfully")

    # Step 2: Generate outer join edge (sample neighborhoods)
    logger.info("\n[Step 2/2] Generating outer join edges...")
    generate_outer_join_edge(
        spark_ctx=spark_ctx,
        graph=graph,
        join_edge_path=join_edge_path,
        partition_instances=partition_instances,
        max_neighbor=max_neighbor
    )
    logger.info("Outer join edges generated successfully")

    logger.info("=" * 70)
    logger.info("Join edge generation completed")
    logger.info("=" * 70)
